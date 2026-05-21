#!/usr/bin/env python3
"""
Retroactively assign stable sentence-pair IDs and per-alignment IDs + statuses
to the Gothic word-spotting data, producing a canonical alignments JSONL with
full provenance.

Inputs (per split):
    - LLM-original annotations JSONL (e.g. train_annotations.jsonl):
      authoritative source for the _o{N} alignment ordering within each
      sentence.
    - Verification TSV (e.g. train_verified_b.tsv): per-row status assigned
      during manual review (okay / unchecked / delete).
    - Verified JSONL (e.g. train_verified_b.jsonl): post-edit state of okay
      alignments — used to detect surface-form edits (kept_edited vs
      verified_correct).
    - Diversified JSONL (e.g. train_verified_b_diversified.jsonl): post-
      diversification state — used to detect which okay alignments were
      replaced (replaced_in_diversification) and which new ones were added.

Shared inputs:
    - Diversification annotations JSONL (diversify_annotations.jsonl):
      authoritative source for _d{N} ordering on diversification-added
      alignments.
    - Diversification manifest JSON: maps each batch's custom_id to source
      file + focal pair.

Output (per split): <split>_alignments.jsonl with one record per LLM-proposed
sentence, each carrying a `sentence_id` and an `alignments` list where every
alignment has an `alignment_id`, `source`, and `status` (plus an `original`
field if it was edited).

Sentence IDs are assigned in the order sentences appear in the annotations
JSONL (which is the canonical LLM-batch order).

Status taxonomy:
    verified_correct           - okay in TSV, exact surface-form match in
                                 verified_b
    kept_edited                - okay in TSV, surface form differs from LLM
                                 original (the `original` field holds the
                                 pre-edit values)
    unverified                 - 'unchecked' in TSV, or any diversification-
                                 added alignment (not yet re-reviewed)
    rejected                   - 'delete' in TSV
    replaced_in_diversification- okay/kept_edited LLM original that was
                                 removed during the diversification pass
                                 (i.e., not retained)

Usage:
    python -m gothic.word_spotting.assign_alignment_ids \\
        --split train \\
        --annotations data/gothic_word_spotting/train_annotations.jsonl \\
        --verified-tsv data/gothic_word_spotting/train_verified_b.tsv \\
        --verified-jsonl data/gothic_word_spotting/train_verified_b.jsonl \\
        --diversified-jsonl data/gothic_word_spotting/train_verified_b_diversified.jsonl \\
        --diversify-annotations data/gothic_word_spotting/diversify_annotations.jsonl \\
        --diversify-manifest data/gothic_word_spotting/diversify_batch_requests.manifest.json \\
        --output data/gothic_word_spotting/train_alignments.jsonl
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_verification_tsv(
    path: Path,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Parse the verification TSV into a dict keyed by
    (gothic_sentence_roman, gothic_surface) -> list of (english_word, status).

    Keying on (sentence, gothic_surface) rather than including english_word
    tolerates the reviewer editing the english_word cell (e.g., adding a
    parenthetical note explaining why an alignment is bad). gothic_surface
    was not edited during review, so it's a stable matching key.

    A list-of-tuples value preserves duplicates (when the LLM proposed the
    same alignment twice for the same sentence, producing two TSV rows).

    Uses the same reader settings as verify_word_spotting.py:finalize() —
    utf-8-sig (handle BOM), newline="" (preserve embedded newlines in cells),
    QUOTE_NONE (don't interpret quotes).
    """
    by_surface: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(
            f, delimiter="\t", quoting=csv.QUOTE_NONE
        )
        for row in reader:
            key = (row["gothic_sentence_roman"], row["gothic_surface"])
            english = row["english_word"]
            status = row["status"].strip().lower()
            by_surface[key].append((english, status))
    return dict(by_surface)


def lookup_tsv_status(
    tsv_index: dict[tuple[str, str], list[tuple[str, str]]],
    gothic_sentence_roman: str,
    gothic_surface: str,
    english_word: str,
) -> str | None:
    """Look up a TSV status for an LLM original.

    Match strategy, in order:
      1. Exact (sentence, gothic_surface, english_word).
      2. Same (sentence, gothic_surface), and the TSV english_word starts
         with the LLM english_word (catches "spoke (better: said)" cases).
      3. Same (sentence, gothic_surface) with any english (parenthetical
         note or any other edit).
      4. None.

    Returns the matched status and pops the matched entry from the list so
    repeated lookups (for LLM duplicates) consume distinct TSV rows.
    """
    key = (gothic_sentence_roman, gothic_surface)
    candidates = tsv_index.get(key)
    if not candidates:
        return None

    chosen_idx = None
    for i, (e, _) in enumerate(candidates):
        if e == english_word:
            chosen_idx = i
            break
    if chosen_idx is None:
        for i, (e, _) in enumerate(candidates):
            if e.startswith(english_word):
                chosen_idx = i
                break
    if chosen_idx is None:
        chosen_idx = 0

    _, status = candidates.pop(chosen_idx)
    return status


def index_by_gothic_roman(records: list[dict]) -> dict[str, dict]:
    """Index a list of sentence records by their gothic_sentence_roman."""
    return {r["gothic_sentence_roman"]: r for r in records}


def index_diversify_responses(
    manifest_path: Path,
    diversify_annotations_path: Path,
) -> dict[tuple[str, int], list[dict]]:
    """For each (source_file, record_idx_in_source) touched by diversification,
    return the model's new_alignments lists keyed by focal pair, plus the
    overall list of new alignments to add.

    Returns:
        A dict (source, record_idx_in_source) ->
            list of dicts, each
            { 'focal_pair': (g, e),
              'new_alignments': [..],   (alignment objs from the model)
              'response_position': int  (0-based position in the response, for
                                         relative ordering within a batch) }
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    annotations = load_jsonl(diversify_annotations_path)
    by_custom_id: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        if "_custom_id" in ann:
            by_custom_id[ann["_custom_id"]].append(ann)

    indexed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for entry in manifest["entries"]:
        custom_id = entry["custom_id"]
        focal = (
            entry["focal_pair"]["gothic"].strip().lower(),
            entry["focal_pair"]["english"].strip().lower(),
        )
        source = entry["source"]
        record_indices = entry["record_indices"]
        anns = by_custom_id.get(custom_id, [])
        if not anns:
            continue
        n = min(len(anns), len(record_indices))
        for pos in range(n):
            record_idx = record_indices[pos]
            new_aligns = anns[pos].get("new_alignments", []) or []
            indexed[(source, record_idx)].append(
                {
                    "focal_pair": focal,
                    "new_alignments": new_aligns,
                }
            )
    return dict(indexed)


def normalize_pair(gothic: str, english: str) -> tuple[str, str]:
    return gothic.strip().lower(), english.strip().lower()


def match_originals_to_verified(
    originals: list[dict],
    verified: list[dict],
) -> list[tuple[dict, dict | None, str]]:
    """Greedy match each LLM original to a verified-b alignment.

    Returns a list of (original, matched_verified_or_None, match_kind) where
    match_kind in {'exact', 'edited', 'no_match'}.

    'edited' covers any partial match (same gothic, different english, or
    vice versa).
    """
    available = list(range(len(verified)))
    pairings: list[tuple[dict, dict | None, str]] = []

    # pass 1: exact
    for orig in originals:
        match_idx = None
        for vi in available:
            v = verified[vi]
            if (
                v["gothic_word_roman"] == orig["gothic_word_roman"]
                and v["target_word"] == orig["target_word"]
            ):
                match_idx = vi
                break
        if match_idx is not None:
            available.remove(match_idx)
            pairings.append((orig, verified[match_idx], "exact"))
        else:
            pairings.append((orig, None, "pending"))

    # pass 2: gothic-only or english-only match among remaining
    for i, (orig, match, kind) in enumerate(pairings):
        if kind != "pending":
            continue
        match_idx = None
        for vi in available:
            v = verified[vi]
            if (
                v["gothic_word_roman"] == orig["gothic_word_roman"]
                or v["target_word"] == orig["target_word"]
            ):
                match_idx = vi
                break
        if match_idx is not None:
            available.remove(match_idx)
            pairings[i] = (orig, verified[match_idx], "edited")
        else:
            pairings[i] = (orig, None, "no_match")

    return pairings


def alignment_record(
    align_id: str,
    source: str,
    status: str,
    surface: dict,
    original: dict | None = None,
    extra: dict | None = None,
) -> dict:
    record = {
        "alignment_id": align_id,
        "source": source,
        "status": status,
        "target_word": surface["target_word"],
        "gothic_word_roman": surface["gothic_word_roman"],
        "gothic_word_gothic": surface["gothic_word_gothic"],
    }
    if original is not None:
        record["original"] = {
            "target_word": original["target_word"],
            "gothic_word_roman": original["gothic_word_roman"],
            "gothic_word_gothic": original["gothic_word_gothic"],
        }
    if extra:
        record.update(extra)
    return record


def process_split(
    split: str,
    annotations_path: Path,
    verified_tsv_path: Path,
    verified_jsonl_path: Path,
    diversified_jsonl_path: Path | None,
    diversify_index: dict[tuple[str, int], list[dict]] | None,
    diversified_source_key: str | None,
) -> list[dict]:
    """Produce the canonical alignments list for one split."""
    annotations = load_jsonl(annotations_path)
    verified_b = load_jsonl(verified_jsonl_path)
    verified_by_gothic = index_by_gothic_roman(verified_b)
    tsv_status = parse_verification_tsv(verified_tsv_path)

    if diversified_jsonl_path is not None:
        diversified = load_jsonl(diversified_jsonl_path)
        diversified_by_gothic = index_by_gothic_roman(diversified)
    else:
        diversified_by_gothic = {}

    # to map diversified-file record positions back to (source_key, record_idx_in_diversified_file),
    # we need the *original* verified_b record index (since the manifest's
    # record_indices are over the verified_b file, not the diversified one).
    # the diversify merge preserved record order, so the i-th record in
    # diversified.jsonl corresponds to the i-th record in verified_b.jsonl.
    verified_b_index_by_gothic = {
        r["gothic_sentence_roman"]: i for i, r in enumerate(verified_b)
    }

    out_records: list[dict] = []
    unmatched_warnings = 0

    for sent_idx, ann in enumerate(annotations):
        sentence_id = f"{split}_{sent_idx:04d}"
        gothic_roman = ann["gothic_sentence_roman"]
        originals = ann.get("alignments", [])
        verified_record = verified_by_gothic.get(gothic_roman)
        verified_alignments = (
            verified_record.get("alignments", []) if verified_record else []
        )
        diversified_record = diversified_by_gothic.get(gothic_roman)
        diversified_alignments = (
            diversified_record.get("alignments", []) if diversified_record else []
        )

        # collect originals with their TSV statuses + a possible verified match
        pairings = match_originals_to_verified(originals, verified_alignments)

        # warn if verified_b has alignments not accounted for by any LLM
        # original — would indicate a wholesale edit during verification (both
        # gothic_surface and english_word changed, breaking all match paths)
        # or a manually-added alignment that needs its own provenance entry.
        matched_ids = {id(m) for _, m, _ in pairings if m is not None}
        for vb in verified_alignments:
            if id(vb) not in matched_ids:
                print(
                    f"Warning: verified-b alignment "
                    f"({vb['gothic_word_roman']} ↔ {vb['target_word']}) "
                    f"in {sentence_id} has no matching LLM original; "
                    f"it will be absent from the canonical output. "
                    f"(Wholesale edit or manual addition during verification.)",
                    file=sys.stderr,
                )

        # determine which (g, e) pairs from verified are present in diversified
        # to detect replaced_in_diversification
        diversified_keys: set[tuple[str, str]] = set()
        for a in diversified_alignments:
            diversified_keys.add(normalize_pair(a["gothic_word_roman"], a["target_word"]))

        result_alignments: list[dict] = []
        used_verified_keys: set[tuple[str, str]] = set()

        for o_idx, (orig, matched_verified, kind) in enumerate(pairings):
            align_id = f"{sentence_id}_o{o_idx}"
            tsv_st = lookup_tsv_status(
                tsv_status,
                gothic_roman,
                orig["gothic_word_roman"],
                orig["target_word"],
            )
            if tsv_st is None:
                # TSV-missing — likely lost during a previous TSV/JSONL
                # round-trip. If the alignment survives in verified_b, infer
                # it was kept; otherwise we genuinely don't know.
                if matched_verified is not None:
                    tsv_st = "okay"
                else:
                    unmatched_warnings += 1
                    if unmatched_warnings <= 5:
                        print(
                            f"Warning: no TSV row for {align_id} "
                            f"({orig['gothic_word_roman']} <-> {orig['target_word']}) "
                            f"and no verified-b match; defaulting to "
                            f"status=unverified",
                            file=sys.stderr,
                        )
                    tsv_st = "unchecked"

            if tsv_st == "delete":
                result_alignments.append(
                    alignment_record(align_id, "llm_original", "rejected", orig)
                )
            elif tsv_st == "unchecked":
                result_alignments.append(
                    alignment_record(align_id, "llm_original", "unverified", orig)
                )
            elif tsv_st == "okay":
                # need to use the verified surface form (possibly edited)
                if matched_verified is None:
                    # TSV said okay but no verified-b match found; fall back to
                    # the original surface forms and flag it.
                    print(
                        f"Warning: {align_id} okay in TSV but no verified-b match; "
                        f"recording with LLM-original surface forms.",
                        file=sys.stderr,
                    )
                    surface = orig
                    edited = False
                else:
                    surface = matched_verified
                    used_verified_keys.add(
                        normalize_pair(
                            matched_verified["gothic_word_roman"],
                            matched_verified["target_word"],
                        )
                    )
                    edited = kind == "edited"

                # was this alignment dropped during diversification?
                if diversified_record is not None:
                    pair_norm = normalize_pair(
                        surface["gothic_word_roman"], surface["target_word"]
                    )
                    if pair_norm not in diversified_keys:
                        status = "replaced_in_diversification"
                    else:
                        status = "kept_edited" if edited else "verified_correct"
                else:
                    status = "kept_edited" if edited else "verified_correct"

                result_alignments.append(
                    alignment_record(
                        align_id,
                        "llm_original",
                        status,
                        surface,
                        original=orig if edited else None,
                    )
                )
            else:
                print(
                    f"Warning: unknown TSV status {tsv_st!r} for {align_id}",
                    file=sys.stderr,
                )
                result_alignments.append(
                    alignment_record(align_id, "llm_original", "unverified", orig)
                )

        # diversification-added alignments: anything in diversified_record but
        # not in verified_b (after edit-matching).
        if (
            diversified_record is not None
            and diversify_index is not None
            and diversified_source_key is not None
        ):
            record_idx_in_verified = verified_b_index_by_gothic.get(gothic_roman)
            entries = diversify_index.get(
                (diversified_source_key, record_idx_in_verified), []
            )
            d_idx = 0
            for entry in entries:
                focal = entry["focal_pair"]
                for new_align in entry["new_alignments"]:
                    pair_norm = normalize_pair(
                        new_align["gothic_word_roman"], new_align["target_word"]
                    )
                    # skip if this pair was already produced as an llm_original
                    # surface form (rare edge case)
                    already = any(
                        normalize_pair(a["gothic_word_roman"], a["target_word"])
                        == pair_norm
                        and a["source"] == "llm_original"
                        for a in result_alignments
                    )
                    if already:
                        continue
                    align_id = f"{sentence_id}_d{d_idx}"
                    d_idx += 1
                    extra = {
                        "replaced_focal_pair": {
                            "gothic_word_roman": focal[0],
                            "target_word": focal[1],
                        }
                    }
                    result_alignments.append(
                        alignment_record(
                            align_id,
                            "llm_diversify",
                            "unverified",
                            new_align,
                            extra=extra,
                        )
                    )

        out_records.append(
            {
                "sentence_id": sentence_id,
                "english_sentence": ann["english_sentence"],
                "gothic_sentence_roman": gothic_roman,
                "gothic_sentence_gothic": ann["gothic_sentence_gothic"],
                "alignments": result_alignments,
            }
        )

    if unmatched_warnings > 5:
        print(
            f"  ... {unmatched_warnings - 5} more TSV-lookup misses suppressed.",
            file=sys.stderr,
        )
    return out_records


def summarize_statuses(records: list[dict]) -> None:
    from collections import Counter

    counts: Counter = Counter()
    by_source: Counter = Counter()
    for r in records:
        for a in r["alignments"]:
            counts[a["status"]] += 1
            by_source[(a["source"], a["status"])] += 1
    total = sum(counts.values())
    print(f"  Total alignments: {total}", file=sys.stderr)
    for status, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {status}: {n}", file=sys.stderr)
    print("  By source:", file=sys.stderr)
    for (src, status), n in sorted(by_source.items()):
        print(f"    {src} / {status}: {n}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--verified-tsv", required=True, type=Path)
    parser.add_argument("--verified-jsonl", required=True, type=Path)
    parser.add_argument("--diversified-jsonl", type=Path, default=None)
    parser.add_argument("--diversify-annotations", type=Path, default=None)
    parser.add_argument("--diversify-manifest", type=Path, default=None)
    parser.add_argument(
        "--diversified-source-key",
        default=None,
        help=(
            "The string used as 'source' in the diversify manifest for this "
            "split (e.g. 'data/gothic_word_spotting/train_verified_b.jsonl'). "
            "Defaults to the value of --verified-jsonl."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    diversify_index = None
    if args.diversify_annotations and args.diversify_manifest:
        diversify_index = index_diversify_responses(
            args.diversify_manifest, args.diversify_annotations
        )
    elif bool(args.diversify_annotations) != bool(args.diversify_manifest):
        print(
            "Error: --diversify-annotations and --diversify-manifest must be "
            "provided together.",
            file=sys.stderr,
        )
        sys.exit(1)

    diversified_source_key = args.diversified_source_key or str(args.verified_jsonl)

    print(f"Processing {args.split}...", file=sys.stderr)
    out_records = process_split(
        args.split,
        args.annotations,
        args.verified_tsv,
        args.verified_jsonl,
        args.diversified_jsonl,
        diversify_index,
        diversified_source_key,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in out_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_records)} sentences to {args.output}", file=sys.stderr)
    print("\nStatus summary:", file=sys.stderr)
    summarize_statuses(out_records)


if __name__ == "__main__":
    main()
