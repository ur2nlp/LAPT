#!/usr/bin/env python3
"""
Merge diversification-pass annotations back into the verified word-spotting
JSONL files.

Reads:
    --annotations  Flat JSONL of model outputs from run_annotation.py
                   (each line tagged with _custom_id).
    --manifest     The manifest JSON written by prepare_batches.py in
                   diversify mode.

For each source file referenced in the manifest, writes a new JSONL with the
focal-pair alignments replaced by the model's new alignments, subject to a
retention proportion (default 0.3): a configurable fraction of each focal
pair's occurrences are kept in the output so that the pair's training-signal
presence is reduced but not eliminated.

Output paths default to <source>_diversified.jsonl. Override with
--output-suffix or --output-map.

Usage:
    python -m gothic.word_spotting.merge_diversification \\
        --annotations data/gothic_word_spotting/diversify_annotations.jsonl \\
        --manifest data/gothic_word_spotting/diversify_batch_requests.manifest.json
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def normalize_pair(gothic: str, english: str) -> tuple[str, str]:
    return gothic.strip().lower(), english.strip().lower()


def normalize_alignment(alignment: dict) -> tuple[str, str]:
    return normalize_pair(alignment["gothic_word_roman"], alignment["target_word"])


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_annotations_by_custom_id(
    annotations: list[dict],
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        custom_id = ann.get("_custom_id")
        if not custom_id:
            print(
                "Warning: annotation missing _custom_id; skipping.",
                file=sys.stderr,
            )
            continue
        grouped[custom_id].append(ann)
    return dict(grouped)


def build_record_ops(
    manifest: dict,
    annotations_by_custom_id: dict[str, list[dict]],
    source_records: dict[str, list[dict]],
) -> dict[tuple[str, int], dict]:
    """For each (source, record_idx) touched by diversification, accumulate:
        removed:     set of focal (gothic, english) pairs to strip
        added:       list of new alignment dicts from the model
        abstentions: set of focal pairs where the model returned [] for this
                     record (signals "no good alternative")
    """
    record_ops: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"removed": set(), "added": [], "abstentions": set()}
    )

    missing_responses = 0
    mismatch_warnings = 0

    for entry in manifest["entries"]:
        custom_id = entry["custom_id"]
        focal = normalize_pair(
            entry["focal_pair"]["gothic"], entry["focal_pair"]["english"]
        )
        source = entry["source"]
        record_indices = entry["record_indices"]
        anns = annotations_by_custom_id.get(custom_id, [])

        if not anns:
            # batch failed or no parseable output; leave these records untouched
            missing_responses += 1
            print(
                f"Warning: no annotations for {custom_id} "
                f"(focal {focal[0]} ↔ {focal[1]}, {len(record_indices)} records); "
                f"leaving records untouched.",
                file=sys.stderr,
            )
            continue

        if len(anns) != len(record_indices):
            print(
                f"Warning: {custom_id} got {len(anns)} annotations but expected "
                f"{len(record_indices)}; processing min().",
                file=sys.stderr,
            )

        n = min(len(anns), len(record_indices))
        for pos in range(n):
            record_idx = record_indices[pos]
            ann = anns[pos]
            # sanity-check english_sentence matches the expected source record
            expected = source_records[source][record_idx]
            if ann.get("english_sentence") != expected["english_sentence"]:
                mismatch_warnings += 1
                if mismatch_warnings <= 5:
                    print(
                        f"Warning: english_sentence mismatch in {custom_id} "
                        f"position {pos} (source record {record_idx}): "
                        f"model returned {ann.get('english_sentence', '')[:60]!r}, "
                        f"expected {expected['english_sentence'][:60]!r}",
                        file=sys.stderr,
                    )

            key = (source, record_idx)
            record_ops[key]["removed"].add(focal)
            new_aligns = ann.get("new_alignments", []) or []
            if not new_aligns:
                record_ops[key]["abstentions"].add(focal)
            else:
                record_ops[key]["added"].extend(new_aligns)

    if missing_responses:
        print(
            f"Warning: {missing_responses} batches had no annotations.",
            file=sys.stderr,
        )
    if mismatch_warnings > 5:
        print(
            f"  ... {mismatch_warnings - 5} more english_sentence mismatches suppressed.",
            file=sys.stderr,
        )

    return dict(record_ops)


def alignment_count_after(record: dict, ops: dict) -> int:
    """Estimate how many alignments record will have after merge, ignoring
    retention (treats all `removed` pairs as removed and all `added` as kept).
    Used to prioritize retention toward sparser records.
    """
    surviving = [
        a
        for a in record["alignments"]
        if normalize_alignment(a) not in ops["removed"]
    ]
    # crude (non-deduped) count; good enough for ranking
    return len(surviving) + len(ops["added"])


def select_retained(
    record_ops: dict[tuple[str, int], dict],
    source_records: dict[str, list[dict]],
    retention_proportion: float,
    rng: random.Random,
) -> dict[tuple[str, str], set[tuple[str, int]]]:
    """For each focal pair, choose which (source, record_idx) keys retain it.

    Priority for retention:
        1. Records where the model abstained (returned [] new_alignments).
        2. Records with the fewest surviving alignments after merge.
    Ties broken by stable shuffle for reproducibility.
    """
    # group records by focal pair
    pair_to_keys: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for key, ops in record_ops.items():
        for pair in ops["removed"]:
            pair_to_keys[pair].append(key)

    retained: dict[tuple[str, str], set[tuple[str, int]]] = {}

    for pair, keys in pair_to_keys.items():
        n_total = len(keys)
        n_retain = round(n_total * retention_proportion)

        if n_retain <= 0:
            retained[pair] = set()
            continue

        abstainers = [k for k in keys if pair in record_ops[k]["abstentions"]]
        non_abstainers = [k for k in keys if pair not in record_ops[k]["abstentions"]]

        # stable shuffle within each priority tier
        rng.shuffle(abstainers)
        # rank non-abstainers by surviving alignment count (ascending);
        # add a random tiebreaker to spread ties evenly across the corpus
        non_abstainers.sort(
            key=lambda k: (
                alignment_count_after(
                    source_records[k[0]][k[1]], record_ops[k]
                ),
                rng.random(),
            )
        )

        chosen = abstainers[:n_retain]
        if len(chosen) < n_retain:
            chosen.extend(non_abstainers[: n_retain - len(chosen)])
        retained[pair] = set(chosen)

    return retained


def merge_records(
    source: str,
    records: list[dict],
    record_ops: dict[tuple[str, int], dict],
    retained: dict[tuple[str, str], set[tuple[str, int]]],
) -> list[dict]:
    """Apply record_ops + retention to a single source file's records."""
    output: list[dict] = []
    for i, record in enumerate(records):
        key = (source, i)
        if key not in record_ops:
            output.append(record)
            continue

        ops = record_ops[key]
        # which focal pairs are actually being removed from this record
        # (= ops['removed'] minus those that this record is keeping by retention)
        effective_removed = {
            pair for pair in ops["removed"] if key not in retained.get(pair, set())
        }

        new_alignments: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()
        for a in record["alignments"]:
            pair = normalize_alignment(a)
            if pair in effective_removed:
                continue
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            new_alignments.append(a)
        for a in ops["added"]:
            pair = normalize_alignment(a)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            new_alignments.append(a)

        new_record = dict(record)
        new_record["alignments"] = new_alignments
        output.append(new_record)
    return output


def summarize(
    source: str,
    before: list[dict],
    after: list[dict],
) -> None:
    def pair_counts(records: list[dict]):
        from collections import Counter

        c: Counter = Counter()
        for r in records:
            for a in r.get("alignments", []):
                c[normalize_alignment(a)] += 1
        return c

    before_counts = pair_counts(before)
    after_counts = pair_counts(after)
    total_before = sum(before_counts.values())
    total_after = sum(after_counts.values())
    print(f"\n== {source} ==", file=sys.stderr)
    print(
        f"  total alignments: {total_before} -> {total_after}", file=sys.stderr
    )
    print(
        f"  distinct pairs:   {len(before_counts)} -> {len(after_counts)}",
        file=sys.stderr,
    )
    # show top changes
    deltas = []
    all_pairs = set(before_counts) | set(after_counts)
    for p in all_pairs:
        deltas.append((p, after_counts[p] - before_counts[p]))
    deltas.sort(key=lambda x: x[1])
    print("  largest decreases:", file=sys.stderr)
    for pair, d in deltas[:8]:
        if d >= 0:
            break
        print(
            f"    {pair[0]} ↔ {pair[1]}: {before_counts[pair]} -> {after_counts[pair]}  "
            f"({d:+d})",
            file=sys.stderr,
        )
    print("  largest increases:", file=sys.stderr)
    for pair, d in reversed(deltas[-8:]):
        if d <= 0:
            break
        print(
            f"    {pair[0]} ↔ {pair[1]}: {before_counts[pair]} -> {after_counts[pair]}  "
            f"({d:+d})",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Flat JSONL of diversification annotations from run_annotation.py.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Manifest JSON written by prepare_batches.py in diversify mode.",
    )
    parser.add_argument(
        "--retention-proportion",
        type=float,
        default=0.3,
        help=(
            "Fraction of each focal pair's occurrences to retain in the "
            "output (default: 0.3). Retention is concentrated on records "
            "where the model abstained, then on records with the fewest "
            "surviving alignments."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default="_diversified",
        help=(
            "Suffix appended to each source file's stem when writing the "
            "merged output (default: _diversified). For example, "
            "train_verified_b.jsonl -> train_verified_b_diversified.jsonl."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for retention selection (default: 1).",
    )
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Loading annotations from {args.annotations}...", file=sys.stderr)
    annotations = load_jsonl(args.annotations)
    annotations_by_custom_id = group_annotations_by_custom_id(annotations)
    print(
        f"  {len(annotations)} annotations across "
        f"{len(annotations_by_custom_id)} custom_ids",
        file=sys.stderr,
    )

    # load all source files referenced by the manifest
    sources = manifest["alignments_sources"]
    source_records: dict[str, list[dict]] = {}
    for source in sources:
        path = Path(source)
        if not path.exists():
            print(f"Error: source file not found: {path}", file=sys.stderr)
            sys.exit(1)
        source_records[source] = load_jsonl(path)
        print(
            f"Loaded {len(source_records[source])} records from {path}",
            file=sys.stderr,
        )

    record_ops = build_record_ops(
        manifest, annotations_by_custom_id, source_records
    )
    print(
        f"\nAccumulated ops for {len(record_ops)} records "
        f"across {len(sources)} source file(s).",
        file=sys.stderr,
    )

    rng = random.Random(args.seed)
    retained = select_retained(
        record_ops, source_records, args.retention_proportion, rng
    )

    # diagnostic: how many records retain each focal pair
    print(
        f"\nRetention summary (proportion={args.retention_proportion}):",
        file=sys.stderr,
    )
    pair_to_keys = defaultdict(list)
    for key, ops in record_ops.items():
        for pair in ops["removed"]:
            pair_to_keys[pair].append(key)
    for pair in sorted(pair_to_keys, key=lambda p: -len(pair_to_keys[p]))[:10]:
        keys = pair_to_keys[pair]
        n_retained = len(retained.get(pair, set()))
        print(
            f"  {pair[0]} ↔ {pair[1]}: {len(keys)} touched, "
            f"{n_retained} retained, {len(keys) - n_retained} replaced",
            file=sys.stderr,
        )

    # merge and write
    for source in sources:
        path = Path(source)
        output_path = path.with_name(path.stem + args.output_suffix + path.suffix)
        merged = merge_records(
            source, source_records[source], record_ops, retained
        )
        with open(output_path, "w", encoding="utf-8") as f:
            for record in merged:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(merged)} records to {output_path}", file=sys.stderr)
        summarize(source, source_records[source], merged)


if __name__ == "__main__":
    main()
