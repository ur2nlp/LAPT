#!/usr/bin/env python3
"""
Review the still-``unverified`` alignments in a canonical word-spotting file and
patch the reviewed statuses back in place, keyed by ``alignment_id``.

This is the v0.4 -> v0.5 review round-trip. Unlike ``verify_word_spotting.py``
(which scores a flat per-sentence JSONL and *rebuilds* a new file on finalize),
this module operates on ``{train,test}_alignments.jsonl`` — the canonical file
with stable ``alignment_id`` / ``status`` / provenance — and touches ONLY the
alignments named in the reviewed TSV. Every other alignment is left
byte-identical.

By default it reviews the ``unverified`` subset, but ``--statuses`` re-targets it
to any status or set of statuses (e.g. ``verified_correct,kept_edited`` to
re-audit already-resolved rows). The apply patch is keyed purely on
``alignment_id`` and is status-agnostic; ``--statuses`` only governs which rows
the export emits and which the apply expects to see covered.

Export mode (default):
    Filter the canonical file to the reviewed statuses, score each against the
    Koebler dictionary (same tiers as ``verify_word_spotting.py``), and write a
    review TSV. The first column is ``alignment_id`` — the join key used to patch
    the file back; the ``status`` column carries each alignment's current status.
    Sentence fields are carried for review context.

    The reviewer, in a spreadsheet:
      - sets ``status`` to ``correct`` (accept as-is) or ``reject`` (drop);
      - optionally edits ``english_word`` / ``gothic_surface`` /
        ``gothic_surface_gothic`` to fix an alignment — an accepted row whose
        surfaces differ from the canonical values becomes ``kept_edited``.

Apply mode (--apply):
    Read the reviewed TSV, look up each row by ``alignment_id`` in the canonical
    file, and update only those alignments:
      - ``reject``  -> status ``rejected``
      - ``correct`` (unchanged surfaces) -> status ``verified_correct``
      - ``correct`` (edited surfaces)    -> status ``kept_edited``; the pre-edit
        ``target_word`` / ``gothic_word_roman`` / ``gothic_word_gothic`` are stored
        under ``original``, then overwritten with the reviewed values.
    Rows still left blank/``unverified`` are reported and skipped (status unchanged).

Usage:
    # Export the unverified subset to a review TSV
    python -m gothic.word_spotting.review_unverified \
        data/gothic_word_spotting/train_alignments.jsonl \
        --output data/gothic_word_spotting/verification/train_v05_review.tsv \
        --normalize --sort worst-first

    # [review in spreadsheet]

    # Patch the reviewed statuses back into the canonical file (in place)
    python -m gothic.word_spotting.review_unverified \
        data/gothic_word_spotting/train_alignments.jsonl \
        --apply data/gothic_word_spotting/verification/train_v05_review.tsv
"""

import argparse
import csv
import json
import os
import sys

from gothic.orthography import transliterate_gothic_to_latin
from gothic.word_spotting.verify_word_spotting import (
    load_dictionary,
    score_alignment,
)

# Default status reviewed by this round-trip (the v0.4 -> v0.5 unverified subset).
# Override with --statuses to re-review any other status or set of statuses.
UNVERIFIED_STATUS = "unverified"
DEFAULT_REVIEW_STATUSES = frozenset({UNVERIFIED_STATUS})

# Reviewer status-cell vocabularies (lowercased, stripped).
REJECT_STATUSES = {"reject", "rejected", "delete", "deleted", "drop", "remove", "x", "✗", "no"}
ACCEPT_STATUSES = {
    "correct", "ok", "okay", "yes", "keep", "kept", "good",
    "verified", "verified_correct", "kept_edited",
}
PENDING_STATUSES = {"", "unverified", "unchecked", "?", "todo", "skip"}


def load_alignment_records(jsonl_path: str) -> list[dict]:
    """Load a canonical alignments JSONL into a list of sentence records."""
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export_review_tsv(
    jsonl_path: str,
    output_path: str | None,
    dict_path: str,
    threshold: int,
    use_normalize: bool,
    top_n: int,
    sort_order: str,
    review_statuses: frozenset[str] = DEFAULT_REVIEW_STATUSES,
) -> None:
    """Write a review TSV containing the alignments whose status is reviewable.

    Args:
        review_statuses: Statuses to export (default: ``{"unverified"}``). Pass a
            larger set via ``--statuses`` to re-review already-resolved
            alignments (e.g. ``verified_correct``).
    """
    all_forms, english_to_forms = load_dictionary(dict_path)
    form_to_glosses = {cf: gs for cf, gs in all_forms}

    records = load_alignment_records(jsonl_path)

    rows: list[dict] = []
    for record in records:
        english_sentence = record["english_sentence"]
        gothic_sentence_roman = record["gothic_sentence_roman"]
        gothic_sentence_gothic = record["gothic_sentence_gothic"]

        for alignment in record["alignments"]:
            if alignment.get("status") not in review_statuses:
                continue

            target_word = alignment["target_word"]
            gothic_surface = alignment["gothic_word_roman"]
            gothic_surface_gothic = alignment["gothic_word_gothic"]

            scored = score_alignment(
                target_word, gothic_surface, gothic_surface_gothic,
                all_forms, english_to_forms, form_to_glosses,
                threshold, use_normalize, top_n,
            )
            if scored is None:
                print(
                    f"Warning: empty surface form after cleaning "
                    f"'{gothic_surface}' ({alignment['alignment_id']}), skipping",
                    file=sys.stderr,
                )
                continue

            row = {
                "status": alignment.get("status", ""),
                "alignment_id": alignment["alignment_id"],
                "source": alignment.get("source", ""),
                "tier": scored["tier"],
                "english_word": target_word,
                "gothic_surface": gothic_surface,
                "gothic_surface_gothic": gothic_surface_gothic,
                "script_ok": scored["script_ok"],
            }
            for i in range(1, top_n + 1):
                row[f"dist_{i}"] = scored[f"dist_{i}"]
                row[f"match_{i}"] = scored[f"match_{i}"]
            row["english_sentence"] = english_sentence
            row["gothic_sentence_roman"] = gothic_sentence_roman
            row["gothic_sentence_gothic"] = gothic_sentence_gothic

            rows.append(row)

    if sort_order == "worst-first":
        rows.sort(key=lambda r: (-r["tier"], -(r.get("dist_1") or 0)))
    else:
        rows.sort(key=lambda r: (r["tier"], r.get("dist_1") or 0))

    fieldnames = [
        "status", "alignment_id", "source", "tier",
        "english_word", "gothic_surface", "gothic_surface_gothic", "script_ok",
    ]
    for i in range(1, top_n + 1):
        fieldnames.extend([f"dist_{i}", f"match_{i}"])
    fieldnames.extend([
        "english_sentence", "gothic_sentence_roman", "gothic_sentence_gothic",
    ])

    if output_path:
        out_file = open(output_path, "w", encoding="utf-8-sig", newline="")
    else:
        out_file = sys.stdout
    try:
        # The Google Sheets round-trip is ASYMMETRIC, so write and read use
        # different quoting (see apply_review for the read side):
        #   - WRITE with RFC-4180 QUOTE_MINIMAL. On *import*, Sheets honors quoting;
        #     a bare '"' opening a Bible verse would otherwise be read as a
        #     field-quote opener and swallow subsequent rows into one cell.
        writer = csv.DictWriter(
            out_file, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output_path:
            out_file.close()

    tier_counts = {1: 0, 2: 0, 3: 0}
    for row in rows:
        tier_counts[row["tier"]] += 1
    statuses_label = ", ".join(sorted(review_statuses))
    print(
        f"Alignments exported (status in {{{statuses_label}}}): {len(rows)}",
        file=sys.stderr,
    )
    print(f"  Tier 1 (high):       {tier_counts[1]}", file=sys.stderr)
    print(f"  Tier 2 (form):       {tier_counts[2]}", file=sys.stderr)
    print(f"  Tier 3 (suspicious): {tier_counts[3]}", file=sys.stderr)


def apply_review(
    jsonl_path: str,
    review_path: str,
    output_path: str | None,
    review_statuses: frozenset[str] = DEFAULT_REVIEW_STATUSES,
    coverage_check: bool = True,
) -> None:
    """Patch reviewed statuses back into the canonical file, keyed by alignment_id.

    The patch itself is status-agnostic: every alignment whose ``alignment_id``
    appears in the reviewed TSV is updated, regardless of its prior status. The
    ``review_statuses`` set is used only for coverage accounting — to detect
    alignments that were *meant* to be reviewed but have no row in the TSV (the
    failure mode of a spreadsheet round-trip silently dropping rows).

    Args:
        coverage_check: When True (the default), warn about alignments whose
            status is in ``review_statuses`` but which have no row in the TSV.
            Disable this for a **content-defined subset** review (e.g. the
            English-mismatch TSV, which intentionally covers only a few of many
            same-status rows), where the snapshot would flag the unreviewed
            remainder. The orphan check (TSV ids absent from the file) always
            runs.
    """
    with open(review_path, encoding="utf-8-sig", newline="") as f:
        # READ with QUOTE_NONE: Google Sheets' TSV *export* is unquoted (it dumps
        # raw cell values, passing '"' through literally), so a QUOTE_MINIMAL reader
        # would treat a verse's leading '"' as a field-quote opener and merge rows.
        # Quirk verified against real Sheets output; see word_spotting.md § Step 4.7.
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        review_rows = list(reader)

    review_by_id: dict[str, dict] = {}
    for row in review_rows:
        alignment_id = (row.get("alignment_id") or "").strip()
        if not alignment_id:
            print("Warning: review row with no alignment_id, skipping", file=sys.stderr)
            continue
        if alignment_id in review_by_id:
            print(
                f"Warning: duplicate alignment_id {alignment_id} in review TSV; "
                f"using the last occurrence",
                file=sys.stderr,
            )
        review_by_id[alignment_id] = row

    records = load_alignment_records(jsonl_path)

    # Snapshot which alignments were awaiting review, to detect rows that the
    # review TSV failed to cover (e.g. lost to a spreadsheet round-trip). Skipped
    # for content-defined subset reviews, where it would flag the remainder.
    if coverage_check:
        review_before = {
            alignment["alignment_id"]
            for record in records
            for alignment in record["alignments"]
            if alignment.get("status") in review_statuses
        }
    else:
        review_before = set()

    counts = {"verified_correct": 0, "kept_edited": 0, "rejected": 0}
    pending: list[str] = []
    unknown_status: list[str] = []
    matched_ids: set[str] = set()

    for record in records:
        for alignment in record["alignments"]:
            alignment_id = alignment.get("alignment_id")
            if alignment_id not in review_by_id:
                continue
            matched_ids.add(alignment_id)
            row = review_by_id[alignment_id]
            status_cell = (row.get("status") or "").strip().lower()

            if status_cell in PENDING_STATUSES:
                pending.append(alignment_id)
                continue
            if status_cell in REJECT_STATUSES:
                alignment["status"] = "rejected"
                counts["rejected"] += 1
                continue
            if status_cell not in ACCEPT_STATUSES:
                unknown_status.append(f"{alignment_id} ('{status_cell}')")
                pending.append(alignment_id)
                continue

            # Accepted: detect reviewer edits to the surface forms.
            new_target = (row.get("english_word") or "").strip()
            new_roman = (row.get("gothic_surface") or "").strip()
            new_gothic = (row.get("gothic_surface_gothic") or "").strip()

            # Compare against the PRISTINE (pre-edit) surfaces, not the current
            # ones, so the apply is idempotent: re-running on an already-patched
            # file (where the surfaces already hold the edit) still recognizes the
            # edit instead of silently demoting kept_edited -> verified_correct.
            pristine = alignment.get("original", {
                "target_word": alignment["target_word"],
                "gothic_word_roman": alignment["gothic_word_roman"],
                "gothic_word_gothic": alignment["gothic_word_gothic"],
            })
            edited = (
                new_target != pristine["target_word"]
                or new_roman != pristine["gothic_word_roman"]
                or new_gothic != pristine["gothic_word_gothic"]
            )
            if edited:
                alignment["original"] = pristine
                alignment["target_word"] = new_target
                alignment["gothic_word_roman"] = new_roman
                alignment["gothic_word_gothic"] = new_gothic
                alignment["status"] = "kept_edited"
                counts["kept_edited"] += 1

                transliterated = transliterate_gothic_to_latin(new_gothic)
                if transliterated != new_roman:
                    print(
                        f"Warning: edited {alignment_id} script mismatch — "
                        f"Gothic transliterates to '{transliterated}', "
                        f"Roman is '{new_roman}'",
                        file=sys.stderr,
                    )
            else:
                # No edit (or a previously-applied edit the reviewer reverted):
                # restore pristine surfaces and drop the stale original marker.
                alignment["target_word"] = pristine["target_word"]
                alignment["gothic_word_roman"] = pristine["gothic_word_roman"]
                alignment["gothic_word_gothic"] = pristine["gothic_word_gothic"]
                alignment.pop("original", None)
                alignment["status"] = "verified_correct"
                counts["verified_correct"] += 1

    # Reconcile the two sides.
    review_ids = set(review_by_id)
    orphan_review_ids = review_ids - matched_ids
    # Alignments that were awaiting review but have no row in the TSV at all —
    # the failure mode of a spreadsheet round-trip silently dropping/merging rows.
    uncovered_ids = review_before - review_ids

    if output_path is None:
        output_path = jsonl_path
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)

    print(f"Patched {jsonl_path} -> {output_path}", file=sys.stderr)
    print(f"  verified_correct: {counts['verified_correct']}", file=sys.stderr)
    print(f"  kept_edited:      {counts['kept_edited']}", file=sys.stderr)
    print(f"  rejected:         {counts['rejected']}", file=sys.stderr)
    if pending:
        print(
            f"  still pending (status left unchanged): {len(pending)}",
            file=sys.stderr,
        )
    if unknown_status:
        print(
            f"  unrecognized status cells ({len(unknown_status)}): "
            f"{', '.join(unknown_status[:10])}"
            + (" ..." if len(unknown_status) > 10 else ""),
            file=sys.stderr,
        )
    if orphan_review_ids:
        print(
            f"  WARNING: {len(orphan_review_ids)} review alignment_id(s) not found "
            f"in canonical file: {', '.join(sorted(orphan_review_ids)[:10])}"
            + (" ..." if len(orphan_review_ids) > 10 else ""),
            file=sys.stderr,
        )
    if uncovered_ids:
        statuses_label = ", ".join(sorted(review_statuses))
        print(
            f"  WARNING: {len(uncovered_ids)} alignment(s) awaiting review "
            f"(status in {{{statuses_label}}}) have NO row in the review TSV "
            f"(lost in a spreadsheet round-trip?): "
            f"{', '.join(sorted(uncovered_ids)[:10])}"
            + (" ..." if len(uncovered_ids) > 10 else ""),
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export the unverified alignments of a canonical word-spotting file to "
            "a review TSV, or apply a reviewed TSV back in place by alignment_id."
        ),
    )
    parser.add_argument(
        "alignments",
        help="Path to the canonical {train,test}_alignments.jsonl file.",
    )
    parser.add_argument(
        "--apply",
        metavar="REVIEW_TSV",
        default=None,
        help="Apply mode: read this reviewed TSV and patch statuses back in place.",
    )
    parser.add_argument(
        "--statuses",
        default=UNVERIFIED_STATUS,
        help=(
            "Comma-separated status(es) to review (default: 'unverified'). "
            "Export filters to these statuses; apply uses them only for coverage "
            "accounting (the patch itself is keyed on alignment_id). Pass e.g. "
            "'verified_correct,kept_edited' to re-review already-resolved rows."
        ),
    )
    parser.add_argument(
        "--no-coverage-check",
        action="store_true",
        help=(
            "Apply mode: skip the 'awaiting review but missing from TSV' warning. "
            "Use when the TSV is a content-defined subset of a status (e.g. the "
            "English-mismatch TSV from clean_alignments.py)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Export mode: TSV output path (default stdout). "
            "Apply mode: write patched JSONL here instead of overwriting the input."
        ),
    )
    parser.add_argument(
        "--dictionary",
        default="data/gothic_dictionaries/koebler_gothic_english.json",
        help="Path to parsed dictionary JSON (export mode only).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Max edit distance to consider a match (default: 3).",
    )
    parser.add_argument(
        "--sort",
        choices=["worst-first", "best-first"],
        default="worst-first",
        help="Export sort order (default: worst-first, suspicious rows on top).",
    )
    parser.add_argument(
        "--normalize",
        "--strip-macrons",
        action="store_true",
        help="Normalize Koebler orthography before edit distance (recommended).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top dictionary matches to show (default: 3).",
    )
    args = parser.parse_args()

    review_statuses = frozenset(
        status.strip() for status in args.statuses.split(",") if status.strip()
    )
    if not review_statuses:
        print("Error: --statuses must name at least one status.", file=sys.stderr)
        sys.exit(1)

    if args.apply:
        apply_review(
            args.alignments,
            args.apply,
            args.output,
            review_statuses,
            coverage_check=not args.no_coverage_check,
        )
        return

    export_review_tsv(
        args.alignments,
        args.output,
        args.dictionary,
        args.threshold,
        args.normalize,
        args.top_n,
        args.sort,
        review_statuses,
    )


if __name__ == "__main__":
    main()
