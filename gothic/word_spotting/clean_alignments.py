#!/usr/bin/env python3
"""
v0.5 cleanup pass over a canonical word-spotting alignments file.

Three independent operations, all keyed on the canonical
``{train,test}_alignments.jsonl`` schema (with ``alignment_id`` / ``status`` /
``source``):

1. **De-duplicate** within-sentence identical alignments. The diversification
   pass occasionally returned the same ``(target_word, gothic_word)`` twice for
   one sentence; ``assign_alignment_ids.py`` then handed each a distinct ``_dN``
   id. These inflate the trainable set with no new signal. Among the "live"
   statuses (``verified_correct`` / ``kept_edited`` / ``unverified``), identical
   alignments are collapsed to one, keeping the best-verified member (and the
   earliest id as a tiebreak). Rejected / diversification-replaced records are
   never touched — their status is intentional provenance.

2. **Clean Gothic-side editorial artifacts** (restoration parentheses, Arabic
   numeral glosses, enclitic tildes, trailing dashes) from every Gothic
   sentence and surface field, via ``orthography.clean_gothic_artifacts``. The
   same transform is applied to a sentence and to the surfaces drawn from it, so
   a surface that matched its sentence before still matches after. English
   fields are never modified — they carry genuine whitespace-spanning
   parentheticals that are part of the translation.

3. **Report English-side mismatches** (review aid, no mutation). For each
   trainable alignment, check whether the English ``target_word`` actually
   occurs verbatim in ``english_sentence``: a one-word target must appear as a
   token; a multi-word target must have its longest (probable content) word
   appear. Misses are written to a review TSV — they are *not* auto-dropped,
   because the Gothic↔target relation can still be sound when the English gloss
   is non-verbatim.

   The mismatch TSV is written in ``review_unverified.py``'s **apply schema**
   (``status`` / ``alignment_id`` / ``english_word`` / ``gothic_surface`` /
   ``gothic_surface_gothic`` + context), with ``status`` pre-filled to each
   alignment's current value. Review it in a spreadsheet (set ``status`` to
   ``reject``, or edit ``english_word`` to fix the gloss), then patch the
   decisions back with ``review_unverified --apply`` — its apply is keyed purely
   on ``alignment_id`` and is status-agnostic, so it handles these
   ``verified_correct`` rows without modification.

The input file is rewritten in place (temp + ``os.replace``); every untouched
alignment stays byte-identical apart from the artifact cleanup.

Usage:
    # bulk pass: dedupe + clean artifacts, emit the English-mismatch review TSV
    python -m gothic.word_spotting.clean_alignments \
        data/gothic_word_spotting/train_alignments.jsonl \
        --english-mismatch-tsv \
            data/gothic_word_spotting/verification/train_english_mismatch_v05.tsv

    # preview without writing
    python -m gothic.word_spotting.clean_alignments train_alignments.jsonl --dry-run

    # [review the mismatch TSV in a spreadsheet, then patch the decisions back;
    #  --no-coverage-check because this TSV is a content-defined subset of a
    #  status, not the whole status]
    python -m gothic.word_spotting.review_unverified \
        data/gothic_word_spotting/train_alignments.jsonl \
        --apply data/gothic_word_spotting/verification/train_english_mismatch_v05.tsv \
        --no-coverage-check
"""

import argparse
import csv
import json
import os
import string
import sys
import tempfile
from pathlib import Path

from gothic.orthography import clean_gothic_artifacts

# Statuses eligible for de-duplication. Rejected / replaced records are left
# alone so their provenance survives.
LIVE_STATUSES = {"verified_correct", "kept_edited", "unverified"}

# Statuses whose alignments train the model — the set the English check covers.
TRAINABLE_STATUSES = {"verified_correct", "kept_edited"}

# Preference when collapsing a duplicate group: keep the most-verified member.
STATUS_RANK = {"kept_edited": 2, "verified_correct": 1, "unverified": 0}

# Gothic text fields cleaned on the parent entry.
SENTENCE_FIELDS = ("gothic_sentence_roman", "gothic_sentence_gothic")

# Gothic surface fields cleaned on each alignment (and its ``original`` stash).
SURFACE_FIELDS = ("gothic_word_roman", "gothic_word_gothic")


def _norm_english(word: str) -> str:
    """Lowercase a word and strip surrounding punctuation/quotes for comparison."""
    return word.strip(string.punctuation).lower()


def clean_entry_text(entry: dict) -> int:
    """Clean Gothic artifacts in an entry's sentences and surfaces, in place.

    Args:
        entry: A canonical word-spotting entry.

    Returns:
        The number of string fields whose value changed.
    """
    changed = 0
    for field in SENTENCE_FIELDS:
        cleaned = clean_gothic_artifacts(entry[field])
        if cleaned != entry[field]:
            entry[field] = cleaned
            changed += 1

    for alignment in entry["alignments"]:
        targets = [alignment]
        if isinstance(alignment.get("original"), dict):
            targets.append(alignment["original"])
        for target in targets:
            for field in SURFACE_FIELDS:
                if field not in target:
                    continue
                cleaned = clean_gothic_artifacts(target[field])
                if cleaned != target[field]:
                    target[field] = cleaned
                    changed += 1

    return changed


def dedupe_entry(entry: dict) -> list[tuple]:
    """Collapse within-sentence identical live alignments, in place.

    Args:
        entry: A canonical word-spotting entry (surfaces already cleaned).

    Returns:
        A list of (sentence_id, key, kept_id, dropped_ids) tuples, one per
        collapsed group.
    """
    alignments = entry["alignments"]
    groups: dict[tuple, list[int]] = {}
    for index, alignment in enumerate(alignments):
        if alignment["status"] not in LIVE_STATUSES:
            continue
        key = (
            alignment["target_word"],
            alignment["gothic_word_roman"],
            alignment["gothic_word_gothic"],
        )
        groups.setdefault(key, []).append(index)

    drop_indices: set[int] = set()
    report: list[tuple] = []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        # Keep the most-verified member; tiebreak on the earliest alignment_id.
        keep = max(
            indices,
            key=lambda i: (
                STATUS_RANK[alignments[i]["status"]],
                -i,
            ),
        )
        dropped = [i for i in indices if i != keep]
        drop_indices.update(dropped)
        report.append(
            (
                entry["sentence_id"],
                key,
                alignments[keep]["alignment_id"],
                [alignments[i]["alignment_id"] for i in dropped],
            )
        )

    if drop_indices:
        entry["alignments"] = [
            alignment
            for index, alignment in enumerate(alignments)
            if index not in drop_indices
        ]
    return report


def english_mismatches(entry: dict) -> list[dict]:
    """Find trainable alignments whose English target is not verbatim in the text.

    Args:
        entry: A canonical word-spotting entry.

    Returns:
        A list of mismatch rows (dicts) ready for the review TSV.
    """
    sentence_tokens = {
        _norm_english(token) for token in entry["english_sentence"].split()
    }
    rows: list[dict] = []
    for alignment in entry["alignments"]:
        if alignment["status"] not in TRAINABLE_STATUSES:
            continue
        target_words = alignment["target_word"].split()
        if len(target_words) == 1:
            probe = _norm_english(target_words[0])
            reason = "single_word"
        else:
            probe = _norm_english(max(target_words, key=len))
            reason = f"longest={probe}"
        if probe and probe not in sentence_tokens:
            # Column names match review_unverified.py's apply reader so the TSV
            # can be reviewed and patched back through that tool unmodified.
            rows.append(
                {
                    "status": alignment["status"],
                    "alignment_id": alignment["alignment_id"],
                    "source": alignment.get("source", ""),
                    "reason": reason,
                    "english_word": alignment["target_word"],
                    "gothic_surface": alignment["gothic_word_roman"],
                    "gothic_surface_gothic": alignment["gothic_word_gothic"],
                    "english_sentence": entry["english_sentence"],
                    "gothic_sentence_roman": entry["gothic_sentence_roman"],
                }
            )
    return rows


def write_jsonl_atomic(path: Path, entries: list[dict]) -> None:
    """Write entries to a JSONL file atomically (temp + os.replace)."""
    directory = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        delete=False,
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)


def main():
    parser = argparse.ArgumentParser(
        description="v0.5 dedupe + Gothic-artifact cleanup for a word-spotting "
        "alignments file.",
    )
    parser.add_argument(
        "input",
        help="Canonical {train,test}_alignments.jsonl to clean in place.",
    )
    parser.add_argument(
        "--english-mismatch-tsv",
        default=None,
        help="Path to write the English-target review TSV (default: skip).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without rewriting the input file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]

    fields_changed = 0
    dedupe_report: list[tuple] = []
    mismatch_rows: list[dict] = []
    for entry in entries:
        fields_changed += clean_entry_text(entry)
        dedupe_report.extend(dedupe_entry(entry))
        mismatch_rows.extend(english_mismatches(entry))

    dropped_total = sum(len(dropped) for _, _, _, dropped in dedupe_report)

    if args.english_mismatch_tsv and mismatch_rows:
        tsv_path = Path(args.english_mismatch_tsv)
        columns = [
            "status",
            "alignment_id",
            "source",
            "reason",
            "english_word",
            "gothic_surface",
            "gothic_surface_gothic",
            "english_sentence",
            "gothic_sentence_roman",
        ]
        # utf-8-sig + QUOTE_MINIMAL match review_unverified.py's export side so
        # the Google Sheets round-trip behaves identically (see its apply docs).
        with open(tsv_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                delimiter="\t",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            writer.writerows(mismatch_rows)

    if not args.dry_run:
        write_jsonl_atomic(input_path, entries)

    # Summary to stderr; the dedupe detail goes to stdout for the record.
    for sentence_id, key, kept_id, dropped_ids in dedupe_report:
        print(
            f"dedupe {sentence_id}: kept {kept_id}, dropped {', '.join(dropped_ids)} "
            f"({key[0]!r} -> {key[1]!r})"
        )

    mode = "DRY RUN — no file written" if args.dry_run else f"rewrote {input_path}"
    print(
        f"{input_path.name}: {len(entries)} sentences | "
        f"cleaned {fields_changed} Gothic field(s) | "
        f"collapsed {len(dedupe_report)} duplicate group(s) "
        f"({dropped_total} alignment(s) dropped) | "
        f"{len(mismatch_rows)} English mismatch(es)"
        + (
            f" -> {args.english_mismatch_tsv}"
            if args.english_mismatch_tsv and mismatch_rows
            else ""
        )
        + f" | {mode}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
