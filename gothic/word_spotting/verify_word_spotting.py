#!/usr/bin/env python3
"""
Verify LLM-generated Gothic word-spotting alignments against the Koebler dictionary,
and finalize reviewed TSV back into JSONL.

Verification mode (default):
    For each alignment, computes edit distance between the Gothic surface form and
    dictionary citation forms, assigns a confidence tier, and outputs a sorted TSV
    for human review.

    Tiers:
        1 - Gothic form matches a citation form whose English gloss also matches
        2 - Gothic form matches some citation form, but not one matching the English
        3 - Gothic form doesn't match any citation form within the threshold

Finalize mode (--finalize):
    Reads a reviewed TSV (with status column marked by the reviewer), drops deleted
    rows, re-groups alignments by sentence pair, and writes clean JSONL.

Usage:
    # Verify
    python verify_word_spotting.py input.jsonl --output verified.tsv --normalize

    # [review in spreadsheet, set status column to correct/delete]

    # Finalize
    python verify_word_spotting.py --finalize verified.tsv --output final.jsonl
"""

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict

from gothic.orthography import normalize_orthography, transliterate_gothic_to_latin


def levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                prev_row[j + 1] + 1,
                curr_row[j] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row

    return prev_row[-1]


def strip_punctuation(surface_form: str) -> str:
    """Strip leading/trailing punctuation and lowercase a Gothic surface form."""
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", surface_form, flags=re.UNICODE)
    return cleaned.lower()


def load_dictionary(
    dict_path: str,
) -> tuple[list[tuple[str, str]], dict[str, set[str]]]:
    """Load the parsed Koebler dictionary and build lookup structures.

    Returns:
        Tuple of:
        - all_forms: list of (citation_form, glosses_display_str) for every unique
          citation form, with glosses merged across entries that share a form.
        - english_to_forms: dict mapping each lowercase English gloss word to
          the set of citation forms associated with that gloss.
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Merge glosses for citation forms that appear in multiple entries
    form_glosses: dict[str, set[str]] = {}
    for entry in entries:
        for gothic in entry["gothic_entries"]:
            cf = gothic["citation_form"]
            if cf not in form_glosses:
                form_glosses[cf] = set()
            for gloss in gothic["english_glosses"]:
                form_glosses[cf].add(gloss)

    all_forms = []
    for cf, glosses in form_glosses.items():
        glosses_str = ", ".join(sorted(glosses))
        all_forms.append((cf, glosses_str))

    # Build reverse index: English word -> set of citation forms
    english_to_forms: dict[str, set[str]] = {}
    for entry in entries:
        for gothic in entry["gothic_entries"]:
            cf = gothic["citation_form"]
            for gloss in gothic["english_glosses"]:
                for word in tokenize_english(gloss):
                    if word not in english_to_forms:
                        english_to_forms[word] = set()
                    english_to_forms[word].add(cf)

    return all_forms, english_to_forms


def tokenize_english(text: str) -> list[str]:
    """Tokenize an English string into lowercase words, stripping punctuation."""
    return [w.lower() for w in re.findall(r"[a-zA-Z]+", text)]


def english_words_match(
    target_word: str,
    english_to_forms: dict[str, set[str]],
) -> set[str]:
    """Find citation forms whose English glosses contain the target word.

    Returns the set of citation forms that have the target word (or any of its
    tokens) appearing in their English glosses.
    """
    forms = set()
    for word in tokenize_english(target_word):
        if word in english_to_forms:
            forms.update(english_to_forms[word])
    return forms


def find_top_matches(
    surface_form: str,
    candidates: list[tuple[str, str]],
    n: int,
    use_normalize: bool,
) -> list[tuple[int, str, str]]:
    """Find the top N closest citation forms to a surface form by edit distance.

    Args:
        surface_form: The cleaned Gothic surface form to match.
        candidates: List of (citation_form, glosses_str) tuples.
        n: Number of top matches to return.
        use_normalize: Whether to normalize orthography before comparing.

    Returns:
        List of (distance, original_citation_form, glosses_str), sorted by distance.
    """
    compare_surface = normalize_orthography(surface_form) if use_normalize else surface_form

    scored = []
    for citation_form, glosses_str in candidates:
        compare_cf = normalize_orthography(citation_form) if use_normalize else citation_form
        dist = levenshtein(compare_surface, compare_cf.lower())
        scored.append((dist, citation_form, glosses_str))

    scored.sort(key=lambda x: x[0])
    return scored[:n]


def assign_tier(
    best_global_dist: int,
    best_english_dist: int | None,
    threshold: int,
) -> int:
    """Assign a confidence tier based on match distances.

    Args:
        best_global_dist: Best edit distance from global match.
        best_english_dist: Best edit distance from English-filtered match,
            or None if no English-filtered candidates exist.
        threshold: Maximum edit distance to consider a match.

    Returns:
        Tier: 1 (high confidence), 2 (plausible form), or 3 (suspicious).
    """
    if best_english_dist is not None and best_english_dist <= threshold:
        return 1
    if best_global_dist <= threshold:
        return 2
    return 3


def format_match(distance: int, citation_form: str, glosses_str: str) -> str:
    """Format a match for display: 'citation_form: gloss1, gloss2, ...'"""
    max_gloss_len = 60
    if len(glosses_str) > max_gloss_len:
        glosses_str = glosses_str[:max_gloss_len] + "..."
    return f"{citation_form}: {glosses_str}"


def verify_alignments(
    jsonl_path: str,
    dict_path: str,
    threshold: int,
    use_normalize: bool,
    top_n: int = 3,
) -> list[dict]:
    """Verify all alignments in a word-spotting JSONL file.

    Returns:
        List of row dicts for TSV output.
    """
    all_forms, english_to_forms = load_dictionary(dict_path)

    form_to_glosses = {cf: gs for cf, gs in all_forms}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl_lines = [json.loads(line) for line in f if line.strip()]

    rows = []
    for entry in jsonl_lines:
        english_sentence = entry["english_sentence"]
        gothic_sentence_roman = entry["gothic_sentence_roman"]
        gothic_sentence_gothic = entry["gothic_sentence_gothic"]

        for alignment in entry["alignments"]:
            target_word = alignment["target_word"]
            gothic_surface = alignment["gothic_word_roman"]
            gothic_surface_gothic = alignment["gothic_word_gothic"]
            cleaned_surface = strip_punctuation(gothic_surface)

            if not cleaned_surface:
                print(
                    f"Warning: empty surface form after cleaning '{gothic_surface}', "
                    f"skipping",
                    file=sys.stderr,
                )
                continue

            # Global top-N matches
            global_matches = find_top_matches(
                cleaned_surface, all_forms, top_n, use_normalize,
            )

            # English-filtered matches
            english_filtered_forms = english_words_match(
                target_word, english_to_forms,
            )
            best_english_dist = None
            if english_filtered_forms:
                english_candidates = [
                    (cf, form_to_glosses[cf])
                    for cf in english_filtered_forms
                    if cf in form_to_glosses
                ]
                if english_candidates:
                    english_matches = find_top_matches(
                        cleaned_surface, english_candidates, 1, use_normalize,
                    )
                    if english_matches:
                        best_english_dist = english_matches[0][0]

            best_global_dist = global_matches[0][0] if global_matches else 999
            tier = assign_tier(best_global_dist, best_english_dist, threshold)

            # Check that Gothic script word transliterates to the Roman word
            transliterated = transliterate_gothic_to_latin(gothic_surface_gothic)
            script_match = "yes" if transliterated == gothic_surface else "no"

            row = {
                "tier": tier,
                "english_word": target_word,
                "gothic_surface": gothic_surface,
                "gothic_surface_gothic": gothic_surface_gothic,
                "script_ok": script_match,
            }

            # Add top-N match columns
            for i in range(top_n):
                idx = i + 1
                if i < len(global_matches):
                    dist, cf, glosses = global_matches[i]
                    row[f"dist_{idx}"] = dist
                    row[f"match_{idx}"] = format_match(dist, cf, glosses)
                else:
                    row[f"dist_{idx}"] = ""
                    row[f"match_{idx}"] = ""

            row["english_sentence"] = english_sentence
            row["gothic_sentence_roman"] = gothic_sentence_roman
            row["gothic_sentence_gothic"] = gothic_sentence_gothic
            row["status"] = "unchecked"

            rows.append(row)

    return rows


def finalize(tsv_path: str, output_path: str | None):
    """Read a reviewed verification TSV and write finalized JSONL.

    Drops rows with status 'delete', re-groups remaining alignments by sentence
    pair, and writes one JSONL line per sentence pair.
    """
    with open(tsv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = list(reader)

    # Filter: keep everything except explicitly deleted and unchecked rows
    delete_statuses = {"unchecked", "delete", "deleted", "drop", "remove", "x", "✗"}
    kept = []
    dropped = 0
    for row in rows:
        status = row.get("status", "").strip().lower()
        if status in delete_statuses:
            dropped += 1
        else:
            kept.append(row)

    # Re-group by sentence pair, preserving encounter order
    sentence_pairs: OrderedDict[tuple[str, str], dict] = OrderedDict()
    for row in kept:
        english_sentence = row["english_sentence"]
        gothic_sentence_roman = row["gothic_sentence_roman"]
        gothic_sentence_gothic = row["gothic_sentence_gothic"]
        key = (english_sentence, gothic_sentence_roman)

        if key not in sentence_pairs:
            sentence_pairs[key] = {
                "english_sentence": english_sentence,
                "gothic_sentence_roman": gothic_sentence_roman,
                "gothic_sentence_gothic": gothic_sentence_gothic,
                "alignments": [],
            }

        sentence_pairs[key]["alignments"].append({
            "target_word": row["english_word"],
            "gothic_word_roman": row["gothic_surface"],
            "gothic_word_gothic": row["gothic_surface_gothic"],
        })

    # Write JSONL
    if output_path:
        out_file = open(output_path, "w", encoding="utf-8")
    else:
        out_file = sys.stdout

    try:
        for entry in sentence_pairs.values():
            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    finally:
        if output_path:
            out_file.close()

    print(f"Rows kept: {len(kept)}, dropped: {dropped}", file=sys.stderr)
    print(
        f"Sentence pairs: {len(sentence_pairs)}, "
        f"total alignments: {sum(len(e['alignments']) for e in sentence_pairs.values())}",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify Gothic word-spotting alignments against the Koebler dictionary, "
            "or finalize a reviewed TSV back into JSONL."
        ),
    )
    parser.add_argument(
        "input",
        help="Path to input file (JSONL for verification, TSV for --finalize).",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Finalize mode: read reviewed TSV and write filtered JSONL.",
    )
    parser.add_argument(
        "--dictionary",
        default="data/gothic_dictionaries/koebler_gothic_english.json",
        help="Path to parsed dictionary JSON (verification mode only).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to stdout.",
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
        default="best-first",
        help="Sort order: worst-first (tier 3 first) or best-first (tier 1 first).",
    )
    parser.add_argument(
        "--normalize",
        "--strip-macrons",
        action="store_true",
        help=(
            "Normalize Koebler orthography before edit distance: strip macrons "
            "(ā→a), accent marks (aí→ai, aú→au), expand ƕ→hw."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top matches to show (default: 3).",
    )
    args = parser.parse_args()

    if args.finalize:
        finalize(args.input, args.output)
        return

    rows = verify_alignments(
        args.input,
        args.dictionary,
        args.threshold,
        args.normalize,
        args.top_n,
    )

    # Sort
    if args.sort == "worst-first":
        rows.sort(key=lambda r: (-r["tier"], -r.get("dist_1", 0)))
    else:
        rows.sort(key=lambda r: (r["tier"], r.get("dist_1", 0)))

    # Build fieldnames
    fieldnames = ["status", "tier", "english_word", "gothic_surface", "gothic_surface_gothic", "script_ok"]
    for i in range(1, args.top_n + 1):
        fieldnames.extend([f"dist_{i}", f"match_{i}"])
    fieldnames.extend([
        "english_sentence", "gothic_sentence_roman", "gothic_sentence_gothic",
    ])

    # Write TSV
    if args.output:
        out_file = open(args.output, "w", encoding="utf-8-sig", newline="")
    else:
        out_file = sys.stdout

    try:
        writer = csv.DictWriter(
            out_file, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore",
            quotechar=None, quoting=csv.QUOTE_NONE,
        )
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            out_file.close()

    # Statistics to stderr
    tier_counts = {1: 0, 2: 0, 3: 0}
    for row in rows:
        tier_counts[row["tier"]] += 1

    print(f"Total alignments:  {len(rows)}", file=sys.stderr)
    print(f"  Tier 1 (high):   {tier_counts[1]}", file=sys.stderr)
    print(f"  Tier 2 (form):   {tier_counts[2]}", file=sys.stderr)
    print(f"  Tier 3 (suspicious): {tier_counts[3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
