#!/usr/bin/env python3
"""
Prepare batch request JSONL for Gothic word-spotting annotation via the Claude API.

Two modes:

original (default) — first-pass annotation:
    Reads a prepared translation file (produced by gothic/data/prepare_gothic_data.py
    with --translation-script both --translation-direction eng_to_gothic
    --delimiter ' || ') containing consecutive pairs of lines:
        English || Gothic (Roman script)
        English || Gothic (Gothic script)
    Groups these pairs into batches and emits API requests.

diversify — second-pass realignment of over-used pairs:
    Reads a verified word-spotting JSONL (with alignments), identifies
    (Gothic, English) surface pairs whose count meets --focal-threshold, and
    for each such "focal" pair emits batches of API requests asking the model
    to pick alternative alignments from the same sentences. Pairs with count
    >= --avoid-threshold are listed in each request as additional pairs to
    avoid producing.

Usage:
    # Original mode: first prepare input, then create requests
    python -m gothic.data.prepare_gothic_data --data-types translation \\
        --translation-script both --translation-direction eng_to_gothic \\
        --delimiter ' || ' --splits train
    python -m gothic.word_spotting.prepare_batches \\
        --mode original \\
        --input data/gothic_prepared/translation_all-codices_both-scripts_eng-to-gothic.txt

    # Diversify mode
    python -m gothic.word_spotting.prepare_batches \\
        --mode diversify \\
        --alignments data/gothic_word_spotting/train_verified_b.jsonl \\
        --output data/gothic_word_spotting/diversify_batch_requests.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_TOKENS = 4096
DEFAULT_OUTPUT = "data/gothic_word_spotting/raw_llm/batch_requests.jsonl"
DEFAULT_DIVERSIFY_OUTPUT = (
    "data/gothic_word_spotting/diversification/diversify_batch_requests.jsonl"
)
DEFAULT_FOCAL_THRESHOLD = 10
DEFAULT_AVOID_THRESHOLD = 5
DELIMITER = " || "


def parse_pairs(input_path: str) -> list[tuple[str, str]]:
    """Parse a prepared translation file into consecutive line pairs.

    Each pair consists of:
        line 1: English || Gothic (Roman script)
        line 2: English || Gothic (Gothic script)

    Validates that both lines in a pair share the same English sentence.

    Returns:
        List of (roman_line, gothic_script_line) tuples.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    if len(lines) % 2 != 0:
        print(
            f"Warning: odd number of lines ({len(lines)}), last line will be skipped",
            file=sys.stderr,
        )

    pairs = []
    for i in range(0, len(lines) - 1, 2):
        roman_line = lines[i]
        gothic_line = lines[i + 1]

        # Validate that English portions match
        if DELIMITER in roman_line and DELIMITER in gothic_line:
            english_roman = roman_line.split(DELIMITER, 1)[0]
            english_gothic = gothic_line.split(DELIMITER, 1)[0]
            if english_roman != english_gothic:
                print(
                    f"Warning: English mismatch at lines {i + 1}-{i + 2}: "
                    f"{english_roman!r} vs {english_gothic!r}",
                    file=sys.stderr,
                )
        else:
            print(
                f"Warning: missing delimiter at lines {i + 1}-{i + 2}",
                file=sys.stderr,
            )

        pairs.append((roman_line, gothic_line))

    return pairs


def make_request(
    custom_id: str,
    user_content: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        },
    }


def build_original_requests(
    pairs: list[tuple[str, str]],
    system_prompt: str,
    batch_size: int,
    model: str,
    max_tokens: int,
) -> list[dict]:
    requests = []
    for batch_idx in range(0, len(pairs), batch_size):
        batch = pairs[batch_idx:batch_idx + batch_size]
        # join each pair into two lines, blank-line-separate the pairs
        formatted_pairs = [f"{roman}\n{gothic}" for roman, gothic in batch]
        user_content = "\n".join(formatted_pairs)
        custom_id = f"batch_{batch_idx // batch_size:04d}"
        requests.append(
            make_request(custom_id, user_content, system_prompt, model, max_tokens)
        )
    return requests


def load_verified_alignments(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_pair(alignment: dict) -> tuple[str, str]:
    gothic = alignment["gothic_word_roman"].strip().lower()
    english = alignment["target_word"].strip().lower()
    return gothic, english


def count_pairs(records_by_source: dict[str, list[dict]]) -> Counter:
    counts: Counter = Counter()
    for records in records_by_source.values():
        for record in records:
            for alignment in record.get("alignments", []):
                counts[normalize_pair(alignment)] += 1
    return counts


def index_sentences_by_pair(
    records_by_source: dict[str, list[dict]],
) -> dict[tuple[str, str], dict[str, list[int]]]:
    """For each pair, map source file path → list of record indices in that file."""
    index: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for source, records in records_by_source.items():
        for i, record in enumerate(records):
            seen_in_record: set[tuple[str, str]] = set()
            for alignment in record.get("alignments", []):
                pair = normalize_pair(alignment)
                if pair not in seen_in_record:
                    index[pair][source].append(i)
                    seen_in_record.add(pair)
    return index


def format_avoid_list(
    avoid_pairs: list[tuple[tuple[str, str], int]],
    focal_pair: tuple[str, str],
) -> str:
    # exclude the focal pair from the avoid list; it is named separately
    lines = []
    for (gothic, english), count in avoid_pairs:
        if (gothic, english) == focal_pair:
            continue
        lines.append(f"  {gothic} ↔ {english}  (count: {count})")
    return "\n".join(lines)


def format_sentence_block(
    record: dict,
    focal_pair: tuple[str, str],
    sentence_idx: int,
) -> str:
    # tag the focal alignment <TO REPLACE> and the rest <KEEP>
    alignment_lines = []
    for alignment in record.get("alignments", []):
        pair = normalize_pair(alignment)
        target = alignment["target_word"]
        gothic = alignment["gothic_word_roman"]
        tag = "<TO REPLACE>" if pair == focal_pair else "<KEEP>"
        alignment_lines.append(f"    {tag} {target} ↔ {gothic}")
    alignments_block = "\n".join(alignment_lines) if alignment_lines else "    (none)"
    return (
        f"SENTENCE {sentence_idx}\n"
        f"  english: {record['english_sentence']}\n"
        f"  gothic_roman: {record['gothic_sentence_roman']}\n"
        f"  gothic_gothic: {record['gothic_sentence_gothic']}\n"
        f"  existing alignments:\n"
        f"{alignments_block}"
    )


def build_diversify_requests(
    records_by_source: dict[str, list[dict]],
    system_prompt: str,
    batch_size: int,
    model: str,
    max_tokens: int,
    focal_threshold: int,
    avoid_threshold: int,
) -> tuple[list[dict], list[dict]]:
    """Build diversification batch requests across one or more verified files.

    Counts and the focal/avoid lists are computed over the union of all source
    files; this avoids the failure mode where the same pair is an offender in
    one split but falls below threshold in another.

    Batches are emitted **per source file** (a single batch never mixes
    records from different files), so the downstream merge can write each
    response back into the correct verified output.

    Returns:
        (requests, manifest) — manifest entries map each custom_id to its
        focal pair, source file, and the record indices within that source
        file.
    """
    counts = count_pairs(records_by_source)
    sentence_index = index_sentences_by_pair(records_by_source)

    focal_pairs = sorted(
        [(pair, c) for pair, c in counts.items() if c >= focal_threshold],
        key=lambda x: (-x[1], x[0]),
    )
    avoid_pairs = sorted(
        [(pair, c) for pair, c in counts.items() if c >= avoid_threshold],
        key=lambda x: (-x[1], x[0]),
    )

    print(
        f"  {len(focal_pairs)} focal pairs (count >= {focal_threshold})",
        file=sys.stderr,
    )
    print(
        f"  {len(avoid_pairs)} pairs on global avoid list (count >= {avoid_threshold})",
        file=sys.stderr,
    )

    requests: list[dict] = []
    manifest: list[dict] = []

    # iterate source files in the order they appeared on the CLI for stable IDs
    source_order = list(records_by_source.keys())

    for focal_idx, (focal_pair, focal_count) in enumerate(focal_pairs):
        gothic, english = focal_pair
        avoid_text = format_avoid_list(avoid_pairs, focal_pair)

        for source_idx, source in enumerate(source_order):
            record_indices = sentence_index[focal_pair].get(source, [])
            if not record_indices:
                continue
            records = records_by_source[source]

            for batch_idx, start in enumerate(range(0, len(record_indices), batch_size)):
                batch_record_indices = record_indices[start:start + batch_size]
                sentence_blocks = []
                for sentence_idx, record_idx in enumerate(batch_record_indices, start=1):
                    sentence_blocks.append(
                        format_sentence_block(
                            records[record_idx], focal_pair, sentence_idx
                        )
                    )

                user_content = (
                    f"OVER-USED PAIR (focal): {gothic} ↔ {english}  "
                    f"(global count: {focal_count})\n\n"
                    f"ALSO AVOID CREATING NEW INSTANCES OF "
                    f"(already over-represented globally):\n"
                    f"{avoid_text}\n\n"
                    + "\n\n".join(sentence_blocks)
                )

                custom_id = (
                    f"diversify_f{focal_idx:03d}_s{source_idx:01d}_b{batch_idx:04d}"
                )
                requests.append(
                    make_request(
                        custom_id, user_content, system_prompt, model, max_tokens
                    )
                )
                manifest.append(
                    {
                        "custom_id": custom_id,
                        "focal_pair": {"gothic": gothic, "english": english},
                        "focal_count": focal_count,
                        "source": source,
                        "record_indices": batch_record_indices,
                    }
                )

    return requests, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare batch request JSONL for Gothic word-spotting annotation.",
    )
    parser.add_argument(
        "--mode",
        choices=["original", "diversify"],
        default="original",
        help="original: first-pass annotation; diversify: realign over-used pairs.",
    )
    parser.add_argument(
        "--input",
        help="(original mode) Prepared translation file from prepare_gothic_data.py.",
    )
    parser.add_argument(
        "--alignments",
        nargs="+",
        help=(
            "(diversify mode) One or more verified word-spotting JSONL files. "
            "Counts and offender lists are computed over the union of all "
            "files; batches are still emitted per-source file so the merge "
            "step can write each response back to the correct file."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Path to system prompt file. Defaults: prompt.txt for original mode, "
            "prompt_diversify.txt for diversify mode."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL path. Defaults: "
            f"{DEFAULT_OUTPUT} (original), {DEFAULT_DIVERSIFY_OUTPUT} (diversify)."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "(diversify mode) Sidecar JSON path mapping each custom_id to its "
            "focal pair and the record indices it covers. Defaults to "
            "<output>.manifest.json."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of sentence pairs per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--focal-threshold",
        type=int,
        default=DEFAULT_FOCAL_THRESHOLD,
        help=(
            "(diversify mode) Minimum global count for a pair to receive a "
            f"diversification request (default: {DEFAULT_FOCAL_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--avoid-threshold",
        type=int,
        default=DEFAULT_AVOID_THRESHOLD,
        help=(
            "(diversify mode) Minimum global count for a pair to appear in the "
            f"per-request avoid list (default: {DEFAULT_AVOID_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per response (default: {DEFAULT_MAX_TOKENS})",
    )
    args = parser.parse_args()

    prompt_dir = Path(__file__).resolve().parent
    if args.prompt is None:
        args.prompt = str(
            prompt_dir / ("prompt.txt" if args.mode == "original" else "prompt_diversify.txt")
        )
    if args.output is None:
        args.output = (
            DEFAULT_OUTPUT if args.mode == "original" else DEFAULT_DIVERSIFY_OUTPUT
        )

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "original":
        if not args.input:
            print("Error: --input is required in original mode.", file=sys.stderr)
            sys.exit(1)
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Reading pairs from {input_path}...", file=sys.stderr)
        pairs = parse_pairs(str(input_path))
        print(f"  {len(pairs)} sentence pairs", file=sys.stderr)

        requests = build_original_requests(
            pairs,
            system_prompt,
            args.batch_size,
            args.model,
            args.max_tokens,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            for request in requests:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        print(
            f"\nWrote {len(requests)} batch requests to {output_path}",
            file=sys.stderr,
        )
        print(
            f"  {len(pairs)} total pairs in {len(requests)} batches",
            file=sys.stderr,
        )
        print(
            f"  Batch size: {args.batch_size}, Model: {args.model}",
            file=sys.stderr,
        )
        return

    # diversify mode
    if not args.alignments:
        print("Error: --alignments is required in diversify mode.", file=sys.stderr)
        sys.exit(1)

    records_by_source: dict[str, list[dict]] = {}
    for raw_path in args.alignments:
        path = Path(raw_path)
        if not path.exists():
            print(f"Error: alignments file not found: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"Reading alignments from {path}...", file=sys.stderr)
        records = load_verified_alignments(path)
        print(f"  {len(records)} sentence records from {path}", file=sys.stderr)
        records_by_source[str(path)] = records

    requests, manifest = build_diversify_requests(
        records_by_source,
        system_prompt,
        args.batch_size,
        args.model,
        args.max_tokens,
        args.focal_threshold,
        args.avoid_threshold,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    manifest_path = (
        Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "alignments_sources": list(records_by_source.keys()),
                "focal_threshold": args.focal_threshold,
                "avoid_threshold": args.avoid_threshold,
                "batch_size": args.batch_size,
                "entries": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nWrote {len(requests)} batch requests to {output_path}",
        file=sys.stderr,
    )
    print(f"Wrote manifest to {manifest_path}", file=sys.stderr)
    print(
        f"  Batch size: {args.batch_size}, Model: {args.model}, "
        f"max_tokens: {args.max_tokens}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
