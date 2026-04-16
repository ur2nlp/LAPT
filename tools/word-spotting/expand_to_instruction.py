#!/usr/bin/env python3
"""
Expand finalized word-spotting JSONL into instruction-tuning format.

Each alignment in the input produces one or two instruction examples (depending
on the --script flag), with the prompt providing the English sentence and asking
the learner to identify the Gothic word for a target English word.

Input format (from verify_word_spotting.py --finalize):
    {"english_sentence": "...", "gothic_sentence_roman": "...",
     "gothic_sentence_gothic": "...",
     "alignments": [{"target_word": "...", "gothic_word_roman": "...",
                      "gothic_word_gothic": "..."}, ...]}

Output format (instruction_jsonl, one line per example):
    {"prompt": "...\\nResponse:", "response": " gothic_word"}

Usage:
    python tools/word-spotting/expand_to_instruction.py \
        --input data/gothic_word_spotting/train_verified_a.jsonl \
        --output data/gothic_word_spotting/train_word_spotting.jsonl

    # Roman script only
    python tools/word-spotting/expand_to_instruction.py \
        --input finalized.jsonl --script roman

    # Random script per alignment
    python tools/word-spotting/expand_to_instruction.py \
        --input finalized.jsonl --script random --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path


PROMPT_TEMPLATE = (
    'In the following Gothic sentence, find the Gothic word for "{target_word}".\n'
    "English: {english_sentence}\n"
    "Gothic: {gothic_sentence}\n"
    "Response:"
)

SCRIPT_FIELDS = {
    "roman": ("gothic_sentence_roman", "gothic_word_roman"),
    "gothic": ("gothic_sentence_gothic", "gothic_word_gothic"),
}


def expand_entry(
    entry: dict,
    scripts: list[str],
    rng: random.Random | None,
) -> list[dict]:
    """Expand one word-spotting entry into instruction examples.

    Args:
        entry: A finalized word-spotting JSONL entry.
        scripts: List of script keys to produce (from SCRIPT_FIELDS).
            For "random" mode, this will be ["roman", "gothic"] and rng will pick one.
        rng: Random instance for "random" mode, or None for deterministic modes.

    Returns:
        List of {"prompt": ..., "response": ...} dicts.
    """
    examples = []
    for alignment in entry["alignments"]:
        if rng is not None:
            selected_scripts = [rng.choice(scripts)]
        else:
            selected_scripts = scripts

        for script_key in selected_scripts:
            sentence_field, word_field = SCRIPT_FIELDS[script_key]
            prompt = PROMPT_TEMPLATE.format(
                target_word=alignment["target_word"],
                english_sentence=entry["english_sentence"],
                gothic_sentence=entry[sentence_field],
            )
            examples.append({
                "prompt": prompt,
                "response": f" {alignment[word_field]}",
            })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Expand word-spotting JSONL into instruction-tuning format.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to finalized word-spotting JSONL.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for instruction JSONL. Defaults to stdout.",
    )
    parser.add_argument(
        "--script",
        choices=["roman", "gothic", "both", "random"],
        default="both",
        help=(
            "Which Gothic script(s) to produce examples for. "
            "'both' emits one example per script per alignment. "
            "'random' picks one script per alignment. (default: both)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --script random (default: 42).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if args.script == "both":
        scripts = ["roman", "gothic"]
        rng = None
    elif args.script == "random":
        scripts = ["roman", "gothic"]
        rng = random.Random(args.seed)
    else:
        scripts = [args.script]
        rng = None

    examples = []
    for entry in entries:
        examples.extend(expand_entry(entry, scripts, rng))

    if args.output:
        out_file = open(args.output, "w", encoding="utf-8")
    else:
        out_file = sys.stdout

    try:
        for example in examples:
            out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out_file.close()

    total_alignments = sum(len(e["alignments"]) for e in entries)
    print(
        f"Expanded {len(entries)} sentence pairs ({total_alignments} alignments) "
        f"into {len(examples)} instruction examples (script={args.script})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
