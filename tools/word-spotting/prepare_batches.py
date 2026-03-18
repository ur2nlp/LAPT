#!/usr/bin/env python3
"""
Prepare batch request JSONL for Gothic word-spotting annotation via the Claude API.

Reads a prepared translation file (produced by tools/prepare_gothic_data.py with
--translation-script both --translation-direction eng_to_gothic --delimiter ' || ')
containing consecutive pairs of lines:
    English || Gothic (Roman script)
    English || Gothic (Gothic script)

Groups these pairs into batches and writes Anthropic Batch API request objects as JSONL.

Usage:
    # First, prepare the input file:
    python tools/prepare_gothic_data.py --data-types translation \\
        --translation-script both --translation-direction eng_to_gothic \\
        --delimiter ' || ' --splits train

    # Then, create batch requests:
    python -m tools.word_spotting.prepare_batches \\
        --input data/gothic_prepared/translation_all-codices_both-scripts_eng-to-gothic.txt
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_TOKENS = 4096
DEFAULT_OUTPUT = "data/gothic_word_spotting/batch_requests.jsonl"
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


def build_batch_requests(
    pairs: list[tuple[str, str]],
    system_prompt: str,
    batch_size: int,
    model: str,
    max_tokens: int,
) -> list[dict]:
    """Group sentence pairs into batch API request objects.

    Args:
        pairs: List of (roman_line, gothic_script_line) tuples.
        system_prompt: System prompt text.
        batch_size: Number of sentence pairs per batch.
        model: Model ID.
        max_tokens: Max tokens per response.

    Returns:
        List of batch request dicts, each containing custom_id and params.
    """
    requests = []
    for batch_idx in range(0, len(pairs), batch_size):
        batch = pairs[batch_idx:batch_idx + batch_size]

        # Each pair is two lines; join pairs with a blank line for readability
        formatted_pairs = []
        for roman_line, gothic_line in batch:
            formatted_pairs.append(f"{roman_line}\n{gothic_line}")

        user_content = "\n".join(formatted_pairs)

        request = {
            "custom_id": f"batch_{batch_idx // batch_size:04d}",
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
            },
        }
        requests.append(request)

    return requests


def main():
    parser = argparse.ArgumentParser(
        description="Prepare batch request JSONL for Gothic word-spotting annotation.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to prepared translation file from prepare_gothic_data.py "
            "(with --delimiter ' || ' --translation-script both "
            "--translation-direction eng_to_gothic)"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=str(Path(__file__).resolve().parent / "prompt.txt"),
        help="Path to system prompt file (default: tools/word_spotting/prompt.txt)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of sentence pairs per batch (default: {DEFAULT_BATCH_SIZE})",
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

    # Parse input pairs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading pairs from {input_path}...", file=sys.stderr)
    pairs = parse_pairs(str(input_path))
    print(f"  {len(pairs)} sentence pairs", file=sys.stderr)

    # Load system prompt
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    # Build batch requests
    requests = build_batch_requests(
        pairs,
        system_prompt,
        args.batch_size,
        args.model,
        args.max_tokens,
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(requests)} batch requests to {output_path}", file=sys.stderr)
    print(f"  {len(pairs)} total pairs in {len(requests)} batches", file=sys.stderr)
    print(
        f"  Batch size: {args.batch_size}, Model: {args.model}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
