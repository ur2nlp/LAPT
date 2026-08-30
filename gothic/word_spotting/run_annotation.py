#!/usr/bin/env python3
"""
Submit Gothic word-spotting annotation requests to the Claude API and collect results.

Two modes:
  - sync:  Sequential Messages API calls. Good for testing small subsets.
  - batch: Anthropic Batch API (50% cost savings). Good for full dataset.

Both modes read the same batch request JSONL produced by prepare_batches.py.

Output is JSONL with one line per batch, containing the model's response. A post-processing
step extracts the JSONL content from responses and writes a single file ready for
verify_word_spotting.py.

Usage:
    # Test on first batch
    python -m gothic.word_spotting.run_annotation --input batch_requests.jsonl --range 0:1

    # Full run with Batch API
    python -m gothic.word_spotting.run_annotation --mode batch --input batch_requests.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    print(
        "Error: anthropic package not found. "
        "Activate the gothic-data environment (see data_environment.yml).",
        file=sys.stderr,
    )
    sys.exit(1)


def load_batch_requests(
    input_path: str,
    range_spec: str | None = None,
) -> list[dict]:
    """Load batch request JSONL, optionally filtering to a range.

    Args:
        input_path: Path to batch request JSONL file.
        range_spec: Optional "START:END" string for slicing batches.

    Returns:
        List of batch request dicts.
    """
    with open(input_path, encoding="utf-8") as f:
        requests = [json.loads(line) for line in f if line.strip()]

    if range_spec:
        parts = range_spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(requests)
        requests = requests[start:end]

    return requests


def run_sync(
    requests: list[dict],
    api_key: str | None,
    max_concurrent: int,
) -> list[dict]:
    """Run annotation requests sequentially via the Messages API.

    Args:
        requests: List of batch request dicts.
        api_key: Anthropic API key (or None to use env var).
        max_concurrent: Not yet used (placeholder for future async).

    Returns:
        List of result dicts with custom_id and response content.
    """
    client = anthropic.Anthropic(api_key=api_key)
    results = []

    for i, request in enumerate(requests):
        custom_id = request["custom_id"]
        params = request["params"]

        print(
            f"  [{i + 1}/{len(requests)}] Processing {custom_id}...",
            file=sys.stderr,
        )

        message = client.messages.create(
            model=params["model"],
            max_tokens=params["max_tokens"],
            system=params["system"],
            messages=params["messages"],
        )

        # Extract text content from response
        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        result = {
            "custom_id": custom_id,
            "response_text": response_text,
            "model": message.model,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
            "stop_reason": message.stop_reason,
        }
        results.append(result)

        print(
            f"    {message.usage.input_tokens} in / {message.usage.output_tokens} out"
            f" ({message.stop_reason})",
            file=sys.stderr,
        )

    return results


def run_batch(
    requests: list[dict],
    api_key: str | None,
    poll_interval: int,
) -> list[dict]:
    """Submit requests via the Anthropic Batch API and poll for results.

    Args:
        requests: List of batch request dicts.
        api_key: Anthropic API key (or None to use env var).
        poll_interval: Seconds between status checks.

    Returns:
        List of result dicts with custom_id and response content.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Format requests for the Batch API
    batch_requests = []
    for request in requests:
        params = request["params"]
        batch_requests.append({
            "custom_id": request["custom_id"],
            "params": {
                "model": params["model"],
                "max_tokens": params["max_tokens"],
                "system": [{"type": "text", "text": params["system"]}],
                "messages": params["messages"],
            },
        })

    print(f"Submitting batch with {len(batch_requests)} requests...", file=sys.stderr)
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"  Batch ID: {batch.id}", file=sys.stderr)

    # Poll for completion
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        print(
            f"  Status: {batch.processing_status} "
            f"({counts.succeeded}/{total} succeeded, "
            f"{counts.errored} errored)",
            file=sys.stderr,
        )

        if batch.processing_status == "ended":
            break

        time.sleep(poll_interval)

    # Collect results
    print("Collecting results...", file=sys.stderr)
    results = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            message = result.result.message
            response_text = ""
            for block in message.content:
                if block.type == "text":
                    response_text += block.text

            results.append({
                "custom_id": result.custom_id,
                "response_text": response_text,
                "model": message.model,
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                },
                "stop_reason": message.stop_reason,
            })
        else:
            print(
                f"  Warning: {result.custom_id} failed: {result.result.type}",
                file=sys.stderr,
            )

    print(f"  Collected {len(results)} results", file=sys.stderr)
    return results


def extract_annotations(results: list[dict]) -> list[str]:
    """Extract JSONL annotation lines from model response texts.

    Accepts both original-task responses (with an "alignments" key) and
    diversification-task responses (with a "new_alignments" key). For
    diversification responses, the source request's custom_id is attached as
    "_custom_id" so downstream merge logic can map responses back to records
    via the diversify manifest.

    Uses raw_decode to robustly find JSON objects in the response, handling
    multiple objects per line, markdown code fences, commentary text, and
    truncated objects.

    Returns:
        List of raw JSON line strings (one per annotated sentence pair).
    """
    decoder = json.JSONDecoder()
    jsonl_lines = []

    for result in results:
        text = result["response_text"]
        custom_id = result["custom_id"]
        pos = 0
        while pos < len(text):
            # find next opening brace
            idx = text.find("{", pos)
            if idx == -1:
                break
            try:
                obj, end = decoder.raw_decode(text, idx)
                has_alignments = "alignments" in obj
                has_new_alignments = "new_alignments" in obj
                if "english_sentence" in obj and (has_alignments or has_new_alignments):
                    if has_new_alignments:
                        obj["_custom_id"] = custom_id
                    jsonl_lines.append(json.dumps(obj, ensure_ascii=False))
                pos = end
            except json.JSONDecodeError:
                pos = idx + 1

    return jsonl_lines


def main():
    parser = argparse.ArgumentParser(
        description="Submit word-spotting annotation requests to the Claude API.",
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "batch"],
        default="sync",
        help="API mode: sync (sequential Messages API) or batch (Batch API) (default: sync)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to batch request JSONL from prepare_batches.py",
    )
    parser.add_argument(
        "--output",
        default="data/gothic_word_spotting/raw_llm/annotation_results.jsonl",
        help=(
            "Output path for raw results JSONL "
            "(default: data/gothic_word_spotting/raw_llm/annotation_results.jsonl)"
        ),
    )
    parser.add_argument(
        "--annotations-output",
        default=None,
        help=(
            "Output path for extracted annotations JSONL, ready for verify_word_spotting.py. "
            "If not set, defaults to the --output path with '_annotations' suffix."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (default: reads ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Max concurrent requests for sync mode (default: 1)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status checks for batch mode (default: 60)",
    )
    parser.add_argument(
        "--range",
        dest="range_spec",
        default=None,
        help="Process only batches START:END, e.g. 0:1 for testing (default: all)",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: No API key. Set ANTHROPIC_API_KEY or use --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load requests
    requests = load_batch_requests(args.input, args.range_spec)
    print(f"Loaded {len(requests)} batch requests", file=sys.stderr)

    if not requests:
        print("No requests to process.", file=sys.stderr)
        sys.exit(0)

    # Run
    print(f"Running in {args.mode} mode...", file=sys.stderr)
    if args.mode == "sync":
        results = run_sync(requests, api_key, args.max_concurrent)
    else:
        results = run_batch(requests, api_key, args.poll_interval)

    # Write raw results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(results)} raw results to {output_path}", file=sys.stderr)

    # Extract and write annotations
    annotations = extract_annotations(results)

    if args.annotations_output:
        annotations_path = Path(args.annotations_output)
    else:
        annotations_path = output_path.with_name(
            output_path.stem + "_annotations" + ".jsonl"
        )
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    with open(annotations_path, "w", encoding="utf-8") as f:
        for line in annotations:
            f.write(line + "\n")

    # Summary
    total_input = sum(r["usage"]["input_tokens"] for r in results)
    total_output = sum(r["usage"]["output_tokens"] for r in results)
    print(f"Wrote {len(annotations)} annotation lines to {annotations_path}", file=sys.stderr)
    print(
        f"Total tokens: {total_input} input, {total_output} output",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
