"""
Measure tokenization efficiency for different tokenizers.

Computes compression metrics to compare how efficiently different tokenizers
encode the same text. Useful for comparing tokenizers trained with different
parameters (e.g., different lambda values in hybrid seed vocabulary).

Metrics:
- Characters per token (compression): Higher = more efficient
- Tokens per sample: Lower = more efficient
- Fertility (tokens per word): Lower = more efficient

Usage:
    # Single tokenizer, single file
    python tools/tokenizer_efficiency.py \
        --tokenizer tokenizers/got/xglm564m_focus-v16k-s200k_seeded-5.0x-lambda0.5 \
        --text-file data/gothic_test.txt

    # Compare multiple tokenizers on same data
    python tools/tokenizer_efficiency.py \
        --tokenizer tokenizers/got/xglm564m_focus-v16k-s200k_unseeded \
        --tokenizer tokenizers/got/xglm564m_focus-v16k-s200k_seeded-5.0x-lambda0.3 \
        --tokenizer tokenizers/got/xglm564m_focus-v16k-s200k_seeded-5.0x-lambda0.5 \
        --tokenizer tokenizers/got/xglm564m_focus-v16k-s200k_seeded-5.0x-lambda0.7 \
        --text-file data/gothic_test.txt

    # Multiple text files
    python tools/tokenizer_efficiency.py \
        --tokenizer tokenizers/got/xglm564m_focus-v16k-s200k_seeded-5.0x-lambda0.5 \
        --text-file data/gothic_test.txt \
        --text-file data/old_english_test.txt \
        --text-file data/old_norse_test.txt

    # Use JSONL format (reads "text" field from each line)
    python tools/tokenizer_efficiency.py \
        --tokenizer tokenizers/got/xglm564m_focus-v16k-s200k_seeded-5.0x-lambda0.5 \
        --text-file training_subset.jsonl \
        --format jsonl

    # Sample large files for faster computation
    python tools/tokenizer_efficiency.py \
        --tokenizer tokenizers/got/xglm564m_focus-v16k-s200k_seeded-5.0x-lambda0.5 \
        --text-file data/large_corpus.txt \
        --max-samples 10000

    # CSV output for plotting with base overlap computation
    python tools/tokenizer_efficiency.py \
        --tokenizer tokenizers/got/lambda0.0 \
        --tokenizer tokenizers/got/lambda0.3 \
        --tokenizer tokenizers/got/lambda0.5 \
        --tokenizer tokenizers/got/lambda0.7 \
        --tokenizer tokenizers/got/lambda1.0 \
        --text-file data/test.txt \
        --base-tokenizer facebook/xglm-564M \
        --output results.csv
"""

import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


def load_text_data(
    file_path: str,
    format: str = "plaintext",
    max_samples: int = None
) -> list[str]:
    """
    Load text data from file.

    Args:
        file_path: Path to text file
        format: 'plaintext' (one line per sample) or 'jsonl' (reads "text" field)
        max_samples: Maximum samples to load (randomly sampled if file is larger)

    Returns:
        List of text samples
    """
    samples = []

    with open(file_path, encoding='utf-8') as f:
        if format == 'plaintext':
            samples = [line.strip() for line in f if line.strip()]
        elif format == 'jsonl':
            for line in f:
                obj = json.loads(line)
                if 'text' in obj and obj['text'].strip():
                    samples.append(obj['text'].strip())
        else:
            raise ValueError(f"Unknown format: {format}")

    # Sample if requested
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)

    return samples


def compute_efficiency_metrics(
    tokenizer,
    samples: list[str]
) -> dict[str, float]:
    """
    Compute tokenization efficiency metrics.

    Args:
        tokenizer: HuggingFace tokenizer
        samples: List of text samples

    Returns:
        Dictionary with metrics:
        - chars_per_token: Average characters per token (compression)
        - tokens_per_sample: Average tokens per sample (sequence length)
        - fertility: Average tokens per word
        - total_chars: Total characters processed
        - total_tokens: Total tokens generated
        - total_words: Total words (whitespace-separated)
        - num_samples: Number of samples processed
    """
    total_chars = 0
    total_tokens = 0
    total_words = 0

    for sample in samples:
        # Tokenize
        tokens = tokenizer.encode(sample, add_special_tokens=False)

        # Count
        total_chars += len(sample)
        total_tokens += len(tokens)
        total_words += len(sample.split())

    # Compute metrics
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    tokens_per_sample = total_tokens / len(samples) if samples else 0
    fertility = total_tokens / total_words if total_words > 0 else 0

    return {
        'chars_per_token': chars_per_token,
        'tokens_per_sample': tokens_per_sample,
        'fertility': fertility,
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'total_words': total_words,
        'num_samples': len(samples)
    }


def compute_vocab_overlap(tokenizer, base_tokenizer) -> float:
    """
    Compute vocabulary overlap with base tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer to compare
        base_tokenizer: Base HuggingFace tokenizer for comparison

    Returns:
        Percentage of tokenizer vocab that overlaps with base (0-100)
    """
    tok_vocab = set(tokenizer.get_vocab().keys())
    base_vocab = set(base_tokenizer.get_vocab().keys())

    overlap = tok_vocab & base_vocab
    overlap_pct = (len(overlap) / len(tok_vocab)) * 100 if tok_vocab else 0

    return overlap_pct


def format_tokenizer_name(tokenizer_path: str) -> str:
    """Extract a short name from tokenizer path for display."""
    path = Path(tokenizer_path)

    # If it looks like a HuggingFace model ID, use that
    if '/' in tokenizer_path and not Path(tokenizer_path).exists():
        return tokenizer_path

    # Otherwise use the directory name
    return path.name


def print_results_table(results: list[tuple[str, str, dict[str, float]]]):
    """
    Print results as formatted table.

    Args:
        results: List of (tokenizer_name, file_name, metrics) tuples
    """
    # Group by file
    files = {}
    for tok_name, file_name, metrics in results:
        if file_name not in files:
            files[file_name] = []
        files[file_name].append((tok_name, metrics))

    # Print results for each file
    for file_name, file_results in files.items():
        print(f"\n{'='*80}")
        print(f"File: {file_name}")
        print(f"{'='*80}")

        # Check if overlap metric is available
        has_overlap = 'base_overlap_pct' in file_results[0][1]

        if has_overlap:
            print(f"{'Tokenizer':<50} {'Chars/Tok':>10} {'Toks/Samp':>11} {'Fertility':>10} {'Overlap %':>10}")
            print(f"{'-'*91}")
            for tok_name, metrics in file_results:
                print(f"{tok_name:<50} "
                      f"{metrics['chars_per_token']:>10.2f} "
                      f"{metrics['tokens_per_sample']:>11.2f} "
                      f"{metrics['fertility']:>10.3f} "
                      f"{metrics['base_overlap_pct']:>10.2f}")
        else:
            print(f"{'Tokenizer':<50} {'Chars/Tok':>10} {'Toks/Samp':>11} {'Fertility':>10}")
            print(f"{'-'*80}")
            for tok_name, metrics in file_results:
                print(f"{tok_name:<50} "
                      f"{metrics['chars_per_token']:>10.2f} "
                      f"{metrics['tokens_per_sample']:>11.2f} "
                      f"{metrics['fertility']:>10.3f}")

        print(f"\n{'Tokenizer':<50} {'Samples':>10} {'Chars':>12} {'Tokens':>12} {'Words':>12}")
        print(f"{'-'*80}")
        for tok_name, metrics in file_results:
            print(f"{tok_name:<50} "
                  f"{metrics['num_samples']:>10,} "
                  f"{metrics['total_chars']:>12,} "
                  f"{metrics['total_tokens']:>12,} "
                  f"{metrics['total_words']:>12,}")


def write_results_csv(results: list[tuple[str, str, dict[str, float]]], output_path: str):
    """
    Write results to CSV file.

    Args:
        results: List of (tokenizer_name, file_name, metrics) tuples
        output_path: Path to output CSV file
    """
    import csv

    # Check if overlap metric is available
    has_overlap = 'base_overlap_pct' in results[0][2] if results else False

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = [
            'tokenizer', 'file', 'chars_per_token', 'tokens_per_sample',
            'fertility', 'num_samples', 'total_chars', 'total_tokens', 'total_words'
        ]
        if has_overlap:
            header.append('base_overlap_pct')
        writer.writerow(header)

        # Data rows
        for tok_name, file_name, metrics in results:
            row = [
                tok_name,
                file_name,
                f"{metrics['chars_per_token']:.3f}",
                f"{metrics['tokens_per_sample']:.3f}",
                f"{metrics['fertility']:.3f}",
                metrics['num_samples'],
                metrics['total_chars'],
                metrics['total_tokens'],
                metrics['total_words']
            ]
            if has_overlap:
                row.append(f"{metrics['base_overlap_pct']:.2f}")
            writer.writerow(row)

    print(f"\nResults written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure tokenization efficiency metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--tokenizer',
        action='append',
        required=True,
        help='Tokenizer path or HuggingFace model ID (can be specified multiple times)'
    )
    parser.add_argument(
        '--text-file',
        action='append',
        required=True,
        help='Text file path (can be specified multiple times)'
    )
    parser.add_argument(
        '--format',
        choices=['plaintext', 'jsonl'],
        default='plaintext',
        help='Text file format (default: plaintext)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum samples to load per file (randomly sampled)'
    )
    parser.add_argument(
        '--output',
        help='Output CSV file path (optional)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--base-tokenizer',
        help='Base tokenizer for overlap comparison (e.g., facebook/xglm-564M)'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load base tokenizer if specified
    base_tokenizer = None
    if args.base_tokenizer:
        print(f"Loading base tokenizer for overlap comparison: {args.base_tokenizer}")
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
            print(f"Base tokenizer vocab size: {len(base_tokenizer.get_vocab()):,}")
        except Exception as e:
            print(f"ERROR loading base tokenizer: {e}")
            print("Continuing without overlap computation...")
            base_tokenizer = None
        print()

    # Collect all results
    all_results = []

    # Process each tokenizer x file combination
    for tokenizer_path in args.tokenizer:
        tok_name = format_tokenizer_name(tokenizer_path)
        print(f"\nLoading tokenizer: {tok_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"ERROR loading tokenizer {tokenizer_path}: {e}")
            continue

        for text_file in args.text_file:
            file_name = Path(text_file).name
            print(f"  Processing: {file_name}...")

            try:
                # Load text data
                samples = load_text_data(text_file, args.format, args.max_samples)

                if not samples:
                    print(f"    WARNING: No samples loaded from {text_file}")
                    continue

                # Compute metrics
                metrics = compute_efficiency_metrics(tokenizer, samples)

                # Compute vocabulary overlap if base tokenizer provided
                if base_tokenizer:
                    metrics['base_overlap_pct'] = compute_vocab_overlap(tokenizer, base_tokenizer)

                all_results.append((tok_name, file_name, metrics))

                # Print metrics
                print(f"    Processed {metrics['num_samples']:,} samples")
                metrics_str = (f"    Chars/token: {metrics['chars_per_token']:.2f}, "
                              f"Tokens/sample: {metrics['tokens_per_sample']:.2f}, "
                              f"Fertility: {metrics['fertility']:.3f}")
                if base_tokenizer:
                    metrics_str += f", Overlap: {metrics['base_overlap_pct']:.2f}%"
                print(metrics_str)

            except Exception as e:
                print(f"    ERROR processing {text_file}: {e}")
                continue

    # Print results table
    if all_results:
        print_results_table(all_results)

        # Write CSV if requested
        if args.output:
            write_results_csv(all_results, args.output)
    else:
        print("\nNo results to display")
        sys.exit(1)


if __name__ == "__main__":
    main()
