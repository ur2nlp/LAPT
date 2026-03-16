"""
Diagnose English tokenization mismatch between a base tokenizer and a FOCUS tokenizer.

The core question: when the FOCUS tokenizer processes English text, how much of it is
tokenized identically to the base XGLM tokenizer? Non-identical tokenization means the
pre-trained XGLM embeddings for those token spans cannot be reused — the model must
re-learn English from scratch for those pieces.

Two levels of analysis:
  - Vocabulary-level: what % of XGLM token instances have their exact token string in the
    FOCUS vocabulary? (Upper bound on reuse; FOCUS might still segment differently)
  - Segmentation-level: what % of XGLM token boundaries are also FOCUS token boundaries?
    (Actual segmentation agreement, requires tokenizing with both)

Usage:
    python tools/compare_english_tokenization.py \\
        --focus-tokenizer /path/to/focus/tokenizer \\
        --text-file data/english_sample.txt \\
        --max-samples 5000

    python tools/compare_english_tokenization.py \\
        --focus-tokenizer /path/to/focus/tokenizer \\
        --text-file data/english_sample.txt \\
        --base-tokenizer facebook/xglm-1.7B \\
        --max-samples 5000
"""

import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


def load_samples(text_file: str, max_samples: int | None, seed: int) -> list[str]:
    """Load text samples from a plaintext file (one sample per line)."""
    with open(text_file, encoding="utf-8") as f:
        samples = [line.strip() for line in f if line.strip()]
    if max_samples and len(samples) > max_samples:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_samples)
    return samples


def get_token_boundaries(tokenizer, text: str) -> list[tuple[int, int]] | None:
    """
    Return a list of (start, end) character offsets for each token in text.

    Uses offset_mapping from the fast tokenizer if available. Returns None if
    offset mapping is not supported.
    """
    try:
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]
        # Filter out zero-length spans (e.g. added special tokens)
        return [(s, e) for s, e in offsets if s != e]
    except Exception:
        return None


def boundary_set(spans: list[tuple[int, int]]) -> set[int]:
    """Convert a list of (start, end) token spans to a set of boundary positions."""
    boundaries = set()
    for start, end in spans:
        boundaries.add(start)
        boundaries.add(end)
    return boundaries


def analyze_samples(
    base_tokenizer,
    focus_tokenizer,
    samples: list[str],
) -> dict:
    """
    Compute tokenization agreement statistics across samples.

    Returns a dict with aggregate counts and rates.
    """
    focus_vocab = set(focus_tokenizer.get_vocab().keys())

    total_base_tokens = 0
    total_focus_tokens = 0
    total_chars = 0

    # Vocabulary-level: base token instances whose string is in FOCUS vocab
    base_instances_in_focus_vocab = 0
    base_chars_in_focus_vocab = 0

    # Segmentation-level: boundary agreement
    total_base_boundaries = 0
    base_boundaries_preserved = 0  # XGLM boundaries that are also FOCUS boundaries
    total_focus_boundaries = 0
    focus_boundaries_matching_base = 0  # FOCUS boundaries that are also XGLM boundaries
    jaccard_sum = 0.0

    exact_matches = 0
    offset_mapping_available = True
    samples_with_offsets = 0

    for text in samples:
        base_spans = get_token_boundaries(base_tokenizer, text)
        focus_spans = get_token_boundaries(focus_tokenizer, text)

        if base_spans is None or focus_spans is None:
            offset_mapping_available = False

        total_chars += len(text)

        # Vocabulary-level analysis: tokenize with base, check each token string
        base_token_strings = base_tokenizer.tokenize(text)
        focus_token_strings = focus_tokenizer.tokenize(text)

        total_base_tokens += len(base_token_strings)
        total_focus_tokens += len(focus_token_strings)

        for token in base_token_strings:
            if token in focus_vocab:
                base_instances_in_focus_vocab += 1
                # Estimate character coverage (strip SPM ▁ prefix for length)
                base_chars_in_focus_vocab += len(token.replace("▁", " ").strip())

        # Exact sequence match
        if base_token_strings == focus_token_strings:
            exact_matches += 1

        # Segmentation-level analysis
        if base_spans is not None and focus_spans is not None:
            samples_with_offsets += 1
            base_bounds = boundary_set(base_spans)
            focus_bounds = boundary_set(focus_spans)

            intersection = base_bounds & focus_bounds
            union = base_bounds | focus_bounds

            total_base_boundaries += len(base_bounds)
            base_boundaries_preserved += len(intersection)
            total_focus_boundaries += len(focus_bounds)
            focus_boundaries_matching_base += len(intersection)

            if union:
                jaccard_sum += len(intersection) / len(union)

    n = len(samples)
    return {
        "n_samples": n,
        "total_chars": total_chars,
        "total_base_tokens": total_base_tokens,
        "total_focus_tokens": total_focus_tokens,
        "relative_fertility": total_focus_tokens / total_base_tokens if total_base_tokens else 0,
        # Vocabulary-level
        "base_instances_in_focus_vocab": base_instances_in_focus_vocab,
        "base_vocab_instance_rate": base_instances_in_focus_vocab / total_base_tokens if total_base_tokens else 0,
        # Segmentation-level (None if offset mapping unavailable)
        "offset_mapping_available": offset_mapping_available,
        "samples_with_offsets": samples_with_offsets,
        "base_boundary_preservation": base_boundaries_preserved / total_base_boundaries if total_base_boundaries else None,
        "focus_boundary_precision": focus_boundaries_matching_base / total_focus_boundaries if total_focus_boundaries else None,
        "mean_boundary_jaccard": jaccard_sum / samples_with_offsets if samples_with_offsets else None,
        "exact_match_rate": exact_matches / n if n else 0,
        "n_exact_matches": exact_matches,
    }


def print_report(
    stats: dict,
    base_name: str,
    focus_name: str,
    base_vocab_size: int,
    focus_vocab_size: int,
    shared_vocab_size: int,
) -> None:
    """Print a formatted summary report."""
    print(f"\n{'='*70}")
    print("English Tokenization Comparison")
    print(f"  Base:  {base_name}")
    print(f"  FOCUS: {focus_name}")
    print(f"{'='*70}")

    print(f"\n--- Vocabulary (type-level) ---")
    print(f"  Base vocab size:    {base_vocab_size:>10,}")
    print(f"  FOCUS vocab size:   {focus_vocab_size:>10,}")
    shared_pct = 100 * shared_vocab_size / focus_vocab_size if focus_vocab_size else 0
    print(f"  Shared strings:     {shared_vocab_size:>10,}  ({shared_pct:.1f}% of FOCUS vocab)")

    print(f"\n--- Instance-level on {stats['n_samples']:,} samples ---")
    print(f"  Total base tokens:  {stats['total_base_tokens']:>10,}")
    print(f"  Total FOCUS tokens: {stats['total_focus_tokens']:>10,}")
    rel = stats['relative_fertility']
    direction = "more" if rel > 1 else "fewer"
    print(f"  Relative fertility: {rel:>10.3f}  (FOCUS has {abs(rel-1)*100:.1f}% {direction} tokens)")

    print(f"\n--- Vocabulary-level reuse (upper bound) ---")
    print(f"  Base token instances whose string is in FOCUS vocab:")
    rate = stats['base_vocab_instance_rate'] * 100
    n = stats['base_instances_in_focus_vocab']
    total = stats['total_base_tokens']
    print(f"    {n:,} / {total:,}  =  {rate:.1f}%")
    print(f"  (If FOCUS vocab includes this token, it *can* reuse the embedding)")

    print(f"\n--- Segmentation-level agreement ---")
    if stats['offset_mapping_available']:
        bp = stats['base_boundary_preservation'] * 100
        fp = stats['focus_boundary_precision'] * 100
        jac = stats['mean_boundary_jaccard']
        print(f"  XGLM boundary preservation:  {bp:.1f}%  (XGLM boundaries also in FOCUS)")
        print(f"  FOCUS boundary precision:    {fp:.1f}%  (FOCUS boundaries also in XGLM)")
        print(f"  Mean boundary Jaccard:       {jac:.3f}")
    else:
        print("  (offset_mapping not available for these tokenizers — skipped)")

    exact_pct = stats['exact_match_rate'] * 100
    print(f"  Exact sequence match rate:   {exact_pct:.1f}%  ({stats['n_exact_matches']:,}/{stats['n_samples']:,} samples)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose English tokenization mismatch between base and FOCUS tokenizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--focus-tokenizer",
        required=True,
        help="Path to the FOCUS tokenizer directory",
    )
    parser.add_argument(
        "--base-tokenizer",
        default="facebook/xglm-1.7B",
        help="Base tokenizer (HuggingFace model ID or path, default: facebook/xglm-1.7B)",
    )
    parser.add_argument(
        "--text-file",
        required=True,
        help="Plaintext file to analyze (one sample per line)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Number of samples to analyze (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    print(f"Loading base tokenizer: {args.base_tokenizer} ...", file=sys.stderr)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)

    print(f"Loading FOCUS tokenizer: {args.focus_tokenizer} ...", file=sys.stderr)
    focus_tokenizer = AutoTokenizer.from_pretrained(args.focus_tokenizer)

    print(f"Loading samples from: {args.text_file} ...", file=sys.stderr)
    samples = load_samples(args.text_file, args.max_samples, args.seed)
    print(f"Loaded {len(samples):,} samples.", file=sys.stderr)

    print("Analyzing...", file=sys.stderr)
    stats = analyze_samples(base_tokenizer, focus_tokenizer, samples)

    base_vocab = set(base_tokenizer.get_vocab().keys())
    focus_vocab = set(focus_tokenizer.get_vocab().keys())
    shared = base_vocab & focus_vocab

    print_report(
        stats,
        base_name=args.base_tokenizer,
        focus_name=Path(args.focus_tokenizer).name,
        base_vocab_size=len(base_vocab),
        focus_vocab_size=len(focus_vocab),
        shared_vocab_size=len(shared),
    )


if __name__ == "__main__":
    main()
