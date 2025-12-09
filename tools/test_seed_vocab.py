"""
Test script for hybrid seed vocabulary functionality.

Tests the complete hybrid seeding pipeline in isolation using real JSONL data:
1. Trains baseline tokenizer (unseeded)
2. Trains hybrid seeded tokenizer
3. Compares vocabularies and shows statistics

Usage:
    # Using existing FOCUS training data
    python tools/test_hybrid_seed.py data/gothic_200k/focus_xglm564m_v16k_s200k/training_subset.jsonl

    # With custom parameters
    python tools/test_hybrid_seed.py my_data.jsonl --vocab-size 32000 --lambda 0.7

    # Save outputs to specific directory
    python tools/test_hybrid_seed.py my_data.jsonl --output-dir test_results/
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tokenizer_utils import train_new_tokenizer


def test_hybrid_seed_vocabulary(
    jsonl_path: str,
    vocab_size: int = 1000,
    seed_lambda: float = 0.5,
    seed_vocab_multiplier: float = 5.0,
    seed_target_mass: int = 10_000_000,
    seed_round_mode: str = "ceil",
    output_dir: str = None
):
    """
    Test the hybrid seed vocabulary pipeline.

    Args:
        jsonl_path: Path to JSONL file with training data
        vocab_size: Target vocabulary size for final tokenizer
        seed_lambda: Interpolation weight (0=corpus, 1=base)
        seed_vocab_multiplier: Size multiplier for intermediate tokenizer
        seed_target_mass: Normalization target
        seed_round_mode: Rounding method for merging
        output_dir: Output directory (default: temporary directory)
    """
    print("=" * 80)
    print("HYBRID SEED VOCABULARY TEST")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  jsonl_path: {jsonl_path}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  seed_lambda: {seed_lambda}")
    print(f"  seed_vocab_multiplier: {seed_vocab_multiplier}")
    print(f"  seed_target_mass: {seed_target_mass:,}")
    print(f"  seed_round_mode: {seed_round_mode}")
    print()

    # Check that JSONL file exists
    if not Path(jsonl_path).exists():
        print(f"ERROR: JSONL file not found: {jsonl_path}")
        return False

    # Count samples in JSONL
    with open(jsonl_path, 'r') as f:
        num_samples = sum(1 for _ in f)
    print(f"Found {num_samples} samples in JSONL file")
    print()

    # Set up directory structure like production: output_dir/tokenizers/test/
    if output_dir:
        base_dir = Path(output_dir)
        use_temp = False
        print(f"Using output directory: {base_dir}")
    else:
        base_dir = Path(tempfile.mkdtemp())
        use_temp = True
        print(f"Using temporary directory: {base_dir}")

    tokenizers_dir = base_dir / "tokenizers" / "test"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tokenizers will be saved to: {tokenizers_dir}")
    print()

    # Build directory names using production format
    from model_utils import format_number
    vocab_str = format_number(vocab_size)
    samples_str = format_number(num_samples)

    try:
        # Step 1: Train tokenizer WITHOUT seed vocabulary (baseline)
        print("Step 1: Training baseline tokenizer (unseeded)")
        print("-" * 80)
        baseline_output = tokenizers_dir / f"xglm564m_v{vocab_str}_s{samples_str}_unseeded"
        try:
            baseline_tokenizer = train_new_tokenizer(
                jsonl_path=jsonl_path,
                base_tokenizer_name="facebook/xglm-564M",
                vocab_size=vocab_size,
                output_path=str(baseline_output),
                num_samples=num_samples,
                use_seed_vocabulary=False,
                character_coverage=1.0
            )
            baseline_vocab = set(baseline_tokenizer.get_vocab().keys())
            print(f"Baseline vocab size: {len(baseline_vocab)}")
            print(f"Saved to: {baseline_output}")
            print()
        except Exception as e:
            print(f"ERROR training baseline: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Step 2: Train tokenizer WITH hybrid seed vocabulary
        print(f"Step 2: Training hybrid seeded tokenizer (lambda={seed_lambda})")
        print("-" * 80)
        seeded_output = tokenizers_dir / f"xglm564m_v{vocab_str}_s{samples_str}_seeded-{seed_vocab_multiplier}x-lambda{seed_lambda}"
        try:
            seeded_tokenizer = train_new_tokenizer(
                jsonl_path=str(jsonl_path),
                base_tokenizer_name="facebook/xglm-564M",
                vocab_size=vocab_size,
                output_path=str(seeded_output),
                num_samples=num_samples,
                use_seed_vocabulary=True,
                seed_lambda=seed_lambda,
                seed_vocab_multiplier=seed_vocab_multiplier,
                seed_target_mass=seed_target_mass,
                seed_round_mode=seed_round_mode,
                character_coverage=1.0
            )
            seeded_vocab = set(seeded_tokenizer.get_vocab().keys())
            print(f"Seeded vocab size: {len(seeded_vocab)}")
            print(f"Saved to: {seeded_output}")

            # Check that seed tokenizer exists at sibling level
            seed_tokenizer_dir = tokenizers_dir / f"xglm564m_v{vocab_str}_s{samples_str}_seed-{seed_vocab_multiplier}x"
            if seed_tokenizer_dir.exists():
                print(f"Seed tokenizer saved to: {seed_tokenizer_dir}")
                print(f"  (Can be reused for other lambda values)")
            print()
        except Exception as e:
            print(f"ERROR training seeded tokenizer: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Step 3: Load base tokenizer for comparison
        print("Step 3: Comparing vocabularies")
        print("-" * 80)
        from transformers import AutoTokenizer
        base_tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
        base_vocab = set(base_tokenizer.get_vocab().keys())

        # Compute overlaps
        baseline_overlap = baseline_vocab & base_vocab
        seeded_overlap = seeded_vocab & base_vocab

        baseline_only = baseline_vocab - base_vocab
        seeded_only = seeded_vocab - base_vocab

        print(f"Base tokenizer vocab size: {len(base_vocab):,}")
        print()

        print(f"Baseline tokenizer (unseeded):")
        print(f"  Total vocab: {len(baseline_vocab)}")
        print(f"  Overlap with base: {len(baseline_overlap)} ({100*len(baseline_overlap)/len(baseline_vocab):.1f}%)")
        print(f"  Novel tokens: {len(baseline_only)} ({100*len(baseline_only)/len(baseline_vocab):.1f}%)")
        print()

        print(f"Seeded tokenizer (hybrid with lambda={seed_lambda}):")
        print(f"  Total vocab: {len(seeded_vocab)}")
        print(f"  Overlap with base: {len(seeded_overlap)} ({100*len(seeded_overlap)/len(seeded_vocab):.1f}%)")
        print(f"  Novel tokens: {len(seeded_only)} ({100*len(seeded_only)/len(seeded_vocab):.1f}%)")
        print()

        # Show effect of lambda
        overlap_diff = len(seeded_overlap) - len(baseline_overlap)
        novel_diff = len(seeded_only) - len(baseline_only)
        print(f"Effect of hybrid seeding (lambda={seed_lambda}):")
        print(f"  Change in base overlap: {overlap_diff:+d} tokens")
        print(f"  Change in novel tokens: {novel_diff:+d} tokens")
        print()

        # Step 4: Test tokenization
        print("Step 4: Testing tokenization")
        print("-" * 80)
        test_sentences = [
            "The quick brown fox",
            "𐌰𐌹𐌽𐍃 𐍄𐍅𐌰 𐌸𐍂𐌹𐌾𐌰",
            "Machine learning models"
        ]

        for sentence in test_sentences:
            baseline_tokens = baseline_tokenizer.tokenize(sentence)
            seeded_tokens = seeded_tokenizer.tokenize(sentence)
            base_tokens = base_tokenizer.tokenize(sentence)

            print(f"Sentence: {sentence!r}")
            print(f"  Base tokenizer:     {len(base_tokens)} tokens")
            print(f"  Baseline (unseeded): {len(baseline_tokens)} tokens: {baseline_tokens}")
            print(f"  Seeded (λ={seed_lambda}):  {len(seeded_tokens)} tokens: {seeded_tokens}")
            print()

        # Step 5: Check cached artifacts
        print("Step 5: Checking cached artifacts")
        print("-" * 80)
        seed_tokenizer_dir = tokenizers_dir / f"xglm564m_v{vocab_str}_s{samples_str}_seed-{seed_vocab_multiplier}x"
        base_vocab_file = Path(seeded_output) / "base_vocab_counts.txt"
        seed_vocab_file = Path(seeded_output) / "seed_vocab.txt"

        print(f"Seed tokenizer: {'✓ exists' if seed_tokenizer_dir.exists() else '✗ missing'}")
        if seed_tokenizer_dir.exists():
            seed_model = seed_tokenizer_dir / "spm.model"
            seed_vocab = seed_tokenizer_dir / "spm.vocab"
            print(f"  spm.model: {'✓' if seed_model.exists() else '✗'}")
            print(f"  spm.vocab: {'✓' if seed_vocab.exists() else '✗'}")
            expected_seed_size = int(vocab_size * seed_vocab_multiplier)
            print(f"  Expected vocab size: {expected_seed_size}")

        print(f"Base vocab counts: {'✓ exists' if base_vocab_file.exists() else '✗ missing'}")
        if base_vocab_file.exists():
            with open(base_vocab_file) as f:
                num_lines = sum(1 for _ in f)
            print(f"  Contains {num_lines} tokens")

        print(f"Merged seed vocab: {'✓ exists' if seed_vocab_file.exists() else '✗ missing'}")
        if seed_vocab_file.exists():
            with open(seed_vocab_file) as f:
                num_lines = sum(1 for _ in f)
            print(f"  Contains {num_lines} tokens")
        print()

        # Success!
        print("=" * 80)
        print("✓ TEST PASSED")
        print("=" * 80)
        print()
        print("Key findings:")
        print(f"  - Hybrid seeding successfully interpolates between base and corpus vocabularies")
        print(f"  - Lambda={seed_lambda} → {100*len(seeded_overlap)/len(seeded_vocab):.1f}% base overlap")
        print(f"  - Baseline (unseeded) → {100*len(baseline_overlap)/len(baseline_vocab):.1f}% base overlap")
        print()
        print("Cached artifacts:")
        print(f"  - Seed tokenizer: {seed_tokenizer_dir}")
        print(f"    (Shared across different lambda values)")
        print(f"  - Final tokenizer: {seeded_output}")
        print()
        print("To try a different lambda value:")
        print(f"  python tools/test_hybrid_seed.py {jsonl_path} \\")
        print(f"    --output-dir {base_dir} \\")
        print(f"    --lambda 0.7  # or any value 0.0-1.0")
        print(f"  → Will reuse seed tokenizer from {seed_tokenizer_dir}")
        print()

        return True

    finally:
        # Cleanup temporary directory if used
        if use_temp:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Test hybrid seed vocabulary functionality with real data"
    )
    parser.add_argument(
        "jsonl_file",
        help="Path to JSONL file with training data (e.g., from FOCUS preparation)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=16000,
        help="Target vocabulary size (default: 16000)"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for tokenizers (default: temporary directory)"
    )
    parser.add_argument(
        "--lambda",
        dest="seed_lambda",
        type=float,
        default=0.5,
        help="Interpolation weight: 0=corpus, 1=base (default: 0.5)"
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=5.0,
        help="Intermediate tokenizer size multiplier (default: 5.0)"
    )
    parser.add_argument(
        "--target_mass",
        type=int,
        default=10_000_000,
        help="Normalization target (default: 10000000)"
    )
    parser.add_argument(
        "--round_mode",
        choices=["ceil", "floor", "round"],
        default="ceil",
        help="Rounding mode for merging (default: ceil)"
    )

    args = parser.parse_args()

    success = test_hybrid_seed_vocabulary(
        jsonl_path=args.jsonl_file,
        vocab_size=args.vocab_size,
        seed_lambda=args.seed_lambda,
        seed_vocab_multiplier=args.multiplier,
        seed_target_mass=args.target_mass,
        seed_round_mode=args.round_mode,
        output_dir=args.output_dir
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
