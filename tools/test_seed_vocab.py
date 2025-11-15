#!/usr/bin/env python3
"""
Test script for seed vocabulary extraction and tokenizer training.

This script demonstrates the seed vocabulary feature that biases SentencePiece
training toward tokens that overlap with the base tokenizer.
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer_utils import extract_base_vocabulary_frequencies, train_new_tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Test seed vocabulary extraction for SentencePiece training"
    )
    parser.add_argument(
        '--filter-single-chars',
        action='store_true',
        help="Filter out single-character tokens from seed vocabulary"
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=1,
        help="Minimum frequency threshold for including tokens (default: 1)"
    )
    args = parser.parse_args()

    # Configuration
    base_tokenizer = "facebook/xglm-564M"

    # Use the parallel Gothic-English data for testing
    # This has both Gothic (poorly tokenized by XGLM) and English (well tokenized)
    input_text = "data/parallel/gothic_english.txt"

    if not os.path.exists(input_text):
        print(f"Error: Input file not found: {input_text}")
        print("Please run the alignment script first:")
        print("  python tools/align_gothic_english.py")
        return

    # Output paths
    suffix = "_no_single_chars" if args.filter_single_chars else ""
    seed_vocab_file = f"test_outputs/seed_vocab{suffix}.txt"
    output_tokenizer_dir = "test_outputs/tokenizer_with_seed"

    os.makedirs("test_outputs", exist_ok=True)

    print("=" * 60)
    print("Testing Seed Vocabulary Extraction")
    print("=" * 60)
    print(f"Filter single chars: {args.filter_single_chars}")
    print(f"Min frequency: {args.min_frequency}")

    # Step 1: Extract base vocabulary frequencies
    print("\n1. Extracting base vocabulary frequencies...")
    seed_file = extract_base_vocabulary_frequencies(
        text_file_path=input_text,
        base_tokenizer_name=base_tokenizer,
        output_seed_file=seed_vocab_file,
        filter_special_tokens=True,
        filter_single_chars=args.filter_single_chars,
        min_frequency=args.min_frequency
    )

    # Show sample of seed vocabulary
    print("\n2. Sample of seed vocabulary (top 20 tokens):")
    with open(seed_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            token, freq = line.strip().split('\t')
            print(f"   {token:<20} {freq:>6}")

    print("\n" + "=" * 60)
    print("Note: This is just the seed vocabulary extraction test.")
    print("To test full tokenizer training with the seed, you would call:")
    print(f"  train_new_tokenizer(..., seed_sentencepieces_file='{seed_file}')")
    print("=" * 60)

if __name__ == "__main__":
    main()
