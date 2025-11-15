#!/usr/bin/env python3
"""
Integration test for seed vocabulary in tokenizer training workflow.

Tests that seed vocabulary generation works when integrated into train_new_tokenizer().
"""

import sys
import os
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer_utils import prepare_focus_training_data, train_new_tokenizer

def main():
    print("=" * 60)
    print("Integration Test: Seed Vocabulary in Tokenizer Training")
    print("=" * 60)

    # Setup test directories
    test_dir = "test_outputs/integration"
    os.makedirs(test_dir, exist_ok=True)

    # Step 1: Prepare FOCUS training data (JSONL)
    print("\n1. Preparing FOCUS training data...")
    from omegaconf import OmegaConf

    # Create a simple dataset config for the parallel data
    dataset_config = OmegaConf.create({
        'type': 'plaintext',
        'path': 'data/parallel/gothic_english.txt',
        'cache_dir': f'{test_dir}/dataset_cache'
    })

    jsonl_path = prepare_focus_training_data(
        num_samples=1000,
        output_jsonl_path=f"{test_dir}/training_subset.jsonl",
        seed=1,
        dataset_config=dataset_config
    )

    # Step 2: Train tokenizer WITHOUT seed vocabulary
    print("\n2. Training tokenizer WITHOUT seed vocabulary...")
    tokenizer_no_seed = train_new_tokenizer(
        jsonl_path=jsonl_path,
        base_tokenizer_name="facebook/xglm-564M",
        vocab_size=3500,  # Small corpus can only support ~3964 tokens
        output_path=f"{test_dir}/tokenizer_no_seed",
        use_seed_vocabulary=False
    )
    print(f"   Tokenizer vocab size: {len(tokenizer_no_seed)}")

    # Step 3: Train tokenizer WITH seed vocabulary
    print("\n3. Training tokenizer WITH seed vocabulary...")
    tokenizer_with_seed = train_new_tokenizer(
        jsonl_path=jsonl_path,
        base_tokenizer_name="facebook/xglm-564M",
        vocab_size=3500,
        output_path=f"{test_dir}/tokenizer_with_seed",
        use_seed_vocabulary=True,
        seed_filter_single_chars=True,
        seed_min_frequency=1
    )
    print(f"   Tokenizer vocab size: {len(tokenizer_with_seed)}")

    # Step 4: Check that seed vocab file was created
    seed_file = f"{test_dir}/tokenizer_with_seed/seed_vocab.txt"
    if os.path.exists(seed_file):
        print(f"\n✓ Seed vocabulary file created: {seed_file}")
        with open(seed_file, 'r') as f:
            num_seed_tokens = sum(1 for _ in f)
        print(f"   Contains {num_seed_tokens} seed tokens")
    else:
        print(f"\n✗ ERROR: Seed vocabulary file not found at {seed_file}")

    # Step 5: Compare overlap with base tokenizer
    print("\n4. Comparing base tokenizer overlap...")
    from transformers import AutoTokenizer
    base_tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")

    base_vocab = set(base_tokenizer.get_vocab().keys())
    no_seed_vocab = set(tokenizer_no_seed.get_vocab().keys())
    with_seed_vocab = set(tokenizer_with_seed.get_vocab().keys())

    overlap_no_seed = len(base_vocab & no_seed_vocab)
    overlap_with_seed = len(base_vocab & with_seed_vocab)

    print(f"   Without seed: {overlap_no_seed} tokens overlap with base ({overlap_no_seed/len(tokenizer_no_seed)*100:.1f}%)")
    print(f"   With seed:    {overlap_with_seed} tokens overlap with base ({overlap_with_seed/len(tokenizer_with_seed)*100:.1f}%)")

    if overlap_with_seed > overlap_no_seed:
        improvement = overlap_with_seed - overlap_no_seed
        print(f"\n✓ Seed vocabulary improved overlap by {improvement} tokens!")
    else:
        print(f"\n⚠ Warning: Seed vocabulary did not improve overlap (may need more data)")

    print("\n" + "=" * 60)
    print("Integration test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
