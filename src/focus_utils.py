"""
Utilities for FOCUS-based vocabulary and embedding reinitialization.

This module provides functions to:
1. Prepare training data in JSONL format for FOCUS
2. Train new language-specific tokenizers
3. Apply FOCUS to initialize embeddings for the new vocabulary
"""

import json
import os
import random
import sys
from typing import Optional

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer


def prepare_focus_training_data(
    num_samples: int,
    output_jsonl_path: str,
    seed: int = 1,
    train_dataset_cache: str = None,
    dataset_config = None
) -> str:
    """
    Extract a random subset of untokenized data and convert to JSONL format.

    Args:
        num_samples: Number of samples to extract
        output_jsonl_path: Path where JSONL file will be saved
        seed: Random seed for reproducible sampling
        train_dataset_cache: Path to training dataset cache directory (for reusing training data)
        dataset_config: Optional separate dataset configuration for FOCUS

    Returns:
        Path to the created JSONL file
    """
    if os.path.exists(output_jsonl_path):
        print(f"JSONL data already exists at {output_jsonl_path}, skipping generation", file=sys.stderr)
        return output_jsonl_path

    print(f"Preparing FOCUS training data: {num_samples} samples", file=sys.stderr)

    # If dataset_config provided, load that dataset; otherwise use training dataset
    if dataset_config is not None:
        from dataset_utils import load_untokenized_dataset
        focus_cache = os.path.dirname(output_jsonl_path)
        untokenized_path = load_untokenized_dataset(
            dataset_config=dataset_config,
            cache_dir=focus_cache
        )
        dataset = load_from_disk(untokenized_path)
    else:
        if train_dataset_cache is None:
            raise ValueError("Either train_dataset_cache or dataset_config must be provided")
        untokenized_path = os.path.join(train_dataset_cache, "untokenized")
        if os.path.exists(untokenized_path):
            dataset = load_from_disk(untokenized_path)
        else:
            raise FileNotFoundError(
                f"Untokenized dataset not found at {untokenized_path}. "
                "Please ensure the dataset is loaded first."
            )

    train_data = dataset['train']
    total_samples = len(train_data)

    if num_samples > total_samples:
        print(
            f"Warning: Requested {num_samples} samples but dataset only has {total_samples}. "
            f"Using all available samples.",
            file=sys.stderr
        )
        num_samples = total_samples

    random.seed(seed)
    indices = random.sample(range(total_samples), num_samples)
    indices.sort()

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx in indices:
            text = train_data[idx]['text']
            json.dump({'text': text}, f, ensure_ascii=False)
            f.write('\n')

    print(f"JSONL data saved to {output_jsonl_path}", file=sys.stderr)
    return output_jsonl_path


def train_new_tokenizer(
    jsonl_path: str,
    base_tokenizer_name: str,
    vocab_size: int,
    output_path: str
) -> AutoTokenizer:
    """
    Train a new tokenizer on JSONL data using HuggingFace tokenizers library.

    The tokenizer will use the same algorithm (BPE, Unigram, etc.) as the base tokenizer.

    Args:
        jsonl_path: Path to JSONL file with training data
        base_tokenizer_name: Name of base model tokenizer (for special tokens and algorithm)
        vocab_size: Target vocabulary size
        output_path: Directory where trained tokenizer will be saved

    Returns:
        Trained tokenizer
    """
    if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "tokenizer.json")):
        print(f"Tokenizer already exists at {output_path}, loading it", file=sys.stderr)
        return AutoTokenizer.from_pretrained(output_path)

    print(f"Training new tokenizer with vocab size {vocab_size}", file=sys.stderr)

    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

    def batch_iterator(batch_size=1000):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                data = json.loads(line)
                batch.append(data['text'])
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    new_tokenizer = base_tokenizer.train_new_from_iterator(
        batch_iterator(),
        vocab_size=vocab_size,
    )

    os.makedirs(output_path, exist_ok=True)
    new_tokenizer.save_pretrained(output_path)

    print(f"Tokenizer saved to {output_path}", file=sys.stderr)
    return new_tokenizer


def apply_focus_initialization(
    source_model,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    training_data_path: str
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply FOCUS to generate new input embeddings and optionally output embeddings.

    Args:
        source_model: Source pretrained model
        source_tokenizer: Tokenizer for the source model
        target_tokenizer: Target language-specific tokenizer
        training_data_path: Path to JSONL training data for FOCUS

    Returns:
        Tuple of (input_embeddings, output_embeddings)
        output_embeddings will be None if model ties word embeddings
    """
    print("Applying FOCUS to initialize embeddings", file=sys.stderr)

    try:
        from deepfocus import FOCUS
    except ImportError:
        raise ImportError(
            "deepfocus package not found. Please install it with: pip install deepfocus"
        )

    source_embeddings = source_model.get_input_embeddings().weight

    new_input_embeddings = FOCUS(
        source_embeddings=source_embeddings,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        target_training_data_path=training_data_path
    )

    new_output_embeddings = None
    if hasattr(source_model.config, 'tie_word_embeddings') and not source_model.config.tie_word_embeddings:
        print("Model uses separate output embeddings, applying FOCUS to output embeddings", file=sys.stderr)
        source_output_embeddings = source_model.get_output_embeddings().weight
        new_output_embeddings = FOCUS(
            source_embeddings=source_output_embeddings,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            target_training_data_path=training_data_path
        )

    print(f"FOCUS initialization complete. New vocab size: {len(target_tokenizer)}", file=sys.stderr)
    return new_input_embeddings, new_output_embeddings
