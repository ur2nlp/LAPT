"""
Utilities for loading and processing datasets for language-adaptive pretraining.

This module handles downloading OSCAR corpus data, converting it to line-based format,
tokenizing with provided tokenizers, and caching results.
"""

import os
import random
import sys
from itertools import chain

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer


def docs_to_lines(examples):
    """
    Convert document-based examples to line-based examples.

    OSCAR data comes as documents with newlines. This function splits
    each document into individual lines for more granular training.

    Args:
        examples: Batch of examples with 'text' field containing documents

    Returns:
        Dictionary with 'text' field containing individual lines
    """
    return {
        'text': list(chain(
            *[doc.split('\n') for doc in examples['text']]
        ))
    }


def load_untokenized_dataset(dataset_config, cache_dir: str) -> str:
    """
    Load untokenized dataset based on configuration.

    This dispatcher routes to the appropriate loader based on dataset type.

    Args:
        dataset_config: Dataset configuration object with type and source info
        cache_dir: Base directory for caching dataset artifacts

    Returns:
        Path to the untokenized dataset
    """
    # Default to oscar for backward compatibility if type not specified
    dataset_type = getattr(dataset_config, 'type', 'oscar')

    if dataset_type == 'oscar':
        language_code = dataset_config.language
        return _load_oscar_dataset(cache_dir, language_code)
    elif dataset_type == 'plaintext':
        file_path = dataset_config.path
        return _load_plaintext_dataset(cache_dir, file_path)
    elif dataset_type == 'concat':
        sources = dataset_config.sources
        return _load_concat_dataset(cache_dir, sources)
    elif dataset_type == 'multinomial':
        sources = dataset_config.sources
        alpha = dataset_config.alpha
        total_samples = dataset_config.total_samples
        return _load_multinomial_dataset(cache_dir, sources, alpha, total_samples)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _load_oscar_dataset(cache_dir: str, language_code: str) -> str:
    """
    Load or download OSCAR dataset for a specific language.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        language_code: Two-letter language code for OSCAR corpus

    Returns:
        Path to the untokenized dataset
    """
    untokenized_path = os.path.join(cache_dir, "untokenized")

    if not os.path.exists(untokenized_path):
        print("Downloading and preparing OSCAR dataset", file=sys.stderr)
        dataset = load_dataset(
            "oscar-corpus/OSCAR-2201",
            token=True,
            language=language_code
        )
        dataset = dataset.map(
            docs_to_lines,
            batched=True,
            remove_columns=dataset['train'].column_names # type: ignore
        )
        dataset.save_to_disk(untokenized_path)
        print(f"Untokenized dataset saved to {untokenized_path}", file=sys.stderr)

    return untokenized_path


def _load_plaintext_dataset(cache_dir: str, file_path: str) -> str:
    """
    Load plaintext file(s) and convert to dataset format.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        file_path: Path to plaintext file (one line per training example)

    Returns:
        Path to the untokenized dataset
    """
    untokenized_path = os.path.join(cache_dir, "untokenized")

    if not os.path.exists(untokenized_path):
        print(f"Loading plaintext data from {file_path}", file=sys.stderr)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Plaintext file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(lines)} lines from plaintext file", file=sys.stderr)

        dataset = Dataset.from_dict({'text': lines})
        dataset_dict = DatasetDict({'train': dataset})
        dataset_dict.save_to_disk(untokenized_path)
        print(f"Untokenized dataset saved to {untokenized_path}", file=sys.stderr)

    return untokenized_path


def _load_concat_dataset(cache_dir: str, sources: list) -> str:
    """
    Concatenate multiple dataset sources into a single dataset.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        sources: List of dataset source configurations

    Returns:
        Path to the untokenized concatenated dataset
    """
    untokenized_path = os.path.join(cache_dir, "untokenized")

    if not os.path.exists(untokenized_path):
        print(f"Concatenating {len(sources)} dataset sources", file=sys.stderr)

        datasets_to_concat = []
        for idx, source_config in enumerate(sources):
            source_cache = os.path.join(cache_dir, f"source_{idx}")
            # Wrap in DictConfig for recursive dispatching
            source_dict_config = DictConfig(source_config)

            # Recursively load each source (supports nested concat/multinomial)
            source_path = load_untokenized_dataset(
                dataset_config=source_dict_config,
                cache_dir=source_cache
            )

            source_dataset = load_from_disk(source_path)
            datasets_to_concat.append(source_dataset['train'])
            print(f"  Source {idx}: {len(source_dataset['train'])} examples", file=sys.stderr)

        concatenated = concatenate_datasets(datasets_to_concat)
        dataset_dict = DatasetDict({'train': concatenated})
        dataset_dict.save_to_disk(untokenized_path)
        print(f"Concatenated dataset saved to {untokenized_path} ({len(concatenated)} total examples)", file=sys.stderr)

    return untokenized_path


def _load_multinomial_dataset(
    cache_dir: str, sources: list, alpha: float, total_samples: int
) -> str:
    """
    Sample from multiple dataset sources using temperature-scaled multinomial sampling.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        sources: List of dataset source configurations
        alpha: Temperature parameter for reweighting (< 1 upsamples smaller datasets)
        total_samples: Total number of examples to sample

    Returns:
        Path to the untokenized sampled dataset
    """
    untokenized_path = os.path.join(cache_dir, "untokenized")

    if not os.path.exists(untokenized_path):
        print(f"Multinomial sampling from {len(sources)} sources with alpha={alpha}", file=sys.stderr)

        source_datasets = []
        source_sizes = []

        # Load all sources and record their sizes
        for idx, source_config in enumerate(sources):
            source_cache = os.path.join(cache_dir, f"source_{idx}")
            source_dict_config = DictConfig(source_config)

            source_path = load_untokenized_dataset(
                dataset_config=source_dict_config,
                cache_dir=source_cache
            )

            source_dataset = load_from_disk(source_path)
            train_data = source_dataset['train']
            source_datasets.append(train_data)
            source_sizes.append(len(train_data))
            print(f"  Source {idx}: {len(train_data)} examples", file=sys.stderr)

        # Calculate sampling probabilities: p_i = (size_i)^alpha / Z
        # alpha < 1 upsamples smaller datasets, alpha > 1 amplifies size differences
        weights = [size ** alpha for size in source_sizes]
        total_weight = sum(weights)
        sampling_probs = [w / total_weight for w in weights]

        # Convert probabilities to integer sample counts
        # Distribute remainder samples round-robin to handle rounding errors
        samples_per_source = [int(prob * total_samples) for prob in sampling_probs]
        remaining = total_samples - sum(samples_per_source)
        for i in range(remaining):
            samples_per_source[i % len(sources)] += 1

        print("Sampling distribution:", file=sys.stderr)
        for idx, count in enumerate(samples_per_source):
            percentage = 100 * count / total_samples
            print(f"  Source {idx}: {count} samples ({percentage:.2f}%)", file=sys.stderr)

        # Use HF's .select() to keep data memory-mapped instead of loading into RAM
        selected_datasets = []
        for idx, (dataset, num_samples) in enumerate(zip(source_datasets, samples_per_source)):
            # Sample without replacement if we have enough data, otherwise with replacement
            if num_samples <= len(dataset):
                indices = random.sample(range(len(dataset)), num_samples)
            else:
                indices = random.choices(range(len(dataset)), k=num_samples)

            # .select() keeps data memory-mapped, handles duplicate indices for sampling with replacement
            selected = dataset.select(indices)
            selected_datasets.append(selected)

        # Concatenate and shuffle efficiently without loading into RAM
        concatenated = concatenate_datasets(selected_datasets)
        concatenated = concatenated.shuffle()
        dataset_dict = DatasetDict({'train': concatenated})
        dataset_dict.save_to_disk(untokenized_path)
        print(f"Multinomial sampled dataset saved to {untokenized_path} ({len(concatenated)} examples)", file=sys.stderr)

    return untokenized_path


def load_or_tokenize_dataset(
    untokenized_path: str,
    tokenized_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    dev_size: float
):
    """
    Load or create tokenized dataset with train/test split.

    Args:
        untokenized_path: Path to untokenized dataset
        tokenized_path: Path where tokenized dataset should be saved/loaded
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length for tokenization
        dev_size: Fraction (0 < dev_size < 1) or absolute count (dev_size >= 1)
                  of data to use for development/test set

    Returns:
        Dataset dictionary with 'train' and 'test' splits
    """
    if not os.path.exists(tokenized_path):
        if dev_size <= 0:
            raise ValueError(f"dev_size must be positive, got {dev_size}")

        print(f"Tokenizing dataset with vocab size {len(tokenizer)}", file=sys.stderr)
        dataset = load_from_disk(untokenized_path)
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples['text'], max_length=max_length, truncation=True
            ),
            batched=True,
            remove_columns='text'
        )

        # dev_size >= 1 is interpreted as absolute count, < 1 as fraction
        test_size = int(dev_size) if dev_size >= 1 else dev_size
        dataset = dataset['train'].train_test_split(test_size=test_size)
        dataset.save_to_disk(tokenized_path)
        print(f"Tokenized dataset saved to {tokenized_path}", file=sys.stderr)
    else:
        print(f"Loading tokenized dataset from {tokenized_path}", file=sys.stderr)
        dataset = load_from_disk(tokenized_path)

    return dataset
