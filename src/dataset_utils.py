"""
Utilities for loading and processing datasets for language-adaptive pretraining.

This module handles downloading OSCAR corpus data, converting it to line-based format,
tokenizing with provided tokenizers, and caching results.
"""

import os
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

        lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

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
            source_dict_config = DictConfig(source_config)

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
        dev_size: Fraction of data to use for development/test set

    Returns:
        Dataset dictionary with 'train' and 'test' splits
    """
    if not os.path.exists(tokenized_path):
        print(f"Tokenizing dataset with vocab size {len(tokenizer)}", file=sys.stderr)
        dataset = load_from_disk(untokenized_path)
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples['text'], max_length=max_length, truncation=True
            ),
            batched=True,
            remove_columns='text'
        )
        dataset = dataset['train'].train_test_split(test_size=dev_size)
        dataset.save_to_disk(tokenized_path)
        print(f"Tokenized dataset saved to {tokenized_path}", file=sys.stderr)
    else:
        print(f"Loading tokenized dataset from {tokenized_path}", file=sys.stderr)
        dataset = load_from_disk(tokenized_path)

    return dataset
