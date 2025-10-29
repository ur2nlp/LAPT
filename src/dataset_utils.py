"""
Utilities for loading and processing datasets for language-adaptive pretraining.

This module handles downloading OSCAR corpus data, converting it to line-based format,
tokenizing with provided tokenizers, and caching results.
"""

import os
import sys
from itertools import chain

from datasets import load_dataset, load_from_disk
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


def load_or_download_untokenized_dataset(
    dataset_path: str,
    language_code: str
) -> str:
    """
    Load or download untokenized OSCAR dataset.

    Args:
        dataset_path: Base path for dataset storage
        language_code: Two-letter language code for OSCAR corpus

    Returns:
        Path to the untokenized dataset
    """
    untokenized_path = os.path.join(dataset_path, "untokenized")

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
