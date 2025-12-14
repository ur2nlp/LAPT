"""
Utilities for loading and processing datasets for language-adaptive pretraining.

This module handles downloading OSCAR corpus data, converting it to line-based format,
tokenizing with provided tokenizers, and caching results.
"""

import glob
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
        Dictionary with 'text' field containing individual lines (blank lines filtered out)
    """
    return {
        'text': list(chain(
            *[[line.strip() for line in doc.split('\n') if line.strip()]
              for doc in examples['text']]
        ))
    }


def load_untokenized_dataset(dataset_config, cache_dir: str, dev_size: float = None) -> str:
    """
    Load untokenized dataset based on configuration.

    This dispatcher routes to the appropriate loader based on dataset type.

    Args:
        dataset_config: Dataset configuration object with type and source info
        cache_dir: Base directory for caching dataset artifacts
        dev_size: Fraction of data for dev set (only used for multinomial sampling)

    Returns:
        Path to the untokenized dataset

    NOTE: Parameters affecting the dataset artifact vary by type (language for OSCAR, path for
    plaintext, alpha/total_samples for multinomial, etc.). When adding new dataset types or
    parameters, update extract_dataset_config() in config_utils.py to track them.
    """
    # Default to oscar for backward compatibility if type not specified
    dataset_type = getattr(dataset_config, 'type', 'oscar')

    if dataset_type == 'oscar':
        language_code = dataset_config.language
        return _load_oscar_dataset(cache_dir, language_code)
    elif dataset_type == 'huggingface':
        name = dataset_config.name
        config = getattr(dataset_config, 'config', None)
        split = getattr(dataset_config, 'split', 'train')
        text_column = getattr(dataset_config, 'text_column', 'text')
        max_samples = getattr(dataset_config, 'max_samples', None)
        return _load_huggingface_dataset(cache_dir, name, config, split, text_column, max_samples)
    elif dataset_type == 'plaintext':
        file_path = dataset_config.path
        return _load_plaintext_dataset(cache_dir, file_path)
    elif dataset_type == 'plaintext_dir':
        directory = dataset_config.directory
        pattern = getattr(dataset_config, 'pattern', '*.txt')
        return _load_plaintext_dir_dataset(cache_dir, directory, pattern)
    elif dataset_type == 'concat':
        sources = dataset_config.sources
        return _load_concat_dataset(cache_dir, sources)
    elif dataset_type == 'multinomial':
        sources = dataset_config.sources
        alpha = dataset_config.alpha
        total_samples = dataset_config.total_samples
        return _load_multinomial_dataset(cache_dir, sources, alpha, total_samples, dev_size)
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


def _load_huggingface_dataset(
    cache_dir: str,
    name: str,
    config: str = None,
    split: str = 'train',
    text_column: str = 'text',
    max_samples: int = None
) -> str:
    """
    Load a generic HuggingFace dataset.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        name: HuggingFace dataset name (e.g., 'wikitext', 'c4')
        config: Dataset configuration/subset (e.g., 'wikitext-103-v1'), optional
        split: Which split to load (default: 'train')
        text_column: Name of the column containing text (default: 'text')
        max_samples: Maximum number of samples to load, uses streaming if specified (optional)

    Returns:
        Path to the untokenized dataset
    """
    untokenized_path = os.path.join(cache_dir, "untokenized")

    if not os.path.exists(untokenized_path):
        print(f"Downloading and preparing HuggingFace dataset: {name}", file=sys.stderr)
        if config:
            print(f"  Config: {config}", file=sys.stderr)
        print(f"  Split: {split}", file=sys.stderr)
        if max_samples:
            print(f"  Max samples: {max_samples}", file=sys.stderr)

        # Use streaming if max_samples specified to avoid downloading entire dataset
        if max_samples:
            dataset = load_dataset(
                name,
                config,
                split=split,
                streaming=True
            )
            # Take only max_samples and convert to regular dataset
            samples = []
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break
                samples.append(example)

            dataset = Dataset.from_list(samples)
            print(f"Loaded {len(dataset)} samples", file=sys.stderr)
        else:
            dataset = load_dataset(name, config, split=split)

        # Standardize column name to 'text' if needed
        if text_column != 'text':
            dataset = dataset.rename_column(text_column, 'text')

        # Convert to line-based format (split documents on newlines)
        original_columns = dataset.column_names
        dataset = dataset.map(
            docs_to_lines,
            batched=True,
            remove_columns=original_columns
        )

        # Wrap in DatasetDict for consistency with other loaders
        dataset_dict = DatasetDict({'train': dataset})
        dataset_dict.save_to_disk(untokenized_path)
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

        if not lines:
            raise ValueError(f"Plaintext file {file_path} contains no non-empty lines")

        print(f"Loaded {len(lines)} lines from plaintext file", file=sys.stderr)

        dataset = Dataset.from_dict({'text': lines})
        dataset_dict = DatasetDict({'train': dataset})
        dataset_dict.save_to_disk(untokenized_path)
        print(f"Untokenized dataset saved to {untokenized_path}", file=sys.stderr)

    return untokenized_path


def _load_plaintext_dir_dataset(cache_dir: str, directory: str, pattern: str) -> str:
    """
    Load all plaintext files from a directory and concatenate them.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        directory: Directory containing text files
        pattern: Glob pattern for matching files (e.g., "*.txt", "*.on.txt")

    Returns:
        Path to the untokenized concatenated dataset
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")

    # Find all matching files
    file_paths = sorted(glob.glob(os.path.join(directory, pattern)))

    if not file_paths:
        raise ValueError(f"No files found matching pattern '{pattern}' in {directory}")

    print(f"Found {len(file_paths)} files matching '{pattern}' in {directory}", file=sys.stderr)

    # Create sources list for concat (reuse plaintext loader for each file)
    sources = [
        {'type': 'plaintext', 'path': path}
        for path in file_paths
    ]

    # Reuse concat implementation
    return _load_concat_dataset(cache_dir, sources)


def _load_concat_dataset(cache_dir: str, sources: list) -> str:
    """
    Concatenate multiple dataset sources into a single dataset.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        sources: List of dataset source configurations

    Returns:
        Path to the untokenized concatenated dataset
    """
    if not sources:
        raise ValueError("Cannot concatenate datasets: sources list is empty")

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
    cache_dir: str, sources: list, alpha: float, total_samples: int, dev_size: float = None
) -> str:
    """
    Sample from multiple dataset sources using temperature-scaled multinomial sampling.

    Splits each source into train/dev BEFORE upsampling to prevent dev set leakage.
    Train splits are upsampled according to alpha, dev splits are kept at natural proportions.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        sources: List of dataset source configurations (should have 'language' field for naming)
        alpha: Temperature parameter for reweighting (< 1 upsamples smaller datasets)
        total_samples: Total number of training examples to sample (dev set size is separate)
        dev_size: Fraction of each source to use for dev set (must be between 0 and 1)

    Returns:
        Path to the untokenized sampled dataset (DatasetDict with train and per-language dev splits)
    """
    if not sources:
        raise ValueError("Cannot sample from datasets: sources list is empty")
    if total_samples <= 0:
        raise ValueError(f"total_samples must be positive, got {total_samples}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if dev_size is None:
        raise ValueError("dev_size must be provided for multinomial sampling")

    # Check for explicit "no dev split" flag
    skip_dev_split = (dev_size == -1)

    if dev_size == 0:
        raise ValueError(
            "dev_size=0 is ambiguous. Use dev_size=-1 to explicitly skip dev split, "
            "or use a value > 0 for fractional split size."
        )
    elif not skip_dev_split and not (0 < dev_size < 1):
        raise ValueError(
            f"Multinomial sampling requires fractional dev_size (0 < dev_size < 1), got {dev_size}. "
            "Use dev_size=-1 to skip dev split (e.g., for FOCUS training). "
            "Fixed-size dev sets are not supported for multinomial sampling."
        )

    if skip_dev_split:
        print("WARNING: dev_size=-1 skips dev split creation.", file=sys.stderr)
        print("  If using this dataset for model training (not FOCUS), this will cause", file=sys.stderr)
        print("  dev-set contamination as upsampled training data won't have a held-out dev set.", file=sys.stderr)
        print("  Only use dev_size=-1 for datasets that don't need evaluation (e.g., FOCUS).", file=sys.stderr)

    untokenized_path = os.path.join(cache_dir, "untokenized")

    if not os.path.exists(untokenized_path):
        print(f"Multinomial sampling from {len(sources)} sources with alpha={alpha}", file=sys.stderr)
        if not skip_dev_split:
            print(f"Dev split: {dev_size:.1%} of each source (before upsampling)", file=sys.stderr)
        else:
            print("No dev split (dev_size=-1, using all data for training)", file=sys.stderr)

        train_datasets = []
        dev_datasets = []
        dev_names = []
        train_sizes = []

        # Load all sources, split into train/dev, and record train sizes
        for idx, source_config in enumerate(sources):
            source_cache = os.path.join(cache_dir, f"source_{idx}")
            source_dict_config = DictConfig(source_config)

            source_path = load_untokenized_dataset(
                dataset_config=source_dict_config,
                cache_dir=source_cache
            )

            source_dataset = load_from_disk(source_path)
            full_data = source_dataset['train']

            # Split into train/dev BEFORE upsampling (skip if dev_size=-1)
            if not skip_dev_split:
                split_dataset = full_data.train_test_split(test_size=dev_size, seed=1)
                train_data = split_dataset['train']
                dev_data = split_dataset['test']
            else:
                # No dev split - use all data for training
                train_data = full_data
                dev_data = None

            # Determine dev split name from language field or default to source index
            language = getattr(source_dict_config, 'language', None)
            if language:
                dev_name = language
            else:
                dev_name = f"source{idx}"

            train_datasets.append(train_data)
            if not skip_dev_split:
                dev_datasets.append(dev_data)
                dev_names.append(dev_name)
            train_sizes.append(len(train_data))

            if not skip_dev_split:
                print(f"  Source {idx} ({dev_name}): {len(train_data)} train, {len(dev_data)} dev examples", file=sys.stderr)
            else:
                print(f"  Source {idx}: {len(train_data)} examples (no dev split)", file=sys.stderr)

        # Check for empty datasets
        if all(size == 0 for size in train_sizes):
            raise ValueError("Cannot sample: all source datasets are empty")

        # Calculate sampling probabilities for TRAIN data: p_i = (size_i)^alpha / Z
        # alpha < 1 upsamples smaller datasets, alpha > 1 amplifies size differences
        weights = [size ** alpha for size in train_sizes]
        total_weight = sum(weights)
        sampling_probs = [w / total_weight for w in weights]

        # Convert probabilities to integer sample counts
        # Distribute remainder samples round-robin to handle rounding errors
        samples_per_source = [int(prob * total_samples) for prob in sampling_probs]
        remaining = total_samples - sum(samples_per_source)
        for i in range(remaining):
            samples_per_source[i % len(sources)] += 1

        print("Train sampling distribution:", file=sys.stderr)
        for idx, count in enumerate(samples_per_source):
            percentage = 100 * count / total_samples
            print(f"  Source {idx}: {count} samples ({percentage:.2f}%)", file=sys.stderr)

        # Sample and upsample TRAIN data only
        selected_train_datasets = []
        for idx, (dataset, num_samples) in enumerate(zip(train_datasets, samples_per_source)):
            # Use "exhaust-first" sampling to maximize coverage of unique examples
            if num_samples <= len(dataset):
                # Sample without replacement
                indices = random.sample(range(len(dataset)), num_samples)
            else:
                # Include ALL examples once, then sample remainder with replacement
                # This ensures we always see all unique examples before any duplication
                all_indices = list(range(len(dataset)))
                num_additional = num_samples - len(dataset)
                additional_indices = random.choices(range(len(dataset)), k=num_additional)
                indices = all_indices + additional_indices
                random.shuffle(indices)  # Shuffle to mix exhaustive + repeated samples

            # .select() keeps data memory-mapped, handles duplicate indices for sampling with replacement
            selected = dataset.select(indices)
            selected_train_datasets.append(selected)

        # Concatenate and shuffle train data
        concatenated_train = concatenate_datasets(selected_train_datasets)
        concatenated_train = concatenated_train.shuffle(seed=1)

        # Build DatasetDict with train and per-language dev splits
        # Dev splits are NOT upsampled - kept at natural proportions
        dataset_dict = {'train': concatenated_train}
        for dev_name, dev_data in zip(dev_names, dev_datasets):
            dataset_dict[dev_name] = dev_data

        dataset_dict = DatasetDict(dataset_dict)
        dataset_dict.save_to_disk(untokenized_path)

        print(f"Multinomial sampled dataset saved to {untokenized_path}", file=sys.stderr)
        print(f"  Train: {len(concatenated_train)} examples (upsampled)", file=sys.stderr)
        if not skip_dev_split:
            print(f"  Dev splits: {', '.join(dev_names)} ({sum(len(d) for d in dev_datasets)} examples total, natural proportions)", file=sys.stderr)

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

    Handles both simple datasets (creates train/test split) and pre-split datasets
    from multinomial sampling (already has train and per-language dev splits).

    Args:
        untokenized_path: Path to untokenized dataset
        tokenized_path: Path where tokenized dataset should be saved/loaded
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length for tokenization
        dev_size: Fraction (0 < dev_size < 1) or absolute count (dev_size >= 1)
                  of data to use for development/test set (ignored if dataset already split)

    Returns:
        Dataset dictionary with 'train' and dev splits
        - Simple datasets: {'train': ..., 'test': ...}
        - Multinomial datasets: {'train': ..., '{language}': ..., '{language}': ..., ...}
          (e.g., {'train': ..., 'got': ..., 'ang': ..., 'non': ...})

    NOTE: Parameters affecting the tokenized dataset artifact (max_length, dev_size, plus all
    upstream dataset and tokenizer parameters) should be tracked in extract_tokenized_config()
    in config_utils.py.
    """
    if not os.path.exists(tokenized_path):
        print(f"Tokenizing dataset with vocab size {len(tokenizer)}", file=sys.stderr)
        dataset = load_from_disk(untokenized_path)

        # Tokenize all splits
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples['text'], max_length=max_length, truncation=True
            ),
            batched=True,
            remove_columns='text'
        )

        # Check if dataset already has dev splits (from multinomial sampling)
        # Dev splits are any non-train splits (e.g., 'got', 'ang', 'non')
        has_dev_splits = any(key != 'train' and key != 'test' for key in dataset.keys())

        if not has_dev_splits:
            # Normal case - need to split train data into train/test
            if dev_size <= 0:
                raise ValueError(f"dev_size must be positive, got {dev_size}")

            # dev_size >= 1 is interpreted as absolute count, < 1 as fraction
            test_size = int(dev_size) if dev_size >= 1 else dev_size
            dataset = dataset['train'].train_test_split(test_size=test_size)
        else:
            print("Dataset already has per-source dev splits", file=sys.stderr)

        dataset.save_to_disk(tokenized_path)
        print(f"Tokenized dataset saved to {tokenized_path}", file=sys.stderr)
    else:
        print(f"Loading tokenized dataset from {tokenized_path}", file=sys.stderr)
        dataset = load_from_disk(tokenized_path)

    return dataset


def load_and_tokenize_external_eval_set(
    eval_config: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> Dataset:
    """
    Load and tokenize an external evaluation dataset.

    Args:
        eval_config: Dictionary with 'name', 'path', and optional 'format' keys
                    - name: Name for the eval set (used in metrics)
                    - path: Path to the data file
                    - format: 'plaintext' (default) or 'jsonl'
                    - text_column: Column name for jsonl format (default: 'text')
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length for tokenization

    Returns:
        Tokenized Dataset ready for evaluation
    """
    name = eval_config['name']
    path = eval_config['path']
    file_format = eval_config.get('format', 'plaintext')
    text_column = eval_config.get('text_column', 'text')

    print(f"Loading external eval set '{name}' from {path}", file=sys.stderr)

    if not os.path.exists(path):
        raise ValueError(f"External eval set file not found: {path}")

    # Load data based on format
    if file_format == 'plaintext':
        # Read lines from plaintext file
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        dataset = Dataset.from_dict({'text': lines})

    elif file_format == 'jsonl':
        # Load JSONL file
        import json
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if text_column in obj:
                        data.append(obj[text_column])
                    else:
                        raise ValueError(
                            f"JSONL file missing '{text_column}' column: {path}"
                        )
        dataset = Dataset.from_dict({'text': data})

    else:
        raise ValueError(
            f"Unsupported format '{file_format}' for external eval set. "
            f"Supported formats: 'plaintext', 'jsonl'"
        )

    print(f"  Loaded {len(dataset)} examples", file=sys.stderr)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False
        )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc=f"Tokenizing external eval set '{name}'"
    )

    return dataset
