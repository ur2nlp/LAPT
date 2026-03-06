"""
Utilities for loading and processing datasets for language-adaptive pretraining.

This module handles downloading OSCAR corpus data, converting it to line-based format,
tokenizing with provided tokenizers, and caching results.
"""

import glob
import json
import os
import random
import sys
from itertools import chain

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer


def _get_source_id(config: DictConfig, fallback: str = None) -> str:
    """
    Extract source identifier from config, with backwards compatibility.

    Checks 'id' field first, then falls back to deprecated 'language' field,
    then to the provided fallback string.

    Args:
        config: Source configuration (DictConfig)
        fallback: Default value if neither 'id' nor 'language' is present

    Returns:
        Source identifier string
    """
    source_id = getattr(config, 'id', None)
    if not source_id:
        # Backwards compatibility: check for deprecated 'language' field
        source_id = getattr(config, 'language', None)
        if source_id:
            print(
                f"Warning: 'language' field for source identification is deprecated, "
                f"use 'id' instead (found language='{source_id}')",
                file=sys.stderr
            )
    if not source_id:
        source_id = fallback
    return source_id


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


def collect_from_stream(stream, limit: int) -> Dataset:
    """
    Collect examples from a streaming dataset up to a limit.

    Args:
        stream: An iterable of examples (typically from load_dataset with streaming=True)
        limit: Maximum number of examples to collect

    Returns:
        Dataset containing the collected examples
    """
    samples = []
    for i, example in enumerate(stream):
        if i >= limit:
            break
        samples.append(example)
    return Dataset.from_list(samples)


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
    parameters, update DatasetConfig in artifact_configs.py to track them.
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
        min_words_per_line = getattr(dataset_config, 'min_words_per_line', None)
        oversampling_factor = getattr(dataset_config, 'oversampling_factor', 3)
        return _load_huggingface_dataset(
            cache_dir, name, config, split, text_column, max_samples, min_words_per_line,
            oversampling_factor
        )
    elif dataset_type == 'plaintext':
        file_path = dataset_config.path
        return _load_plaintext_dataset(cache_dir, file_path)
    elif dataset_type == 'plaintext_dir':
        directory = dataset_config.directory
        pattern = getattr(dataset_config, 'pattern', '*.txt')
        return _load_plaintext_dir_dataset(cache_dir, directory, pattern)
    elif dataset_type == 'concat':
        sources = dataset_config.sources
        parent_id = _get_source_id(dataset_config, fallback=None)
        return _load_concat_dataset(cache_dir, sources, parent_id)
    elif dataset_type == 'multinomial':
        sources = dataset_config.sources
        alpha = dataset_config.alpha
        total_samples = dataset_config.total_samples
        return _load_multinomial_dataset(cache_dir, sources, alpha, total_samples, dev_size)
    elif dataset_type == 'instruction_jsonl':
        file_path = dataset_config.path
        return _load_instruction_jsonl_dataset(cache_dir, file_path)
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
    max_samples: int = None,
    min_words_per_line: int = None,
    oversampling_factor: int = 3
) -> str:
    """
    Load a generic HuggingFace dataset.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        name: HuggingFace dataset name (e.g., 'wikitext', 'c4')
        config: Dataset configuration/subset (e.g., 'wikitext-103-v1'), optional
        split: Which split to load (default: 'train')
        text_column: Name of the column containing text (default: 'text')
        max_samples: Maximum number of LINES to load (after splitting docs), uses streaming
            if specified (optional)
        min_words_per_line: Minimum number of space-separated words per line
            (filters out titles/headers)
        oversampling_factor: When max_samples specified, download this many times more documents
            than estimated needed to maintain document diversity (default: 3). Higher values =
            better diversity but more memory.

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
            print(f"  Max samples (lines): {max_samples}", file=sys.stderr)
            print(f"  Oversampling factor: {oversampling_factor}x", file=sys.stderr)

        # Use streaming if max_samples specified to avoid downloading entire dataset
        if max_samples:
            stream = load_dataset(
                name,
                config,
                split=split,
                streaming=True
            )

            # Phase 1: Sample a small batch to estimate lines per document
            # This helps us download the right number of documents
            estimation_sample_size = min(1000, max_samples // 10)
            print(
                f"  Phase 1: Sampling {estimation_sample_size} documents to estimate lines/doc",
                file=sys.stderr
            )

            # Convert estimation batch to dataset and measure lines/doc
            # Use same processing pipeline as main data for accurate estimation
            estimation_dataset = collect_from_stream(stream, estimation_sample_size)
            num_estimation_docs = len(estimation_dataset)
            estimation_dataset = _docs_to_filtered_lines(
                estimation_dataset, text_column, min_words_per_line
            )

            lines_per_doc = len(estimation_dataset) / num_estimation_docs if num_estimation_docs else 1
            print(
                f"  Estimated {lines_per_doc:.1f} lines per document (after all filters)",
                file=sys.stderr
            )

            # Check if estimation found any valid lines
            if lines_per_doc == 0:
                raise ValueError(
                    f"Estimation phase found 0 lines per document after filtering. "
                    f"This suggests min_words_per_line={min_words_per_line} is too strict, "
                    f"or the dataset has no suitable content."
                )

            # Phase 2: Calculate how many documents to download with oversampling
            # We oversample to maintain document diversity, then randomly sample lines at the end
            docs_needed = int((max_samples / lines_per_doc) * oversampling_factor)

            print(f"  Phase 2: Downloading {docs_needed} documents total", file=sys.stderr)

            # Download all documents from fresh stream
            # (Restarting stream is simpler than trying to resume/combine with estimation samples)
            stream = load_dataset(
                name,
                config,
                split=split,
                streaming=True
            )

            dataset = collect_from_stream(stream, docs_needed)
            print(f"  Downloaded {len(dataset)} documents", file=sys.stderr)
        else:
            dataset = load_dataset(name, config, split=split)

        # Convert to line-based format (rename column if needed, split docs on newlines)
        # Note: Could also pass min_words_per_line here if detailed filtering logs aren't needed
        dataset = _docs_to_filtered_lines(dataset, text_column, min_words_per_line=None)
        print(f"  Converted to {len(dataset)} lines from documents", file=sys.stderr)

        # Filter out short lines (e.g., section titles) if min_words_per_line specified
        if min_words_per_line is not None:
            original_size = len(dataset)
            dataset = dataset.filter(
                lambda x: len(x['text'].split()) >= min_words_per_line
            )
            filtered_size = len(dataset)
            print(
                f"  Filtered {original_size - filtered_size} lines with < {min_words_per_line} words "
                f"({filtered_size} lines remaining)",
                file=sys.stderr
            )

            # Check if we have enough lines after filtering
            if max_samples and filtered_size < max_samples:
                print(
                    f"Warning: After filtering, only {filtered_size} lines remain, but "
                    f"{max_samples} requested. Consider increasing oversampling_factor "
                    f"(current: {oversampling_factor}) or reducing min_words_per_line.",
                    file=sys.stderr
                )

        # If max_samples specified, randomly sample to exactly that many lines
        # This maintains document diversity from oversampling while controlling final size
        if max_samples and len(dataset) > max_samples:
            print(
                f"  Randomly sampling {max_samples} lines from {len(dataset)} available lines",
                file=sys.stderr
            )
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(sorted(indices))
        elif max_samples and len(dataset) < max_samples:
            print(
                f"  Note: Got {len(dataset)} lines, which is less than requested {max_samples}",
                file=sys.stderr
            )

        # Wrap in DatasetDict for consistency with other loaders
        dataset_dict = DatasetDict({'train': dataset})
        dataset_dict.save_to_disk(untokenized_path)
        print(f"Untokenized dataset saved to {untokenized_path}", file=sys.stderr)

    return untokenized_path


def _docs_to_filtered_lines(
    dataset: Dataset,
    text_column: str = 'text',
    min_words_per_line: int = None
) -> Dataset:
    """
    Convert document-based dataset to line-based format with optional filtering.

    This helper standardizes the transformation pipeline used by HuggingFace dataset loaders.

    Args:
        dataset: Dataset with document text
        text_column: Name of the text column (will be renamed to 'text' if different)
        min_words_per_line: Minimum words per line to keep (None to skip filtering)

    Returns:
        Dataset with one line per example
    """
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

    # Filter short lines if specified
    if min_words_per_line is not None:
        dataset = dataset.filter(
            lambda x: len(x['text'].split()) >= min_words_per_line
        )

    return dataset


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


def _load_plaintext_dir_dataset(cache_dir: str, directory: str, pattern: str = '*.txt') -> str:
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


def _load_instruction_jsonl_file(file_path: str) -> tuple[list[str], list[str]]:
    """
    Load prompts and responses from an instruction JSONL file.

    Each line should be a JSON object with 'prompt' and 'response' fields:
    {"prompt": "Translate to Gothic: hello\\nResponse:", "response": " hails"}

    Args:
        file_path: Path to JSONL file

    Returns:
        Tuple of (prompts, responses) lists
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instruction JSONL file not found: {file_path}")

    prompts = []
    responses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")

            if 'prompt' not in obj or 'response' not in obj:
                raise ValueError(
                    f"Line {line_num} missing 'prompt' or 'response' field. "
                    f"Got keys: {list(obj.keys())}"
                )
            prompts.append(obj['prompt'])
            responses.append(obj['response'])

    if not prompts:
        raise ValueError(f"JSONL file {file_path} contains no valid examples")

    return prompts, responses


def _load_instruction_jsonl_dataset(cache_dir: str, file_path: str) -> str:
    """
    Load instruction-tuning data from JSONL file(s).

    Each line should be a JSON object with 'prompt' and 'response' fields:
    {"prompt": "Translate to Gothic: hello\\nResponse:", "response": " hails"}

    Unlike plaintext datasets (which have 'text' column), instruction datasets
    have separate 'prompt' and 'response' columns. This allows for loss masking
    during training where only the response tokens contribute to the loss.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        file_path: Path to JSONL file

    Returns:
        Path to the untokenized dataset (with 'prompt' and 'response' columns)
    """
    untokenized_path = os.path.join(cache_dir, "untokenized")

    if not os.path.exists(untokenized_path):
        print(f"Loading instruction data from {file_path}", file=sys.stderr)

        prompts, responses = _load_instruction_jsonl_file(file_path)

        print(f"Loaded {len(prompts)} instruction examples from JSONL file", file=sys.stderr)

        dataset = Dataset.from_dict({'prompt': prompts, 'response': responses})
        dataset_dict = DatasetDict({'train': dataset})
        dataset_dict.save_to_disk(untokenized_path)
        print(f"Untokenized instruction dataset saved to {untokenized_path}", file=sys.stderr)

    return untokenized_path


def _load_concat_dataset(cache_dir: str, sources: list, parent_id: str = None) -> str:
    """
    Concatenate multiple dataset sources into a single dataset.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        sources: List of dataset source configurations (may include 'id' field for naming)
        parent_id: Optional id from parent concat config (used for fallback naming)

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
            # Wrap in DictConfig for recursive dispatching
            source_dict_config = DictConfig(source_config)

            # Determine source cache name:
            # 1. Use source's id field if present (or deprecated 'language')
            # 2. Use parent_id_{idx} if parent has id
            # 3. Fall back to source_{idx}
            default_id = f"{parent_id}_{idx}" if parent_id else f"source_{idx}"
            source_id = _get_source_id(source_dict_config, fallback=default_id)

            source_cache = os.path.join(cache_dir, source_id)

            # Recursively load each source (supports nested concat/multinomial)
            source_path = load_untokenized_dataset(
                dataset_config=source_dict_config,
                cache_dir=source_cache
            )

            source_dataset = load_from_disk(source_path)
            datasets_to_concat.append(source_dataset['train'])
            print(
                f"  Source {idx} ({source_id}): {len(source_dataset['train'])} examples",
                file=sys.stderr
            )

        concatenated = concatenate_datasets(datasets_to_concat)
        dataset_dict = DatasetDict({'train': concatenated})
        dataset_dict.save_to_disk(untokenized_path)
        print(
            f"Concatenated dataset saved to {untokenized_path} ({len(concatenated)} total examples)",
            file=sys.stderr
        )

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
        sources: List of dataset source configurations (should have 'id' field for naming).
            Each source can optionally include a 'dev_size' field to override the global dev_size.
        alpha: Temperature parameter for reweighting (< 1 upsamples smaller datasets)
        total_samples: Total number of training examples to sample (dev set size is separate)
        dev_size: Global default fraction of each source to use for dev set (must be between 0 and
            1). Individual sources can override this with their own dev_size field, which can be
            fractional (0 < x < 1) or absolute (>= 1) for that specific source. Use -1 to skip dev
            split (either globally or per-source).

    Returns:
        Path to the untokenized sampled dataset (DatasetDict with train and per-source dev splits)
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
            "Use dev_size=-1 to skip dev split (e.g., when using external dev sets). "
            "Fixed-size dev sets are not supported for multinomial sampling."
        )

    if skip_dev_split:
        print("WARNING: dev_size=-1 skips dev split creation.", file=sys.stderr)
        print(
            "If using this dataset for model training (not FOCUS), this will cause dev-set "
            "contamination as upsampled training data won't have a held-out dev set. Only use " 
            "dev_size=-1 for datasets that don't need evaluation (e.g., FOCUS training), or ones "
            "that have an external dev set.",
            file=sys.stderr
        )

    untokenized_path = os.path.join(cache_dir, "untokenized")

    if not os.path.exists(untokenized_path):
        print(
            f"Multinomial sampling from {len(sources)} sources with alpha={alpha}", file=sys.stderr
        )
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
            source_id, train_data, dev_data = _load_and_split_source(
                source_config, cache_dir, dev_size, idx
            )

            train_datasets.append(train_data)
            if dev_data is not None:
                dev_datasets.append(dev_data)
                dev_names.append(source_id)
            train_sizes.append(len(train_data))

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
        for dataset, num_samples in zip(train_datasets, samples_per_source):
            indices = _exhaust_first_sample(len(dataset), num_samples)
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
            print(
                f"  Dev splits: {', '.join(dev_names)} "
                f"({sum(len(d) for d in dev_datasets)} examples total, natural proportions)",
                file=sys.stderr
            )

    return untokenized_path


def _load_and_split_source(
    source_config,
    cache_dir: str,
    global_dev_size: float,
    idx: int
) -> tuple:
    """
    Load a single source dataset and split into train/dev.

    Helper for _load_multinomial_dataset. Handles per-source dev_size overrides
    and validation. Logs progress for this source.

    Args:
        source_config: Source configuration dict
        cache_dir: Parent cache directory for this multinomial dataset
        global_dev_size: Default dev_size (can be overridden per-source)
        idx: Source index (for default naming and error messages)

    Returns:
        Tuple of (source_id, train_data, dev_data) where dev_data is None
        if dev split was skipped for this source.
    """
    source_dict_config = DictConfig(source_config)

    # Determine source id from id field (or deprecated 'language'), default to source_{idx}
    source_id = _get_source_id(source_dict_config, fallback=f"source_{idx}")

    source_cache = os.path.join(cache_dir, source_id)

    source_path = load_untokenized_dataset(
        dataset_config=source_dict_config,
        cache_dir=source_cache
    )

    source_dataset = load_from_disk(source_path)
    full_data = source_dataset['train']

    # Check for per-source dev_size override (fallback to global dev_size)
    source_dev_size = getattr(source_dict_config, 'dev_size', global_dev_size)
    skip_source_dev_split = (source_dev_size == -1)

    # Validate per-source dev_size
    if source_dev_size == 0:
        raise ValueError(
            f"Source {idx}: dev_size=0 is ambiguous. "
            "Use dev_size=-1 to explicitly skip dev split."
        )
    elif not skip_source_dev_split and source_dev_size < 0:
        raise ValueError(
            f"Source {idx}: dev_size must be positive or -1 to skip, got {source_dev_size}."
        )

    # Split into train/dev BEFORE upsampling (skip if dev_size=-1)
    if not skip_source_dev_split:
        split_dataset = full_data.train_test_split(test_size=source_dev_size, seed=1)
        train_data = split_dataset['train']
        dev_data = split_dataset['test']
    else:
        train_data = full_data
        dev_data = None

    # Log with indication if using per-source override
    dev_size_label = (
        f"dev_size={source_dev_size}" if hasattr(source_dict_config, 'dev_size')
        else f"global dev_size={source_dev_size}"
    )
    if dev_data is not None:
        print(
            f"  Source {idx} ({source_id}): "
            f"{len(train_data)} train, {len(dev_data)} dev examples ({dev_size_label})",
            file=sys.stderr
        )
    else:
        print(
            f"  Source {idx} ({source_id}): "
            f"{len(train_data)} examples (no dev split, {dev_size_label})",
            file=sys.stderr
        )

    return source_id, train_data, dev_data


def _exhaust_first_sample(dataset_size: int, num_samples: int) -> list[int]:
    """
    Generate sample indices using exhaust-first strategy.

    Helper for _load_multinomial_dataset. When num_samples > dataset_size,
    includes ALL examples once before any duplication. This maximizes coverage
    of unique examples, which is critical for low-resource datasets.

    Args:
        dataset_size: Number of examples in the dataset
        num_samples: Number of samples to draw

    Returns:
        List of indices (may contain duplicates if num_samples > dataset_size)
    """
    if num_samples <= dataset_size:
        # Sample without replacement
        return random.sample(range(dataset_size), num_samples)
    else:
        # Include ALL examples once, then sample remainder with replacement
        all_indices = list(range(dataset_size))
        num_additional = num_samples - dataset_size
        additional_indices = random.choices(range(dataset_size), k=num_additional)
        indices = all_indices + additional_indices
        random.shuffle(indices)  # Shuffle to mix exhaustive + repeated samples
        return indices


def _tokenize_instruction_examples(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> dict:
    """
    Tokenize instruction examples with label masking.

    For each example, tokenizes prompt and response separately, then concatenates.
    Creates labels where prompt tokens are masked (-100) and only response tokens
    contribute to the loss.

    Also handles mixed datasets where some examples have prompt/response (instruction)
    and others have only text (plaintext). Plaintext examples get labels = input_ids
    (standard causal LM loss on all tokens).

    Args:
        examples: Batch with 'prompt' and 'response' fields, optionally 'text'
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length (prompt + response combined)

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels' fields
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    # Get text column if it exists (for mixed datasets)
    texts = examples.get('text', [None] * len(examples['prompt']))

    for prompt, response, text in zip(examples['prompt'], examples['response'], texts):
        # Check if this is an instruction example or plaintext
        is_instruction = prompt is not None and response is not None

        if is_instruction:
            # Instruction example: tokenize prompt and response separately
            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False
            )

            response_tokens = tokenizer(
                response,
                add_special_tokens=False,
                truncation=False
            )

            # Concatenate
            # TODO: fix linting issue here
            input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids']
            attention_mask = prompt_tokens['attention_mask'] + response_tokens['attention_mask']

            # Create labels: -100 for prompt (masked), actual tokens for response
            prompt_length = len(prompt_tokens['input_ids'])
            labels = [-100] * prompt_length + response_tokens['input_ids']
        else:
            # Plaintext example: standard tokenization, labels = input_ids
            if text is None:
                raise ValueError(
                    "Example has neither valid prompt/response nor text. "
                    "Mixed datasets must have 'text' for plaintext examples."
                )

            tokens = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )

            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            # Standard causal LM: predict all tokens
            labels = list(input_ids)

        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'labels': all_labels
    }


def load_tokenized_dataset(
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

    Also handles instruction datasets (with 'prompt'/'response' columns) by creating
    labels with prompt tokens masked (-100).

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
        - Instruction datasets: same structure but with 'labels' field for loss masking

    NOTE: Parameters affecting the tokenized dataset artifact (max_length, dev_size, plus all
    upstream dataset and tokenizer parameters) should be tracked in TokenizedDatasetConfig
    in artifact_configs.py.
    """
    if not os.path.exists(tokenized_path):
        print(
            f"Tokenizing dataset with {tokenizer.name_or_path} (vocab size {len(tokenizer)})",
            file=sys.stderr
        )
        dataset = load_from_disk(untokenized_path)

        # Check if this is an instruction dataset (has 'prompt'/'response' instead of 'text')
        sample_split = list(dataset.keys())[0]
        is_instruction_dataset = (
            'prompt' in dataset[sample_split].column_names
            and 'response' in dataset[sample_split].column_names
        )

        if is_instruction_dataset:
            print(
                "Detected instruction dataset format, tokenizing with label masking",
                file=sys.stderr
            )
            # Determine columns to remove (prompt, response, and text if present for mixed datasets)
            columns_to_remove = ['prompt', 'response']
            if 'text' in dataset[sample_split].column_names:
                columns_to_remove.append('text')
            dataset = dataset.map(
                lambda examples: _tokenize_instruction_examples(examples, tokenizer, max_length),
                batched=True,
                remove_columns=columns_to_remove
            )
        else:
            # Standard text dataset
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


def load_external_eval_set(
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
        is_instruction = False

    elif file_format == 'jsonl':
        # Load JSONL file
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
        is_instruction = False

    elif file_format == 'instruction_jsonl':
        # Load instruction JSONL with prompt/response fields
        prompts, responses = _load_instruction_jsonl_file(path)
        dataset = Dataset.from_dict({'prompt': prompts, 'response': responses})
        is_instruction = True

    else:
        raise ValueError(
            f"Unsupported format '{file_format}' for external eval set. "
            f"Supported formats: 'plaintext', 'jsonl', 'instruction_jsonl'"
        )

    print(f"  Loaded {len(dataset)} examples", file=sys.stderr)

    # Tokenize the dataset
    if is_instruction:
        # Instruction format: use label masking (loss only on response)
        dataset = dataset.map(
            lambda examples: _tokenize_instruction_examples(examples, tokenizer, max_length),
            batched=True,
            remove_columns=['prompt', 'response'],
            desc=f"Tokenizing external eval set '{name}'"
        )
    else:
        # Plain text format: standard tokenization
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples['text'], max_length=max_length, truncation=True
            ),
            batched=True,
            remove_columns=['text'],
            desc=f"Tokenizing external eval set '{name}'"
        )

    return dataset


def prepare_eval_datasets(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    external_eval_sets: list = None
):
    """
    Prepare evaluation datasets from loaded data and optional external sources.

    Handles both:
    - Extracting dev splits from tokenized dataset (single or per-language)
    - Loading and merging external evaluation sets

    Args:
        dataset: Tokenized DatasetDict with train and dev/test splits
        tokenizer: Tokenizer for tokenizing external eval sets
        max_length: Max sequence length for tokenization
        external_eval_sets: Optional list of external eval configs, each with
            'name', 'path', and optional 'format' keys

    Returns:
        Either a single Dataset (standard case) or dict of Datasets (multinomial
        or when external eval sets are added)
    """
    # Dev splits are any non-train splits except 'test'
    dev_splits = [key for key in dataset.keys() if key != 'train' and key != 'test']

    if dev_splits:
        # Multinomial sampling case: multiple per-language dev sets
        eval_dataset = {key: dataset[key] for key in dev_splits}
        print(
            f"Using {len(eval_dataset)} per-language eval sets: {', '.join(dev_splits)}",
            file=sys.stderr
        )
    else:
        # Standard case: single dev/test split
        eval_dataset = dataset['test']

    # Load external evaluation sets if configured
    if external_eval_sets:
        # If eval_dataset is not already a dict, convert it
        if not isinstance(eval_dataset, dict):
            eval_dataset = {'dev': eval_dataset}
            print("Converted single eval dataset to dict for external eval sets", file=sys.stderr)

        # Load and add each external eval set
        for eval_config in external_eval_sets:
            name = eval_config['name']

            # Check for name conflicts
            if name in eval_dataset:
                raise ValueError(
                    f"External eval set name '{name}' conflicts with existing eval set. "
                    f"Existing eval sets: {list(eval_dataset.keys())}"
                )

            external_dataset = load_external_eval_set(
                eval_config=eval_config,
                tokenizer=tokenizer,
                max_length=max_length
            )
            eval_dataset[name] = external_dataset
            print(
                f"Added external eval set '{name}' with {len(external_dataset)} examples",
                file=sys.stderr
            )

    return eval_dataset


class DataCollatorForInstructionTuning:
    """
    Data collator for instruction tuning with pre-computed labels.

    This collator expects examples that already have 'labels' field with
    prompt tokens masked as -100. It handles padding for:
    - input_ids: padded with tokenizer.pad_token_id
    - attention_mask: padded with 0
    - labels: padded with -100 (ignored by CrossEntropyLoss)

    Unlike DataCollatorForLanguageModeling, this collator does NOT create labels
    from input_ids - it uses the pre-computed labels from the dataset.

    Args:
        tokenizer: Tokenizer used for padding
        padding: Padding strategy ('longest', 'max_length', or False)
        max_length: Maximum length when padding='max_length'
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding: str = 'longest',
        max_length: int = None
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token_id is None:
            # Use EOS token as pad token if not set
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features: list) -> dict:
        """
        Collate a batch of features.

        Args:
            features: List of dicts with 'input_ids', 'attention_mask', and 'labels'

        Returns:
            Batch dict with padded tensors
        """
        import torch

        # Separate labels from other features for custom padding
        labels = [f['labels'] for f in features]
        # Remove labels temporarily for tokenizer padding
        features_without_labels = [{k: v for k, v in f.items() if k != 'labels'} for f in features]

        # Use tokenizer's padding for input_ids and attention_mask
        batch = self.tokenizer.pad(
            features_without_labels,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Pad labels with -100 (ignored by loss function)
        max_label_length = max(len(l) for l in labels)
        padded_labels = []
        for label in labels:
            padding_length = max_label_length - len(label)
            # Pad on the right with -100
            padded_label = label + [-100] * padding_length
            padded_labels.append(padded_label)

        batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


def is_instruction_dataset(dataset) -> bool:
    """
    Check if a dataset is an instruction-tuning dataset (has pre-computed labels).

    Args:
        dataset: A Dataset or DatasetDict

    Returns:
        True if the dataset has 'labels' column, indicating instruction format
    """
    if hasattr(dataset, 'keys'):
        # DatasetDict - check the first split
        sample_split = list(dataset.keys())[0]
        return 'labels' in dataset[sample_split].column_names
    else:
        # Single Dataset
        return 'labels' in dataset.column_names
