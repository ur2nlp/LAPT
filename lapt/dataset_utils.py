"""
Utilities for loading and processing datasets for language-adaptive pretraining.

This module handles downloading OSCAR corpus data, converting it to line-based format,
tokenizing with provided tokenizers, and caching results.
"""

import glob
import hashlib
import json
import os
import random
import re
import sys

import numpy as np
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import PreTrainedTokenizer

from lapt.artifact_configs import (
    DatasetConfig,
    SourceCacheTracking,
    dict_diff,
    multinomial_mix_slug,
    resolve_dev_size,
)
from lapt.core.artifacts import CachedArtifact
from lapt.sources import (
    HuggingFaceDataset,
    InstructionJsonlDataset,
    OscarDataset,
    PlaintextDataset,
)
from lapt.sources.text_processing import (
    read_instruction_jsonl,
)

SOURCE_CONFIG_FILENAME = "source_config.yaml"


def _validate_source_cache(untokenized_path: str, current: dict) -> None:
    """
    Validate that a cached source dataset was built with the same parameters
    currently requested. Raises on mismatch so stale per-source caches can't
    silently propagate into downstream mixes.

    Pre-refactor caches without tracking are allowed through with a warning
    so existing data isn't invalidated on upgrade; mismatches from that point
    on require an explicit regeneration (e.g. fresh_dataset=true).

    Tracked parameters added after a cache was built are tolerated via
    SourceCacheTracking (keyed by dataset ``type``): a parameter the cached
    config lacks is forgiven when the current value equals its registered
    historical default, while a genuinely different value still mismatches.
    """
    config_path = os.path.join(untokenized_path, SOURCE_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        print(
            f"Note: cached source at {untokenized_path} has no source-level "
            "config tracking (pre-dates tracking support). If you recently "
            "changed source parameters, pass fresh_dataset=true to regenerate.",
            file=sys.stderr,
        )
        return

    with open(config_path) as f:
        cached = yaml.safe_load(f) or {}

    legacy_defaults = SourceCacheTracking.legacy_defaults(current.get('type'))
    if legacy_defaults:
        current = {
            key: value
            for key, value in current.items()
            if not (
                key in legacy_defaults
                and key not in cached
                and value == legacy_defaults[key]
            )
        }

    diffs = dict_diff(cached, current)
    if not diffs:
        return

    raise ValueError(
        f"\n{'=' * 70}\n"
        f"SOURCE CACHE MISMATCH: {untokenized_path}\n"
        f"{'=' * 70}\n"
        f"This cached source dataset was built with different parameters:\n\n"
        + "\n".join(f"  {diff}" for diff in diffs)
        + "\n\n"
        f"This matters because the same source cache is reused across every\n"
        f"mix that references this source id, so stale data would silently\n"
        f"feed downstream sampling.\n\n"
        f"To proceed, either:\n\n"
        f"  1. Regenerate this source by passing fresh_dataset=true\n"
        f"     (this will clear the cache dir and rebuild).\n\n"
        f"  2. Change the source id so it resolves to a different cache dir.\n"
        f"{'=' * 70}\n"
    )


def _save_source_cache_config(untokenized_path: str, current: dict) -> None:
    """Write the source-level tracked config alongside a freshly built cache."""
    config_path = os.path.join(untokenized_path, SOURCE_CONFIG_FILENAME)
    with open(config_path, 'w') as f:
        yaml.dump(current, f, default_flow_style=False, sort_keys=False)


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


def _parse_substitutions(raw) -> list[tuple[str, str]]:
    """
    Normalize a dataset's optional ``substitutions`` config into (pattern,
    replacement) pairs.

    Accepts a list of ``{pattern, replacement}`` mappings (the YAML form). The
    ``replacement`` defaults to an empty string if omitted. Returns an empty
    list when no substitutions are configured.

    Args:
        raw: The raw ``substitutions`` value from the dataset config (a
            ListConfig, list, or None).

    Returns:
        List of (pattern, replacement) string tuples, in declaration order.
    """
    if not raw:
        return []
    if isinstance(raw, (ListConfig, DictConfig)):
        raw = OmegaConf.to_container(raw, resolve=True)

    substitutions = []
    for item in raw:
        if 'pattern' not in item:
            raise ValueError(
                f"Each substitution must specify a 'pattern' (got {item!r})."
            )
        pattern = item['pattern']
        replacement = item.get('replacement', '')
        # Fail fast on a malformed regex rather than at map time.
        re.compile(pattern)
        substitutions.append((pattern, replacement))
    return substitutions


def _apply_substitutions(
    base_path: str,
    substitutions: list[tuple[str, str]],
) -> str:
    """
    Apply a sequence of regex substitutions to every string column of an
    untokenized dataset, caching the result in a sibling directory.

    The substitutions are applied in order to each value of each string-valued
    column (e.g. 'text', or 'prompt'/'response' for instruction sources), so a
    pattern like ``\\n+`` -> ``' '`` collapses newlines to spaces across any
    dataset type. The original (raw) cache at ``base_path`` is left untouched;
    the substituted copy lives at ``{base_path}_sub_{hash}`` keyed on the
    substitution list so changing the patterns rebuilds rather than clobbers.

    Args:
        base_path: Path to the untokenized DatasetDict to transform.
        substitutions: Ordered (pattern, replacement) pairs to apply.

    Returns:
        Path to the substituted untokenized dataset.
    """
    normalized = [
        {'pattern': pattern, 'replacement': replacement}
        for pattern, replacement in substitutions
    ]
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True).encode()
    ).hexdigest()[:8]
    substituted_path = f"{base_path}_sub_{digest}"
    tracked = {
        'type': 'substituted',
        'base': os.path.basename(base_path),
        'substitutions': normalized,
    }

    if os.path.exists(substituted_path):
        _validate_source_cache(substituted_path, tracked)
        return substituted_path

    print(
        f"Applying {len(substitutions)} regex substitution(s) to {base_path}",
        file=sys.stderr,
    )
    compiled = [(re.compile(pattern), replacement) for pattern, replacement in substitutions]
    dataset_dict = load_from_disk(base_path)

    def substitute_batch(examples, string_columns):
        for column in string_columns:
            new_values = []
            for value in examples[column]:
                for pattern, replacement in compiled:
                    value = pattern.sub(replacement, value)
                new_values.append(value)
            examples[column] = new_values
        return examples

    substituted_splits = {}
    for split_name, split_dataset in dataset_dict.items():
        string_columns = [
            name
            for name, feature in split_dataset.features.items()
            if getattr(feature, 'dtype', None) == 'string'
        ]
        substituted_splits[split_name] = split_dataset.map(
            lambda examples, columns=string_columns: substitute_batch(examples, columns),
            batched=True,
        )

    substituted_dict = DatasetDict(substituted_splits)
    substituted_dict.save_to_disk(substituted_path)
    _save_source_cache_config(substituted_path, tracked)
    print(
        f"Substituted untokenized dataset saved to {substituted_path}",
        file=sys.stderr,
    )

    return substituted_path


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

    Any dataset may also carry an optional ``substitutions`` field — a list of
    ``{pattern, replacement}`` regexes applied to every string column after the
    type-specific loader runs (see ``_apply_substitutions``). This is type-agnostic
    because all loaders funnel through this dispatcher.
    """
    # Default to oscar for backward compatibility if type not specified
    dataset_type = getattr(dataset_config, 'type', 'oscar')

    if dataset_type == 'oscar':
        language_code = dataset_config.language
        untokenized_path = _load_oscar_dataset(cache_dir, language_code)
    elif dataset_type == 'huggingface':
        name = dataset_config.name
        config = getattr(dataset_config, 'config', None)
        split = getattr(dataset_config, 'split', 'train')
        text_column = getattr(dataset_config, 'text_column', 'text')
        max_samples = getattr(dataset_config, 'max_samples', None)
        min_words_per_line = getattr(dataset_config, 'min_words_per_line', None)
        oversampling_factor = getattr(dataset_config, 'oversampling_factor', 3)
        split_into_lines = getattr(dataset_config, 'split_into_lines', True)
        untokenized_path = _load_huggingface_dataset(
            cache_dir, name, config, split, text_column, max_samples, min_words_per_line,
            oversampling_factor, split_into_lines
        )
    elif dataset_type == 'plaintext':
        file_path = dataset_config.path
        untokenized_path = _load_plaintext_dataset(cache_dir, file_path)
    elif dataset_type == 'plaintext_dir':
        directory = dataset_config.directory
        pattern = getattr(dataset_config, 'pattern', '*.txt')
        untokenized_path = _load_plaintext_dir_dataset(cache_dir, directory, pattern)
    elif dataset_type == 'concat':
        sources = dataset_config.sources
        parent_id = _get_source_id(dataset_config, fallback=None)
        untokenized_path = _load_concat_dataset(cache_dir, sources, parent_id)
    elif dataset_type == 'multinomial':
        sources = dataset_config.sources
        alpha = dataset_config.get('alpha')
        total_samples = dataset_config.total_samples
        untokenized_path = _load_multinomial_dataset(
            cache_dir, sources, alpha, total_samples, dev_size,
        )
    elif dataset_type == 'instruction_jsonl':
        file_path = dataset_config.path
        untokenized_path = _load_instruction_jsonl_dataset(cache_dir, file_path)
    elif dataset_type == 'instruction_hf':
        name = dataset_config.name
        config = getattr(dataset_config, 'config', None)
        split = getattr(dataset_config, 'split', 'train')
        messages_column = getattr(dataset_config, 'messages_column', 'messages')
        prompt_template = getattr(dataset_config, 'prompt_template', '{user} Response:')
        response_template = getattr(dataset_config, 'response_template', ' {assistant}')
        max_samples = getattr(dataset_config, 'max_samples', None)
        untokenized_path = _load_instruction_hf_dataset(
            cache_dir,
            name,
            config,
            split,
            messages_column,
            prompt_template,
            response_template,
            max_samples,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    substitutions = _parse_substitutions(getattr(dataset_config, 'substitutions', None))
    if substitutions:
        untokenized_path = _apply_substitutions(untokenized_path, substitutions)

    return untokenized_path


class UntokenizedDataset(CachedArtifact):
    """The raw, untokenized corpus stage, as a tracked pipeline artifact.

    Wraps `load_untokenized_dataset` so that the config record beside the cache
    is written and validated by the same object that resolves the path, instead
    of by ~35 lines of glue in `__main__`.

    The value of this stage is a *path*, not a dataset: downstream tokenization
    reads it from disk, and for multinomial mixes it is a mix-keyed subfolder
    resolved by `DatasetConfig.effective_cache_dir`. Substitutions, when
    configured, move it again to a `_sub_{digest}` sibling.

    `build` and `read` therefore both delegate to the dispatcher, and `write` is
    a no-op. That is not an accident of the port: the dispatcher already owns a
    second, per-source caching mechanism (`_save_source_cache_config` /
    `_validate_source_cache`) that predates `CachedArtifact` and has not been
    ported yet. Until it is, the dispatcher is the only thing that can decide
    which of those inner caches are valid and what the resulting path is, so
    this class tracks the outer stage only. Porting the inner sources is what
    will let `build` and `read` become genuinely different operations.
    """

    name = "untokenized"

    # Transitional. Sources own `config.yaml` in this directory now, so this
    # record needs a name of its own to avoid overwriting theirs. It still
    # earns its place: `multinomial_mix_slug` keys the mix directory on alpha,
    # total_samples, dev_size and the per-source overrides, but not on `seed`,
    # which does change the sampled mix. Retire this together with the record
    # once MultinomialDataset tracks `seed` itself.
    config_filename = "dataset_config.yaml"

    def __init__(self, args: DictConfig):
        """Initialize from the full Hydra config.

        Args:
            args: Full Hydra configuration, read for `dataset` and `seed`.
        """
        self.args = args
        self._dataset_config = DatasetConfig.from_args(args)
        super().__init__(self._dataset_config.effective_cache_dir(args.dataset.cache_dir))

    def config(self) -> dict:
        """Return the tracked parameters for this dataset (see `DatasetConfig`)."""
        return self._dataset_config.to_dict()

    def artifact_config(self) -> DatasetConfig:
        """Use `DatasetConfig` itself, so mismatch messages keep their name."""
        return self._dataset_config

    def _dispatch(self) -> str:
        """Run the cache-aware loader and return the resulting path."""
        return load_untokenized_dataset(
            dataset_config=self.args.dataset,
            cache_dir=self.args.dataset.cache_dir,
            dev_size=resolve_dev_size(self.args),
        )

    def build(self, deps) -> str:
        return self._dispatch()

    def read(self, path: str) -> str:
        return self._dispatch()

    def write(self, value: str, path: str) -> None:
        """No-op: the dispatcher writes its own output."""


def _load_oscar_dataset(cache_dir: str, language_code: str) -> str:
    """
    Load or download OSCAR dataset for a specific language.

    Thin path-returning wrapper over `OscarDataset`; see `_load_plaintext_dataset`.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        language_code: Two-letter language code for OSCAR corpus

    Returns:
        Path to the untokenized dataset
    """
    source = OscarDataset(cache_dir, language_code)
    source.resolve()
    return source.path


def _load_huggingface_dataset(
    cache_dir: str,
    name: str,
    config: str = None,
    split: str = 'train',
    text_column: str = 'text',
    max_samples: int = None,
    min_words_per_line: int = None,
    oversampling_factor: int = 3,
    split_into_lines: bool = True,
    seed: int = 1,
) -> str:
    """
    Load a generic HuggingFace dataset.

    Thin path-returning wrapper over `HuggingFaceDataset`; see
    `_load_plaintext_dataset`.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        name: HuggingFace dataset name (e.g., 'wikitext', 'c4')
        config: Dataset configuration/subset (e.g., 'wikitext-103-v1'), optional
        split: Which split to load (default: 'train')
        text_column: Name of the column containing text (default: 'text')
        max_samples: Maximum number of examples to load, uses streaming if specified
        min_words_per_line: Minimum number of space-separated words per example
        oversampling_factor: Download this many times more documents than estimated
            needed, to maintain document diversity (default: 3)
        split_into_lines: Split each document into one example per line (default: True)
        seed: Seed for the subsample taken when max_samples is set

    Returns:
        Path to the untokenized dataset
    """
    source = HuggingFaceDataset(
        cache_dir,
        name,
        config=config,
        split=split,
        text_column=text_column,
        max_samples=max_samples,
        min_words_per_line=min_words_per_line,
        oversampling_factor=oversampling_factor,
        split_into_lines=split_into_lines,
        seed=seed,
    )
    source.resolve()
    return source.path


def _load_plaintext_dataset(cache_dir: str, file_path: str) -> str:
    """
    Load plaintext file(s) and convert to dataset format.

    Thin path-returning wrapper over `PlaintextDataset`, which owns the cache
    path, the config record, and the validate-or-build decision. Kept so the
    dispatcher and the composite loaders can keep exchanging paths while the
    remaining source types are converted.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        file_path: Path to plaintext file (one line per training example)

    Returns:
        Path to the untokenized dataset
    """
    source = PlaintextDataset(cache_dir, file_path)
    source.resolve()
    return source.path


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


def _load_instruction_jsonl_dataset(cache_dir: str, file_path: str) -> str:
    """
    Load instruction-tuning data from JSONL file(s).

    Thin path-returning wrapper over `InstructionJsonlDataset`; see
    `_load_plaintext_dataset`.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        file_path: Path to JSONL file

    Returns:
        Path to the untokenized dataset (with 'prompt' and 'response' columns)
    """
    source = InstructionJsonlDataset(cache_dir, file_path)
    source.resolve()
    return source.path


def _load_instruction_hf_dataset(
    cache_dir: str,
    name: str,
    config: str | None = None,
    split: str = 'train',
    messages_column: str = 'messages',
    prompt_template: str = '{user} Response:',
    response_template: str = ' {assistant}',
    max_samples: int | None = None,
) -> str:
    """
    Load an instruction-tuning dataset from HuggingFace.

    Expects a column of chat-formatted messages, where each entry is a list of
    {role, content} dicts (the OpenAI Chat Completions convention used by
    no_robots, Tulu, UltraChat, OpenHermes, etc.). Only examples shaped as
    exactly [user, assistant] are kept; multi-turn and system-prompt examples
    are dropped to match the framework's single-turn prompt/response schema.

    Each kept example is flattened to {prompt, response} columns by applying
    prompt_template and response_template, so the downstream tokenizer can
    consume it identically to instruction_jsonl sources.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        name: HuggingFace dataset name (e.g., 'HuggingFaceH4/no_robots')
        config: Dataset configuration/subset, optional
        split: Which split to load (default: 'train')
        messages_column: Column name holding the list of {role, content} dicts
        prompt_template: Format string with a {user} placeholder, applied to
            the user message to produce the 'prompt' column
        response_template: Format string with an {assistant} placeholder,
            applied to the assistant message to produce the 'response' column
        max_samples: Optional cap on number of examples (random subsample)

    Returns:
        Path to the untokenized dataset (with 'prompt' and 'response' columns)
    """
    untokenized_path = os.path.join(cache_dir, "untokenized")
    tracked = {
        'type': 'instruction_hf',
        'name': name,
        'config': config,
        'split': split,
        'messages_column': messages_column,
        'prompt_template': prompt_template,
        'response_template': response_template,
        'max_samples': max_samples,
    }

    if os.path.exists(untokenized_path):
        _validate_source_cache(untokenized_path, tracked)
        return untokenized_path

    print(f"Downloading instruction dataset from HuggingFace: {name}", file=sys.stderr)
    if config:
        print(f"  Config: {config}", file=sys.stderr)
    print(f"  Split: {split}", file=sys.stderr)

    raw = load_dataset(name, config, split=split)

    def is_simple_pair(ex):
        msgs = ex[messages_column]
        return (
            len(msgs) == 2
            and msgs[0]['role'] == 'user'
            and msgs[1]['role'] == 'assistant'
        )

    n_before = len(raw)
    raw = raw.filter(is_simple_pair)
    n_after = len(raw)
    print(
        f"  Kept {n_after}/{n_before} examples after filtering to single-turn "
        f"[user, assistant] pairs",
        file=sys.stderr,
    )

    if n_after == 0:
        raise ValueError(
            f"No examples remained after filtering. Check that '{messages_column}' "
            f"contains [{{'role': 'user', ...}}, {{'role': 'assistant', ...}}] pairs."
        )

    if max_samples is not None and n_after > max_samples:
        print(
            f"  Subsampling to {max_samples} examples from {n_after}",
            file=sys.stderr,
        )
        indices = random.sample(range(n_after), max_samples)
        raw = raw.select(sorted(indices))

    def to_prompt_response(ex):
        user = ex[messages_column][0]['content']
        assistant = ex[messages_column][1]['content']
        return {
            'prompt': prompt_template.format(user=user),
            'response': response_template.format(assistant=assistant),
        }

    dataset = raw.map(to_prompt_response, remove_columns=raw.column_names)
    dataset_dict = DatasetDict({'train': dataset})
    dataset_dict.save_to_disk(untokenized_path)
    _save_source_cache_config(untokenized_path, tracked)
    print(
        f"Untokenized instruction dataset saved to {untokenized_path}",
        file=sys.stderr,
    )

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

    normalized_sources = [
        OmegaConf.to_container(DictConfig(s), resolve=True) for s in sources
    ]
    tracked = {'type': 'concat', 'sources': normalized_sources}

    if os.path.exists(untokenized_path):
        _validate_source_cache(untokenized_path, tracked)
        return untokenized_path

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
    _save_source_cache_config(untokenized_path, tracked)

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
    if alpha is not None and alpha <= 0:
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

    # Place the upsampled output inside a mix-keyed subdirectory so that
    # changing alpha / total_samples / sampling_prob / upsampling_factor does
    # not clobber the previous mix. Source subdirs stay at the parent level
    # (untouched below) so they remain shared across mixes.
    mix_config = {
        'alpha': alpha,
        'total_samples': total_samples,
        'dev_size': dev_size,
        'sources': [
            OmegaConf.to_container(DictConfig(s), resolve=True) for s in sources
        ],
    }
    mix_dir = os.path.join(cache_dir, multinomial_mix_slug(mix_config))
    os.makedirs(mix_dir, exist_ok=True)
    untokenized_path = os.path.join(mix_dir, "untokenized")

    if not os.path.exists(untokenized_path):
        print(
            f"Multinomial sampling from {len(sources)} sources with alpha={alpha}", file=sys.stderr
        )
        print(f"Mix cache directory: {mix_dir}", file=sys.stderr)
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

        # Calculate sampling probabilities for TRAIN data
        # Sources with explicit sampling_prob get their probability pinned directly.
        # Remaining probability budget is distributed among unpinned sources using
        # alpha-based reweighting: p_i = (size_i)^alpha / Z, scaled to fill the budget.
        sampling_probs = _compute_sampling_probs(sources, train_sizes, alpha)

        # Convert probabilities to integer sample counts
        # Distribute remainder samples round-robin to handle rounding errors
        samples_per_source = [int(prob * total_samples) for prob in sampling_probs]
        remaining = total_samples - sum(samples_per_source)
        for i in range(remaining):
            samples_per_source[i % len(sources)] += 1

        print("Train sampling distribution:", file=sys.stderr)
        for idx, count in enumerate(samples_per_source):
            percentage = 100 * count / total_samples
            source_id = _get_source_id(DictConfig(sources[idx]), fallback=f"source_{idx}")
            pinned = sources[idx].get('sampling_prob') is not None
            pin_marker = " (pinned)" if pinned else ""
            print(
                f"  {source_id}: {count} samples ({percentage:.2f}%){pin_marker}",
                file=sys.stderr,
            )

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


def _compute_sampling_probs(
    sources: list[dict],
    train_sizes: list[int],
    alpha: float | None,
) -> list[float]:
    """
    Compute per-source sampling probabilities, respecting pinned sampling_prob values.

    Sources with an explicit `sampling_prob` or `upsampling_factor field get that probability
    directly. The remaining probability budget is distributed among unpinned sources using
    alpha-based temperature scaling: p_i = (size_i)^alpha / Z, scaled to fill the budget.

    Alpha may be None when it has nothing to do: when every source is pinned, or when
    exactly one source is unpinned and therefore takes the whole remaining budget
    regardless of the exponent. It is required whenever two or more sources are unpinned.

    Args:
        sources: List of source config dicts (may contain 'sampling_prob' field)
        train_sizes: Number of training examples per source (after dev split)
        alpha: Temperature parameter for unpinned source reweighting. May be None only
            if it cannot affect the result.

    Returns:
        List of sampling probabilities (one per source, sums to 1.0)
    """
    num_sources = len(sources)
    total_size = sum(train_sizes)
    pinned_probs = {}
    for idx, source in enumerate(sources):
        prob = source.get('sampling_prob')
        upsampling_factor = source.get('upsampling_factor')
        if upsampling_factor is not None and prob is None:
            pinned_probs[idx] = train_sizes[idx] * upsampling_factor / total_size
        if prob is not None:
            if prob <= 0 or prob >= 1.0:
                source_id = _get_source_id(DictConfig(source), fallback=f"source_{idx}")
                raise ValueError(
                    f"Source '{source_id}': sampling_prob must be between 0 and 1 exclusive, "
                    f"got {prob}"
                )
            pinned_probs[idx] = prob

    pinned_total = sum(pinned_probs.values())

    # If every source is pinned, they must sum to exactly 1.0
    if len(pinned_probs) == num_sources:
        if abs(pinned_total - 1.0) > 1e-9:
            raise ValueError(
                f"All sources have sampling_prob but they sum to {pinned_total:.6f}, not 1.0"
            )
        return [pinned_probs[i] for i in range(num_sources)]

    # With unpinned sources present, pinned probs must leave room for them
    if pinned_total >= 1.0:
        raise ValueError(
            f"Sum of pinned sampling_prob values is {pinned_total:.4f}, "
            "must be less than 1.0 to leave budget for remaining sources"
        )

    # Distribute remaining budget among unpinned sources using alpha-based weighting
    remaining_budget = 1.0 - pinned_total
    unpinned_indices = [i for i in range(num_sources) if i not in pinned_probs]

    unpinned_sizes = [train_sizes[i] for i in unpinned_indices]
    if all(s == 0 for s in unpinned_sizes):
        raise ValueError("Cannot compute sampling probabilities: all unpinned sources are empty")

    # a lone unpinned source takes the whole remaining budget: its weight normalizes
    # to 1.0 for any exponent, so alpha is not needed to resolve the mixture
    if len(unpinned_indices) == 1:
        lone_probs = dict(pinned_probs)
        lone_probs[unpinned_indices[0]] = remaining_budget
        return [lone_probs[i] for i in range(num_sources)]

    if alpha is None:
        unpinned_ids = [
            _get_source_id(DictConfig(sources[i]), fallback=f"source_{i}")
            for i in unpinned_indices
        ]
        raise ValueError(
            f"alpha is required when two or more sources are unpinned "
            f"({', '.join(unpinned_ids)}): it sets how the remaining probability "
            "budget is split between them"
        )

    weights = [size ** alpha for size in unpinned_sizes]
    total_weight = sum(weights)
    unpinned_probs = {
        idx: (weights[j] / total_weight) * remaining_budget
        for j, idx in enumerate(unpinned_indices)
    }

    return [pinned_probs.get(i, unpinned_probs.get(i)) for i in range(num_sources)]


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


def _tokenize_plaintext_with_labels(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> dict:
    """
    Tokenize plaintext examples and add labels for causal LM loss.

    Used for plaintext splits in mixed instruction/plaintext datasets, where the
    DataCollatorForInstructionTuning expects all examples to have 'labels'.
    For plaintext, labels = input_ids (loss on all tokens).

    Args:
        examples: Batch with 'text' field
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels' fields
    """
    tokenized = tokenizer(
        examples['text'], max_length=max_length, truncation=True
    )
    # For plaintext, labels = input_ids (standard causal LM loss on all tokens)
    tokenized['labels'] = [ids.copy() for ids in tokenized['input_ids']]
    return tokenized


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

            # Append EOS so the model learns to terminate responses
            response_ids = response_tokens['input_ids'] + [tokenizer.eos_token_id]
            response_mask = response_tokens['attention_mask'] + [1]

            # Concatenate
            # TODO: fix linting issue here
            input_ids = prompt_tokens['input_ids'] + response_ids
            attention_mask = prompt_tokens['attention_mask'] + response_mask

            # Create labels: -100 for prompt (masked), actual tokens for response
            prompt_length = len(prompt_tokens['input_ids'])
            labels = [-100] * prompt_length + response_ids
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


# Plan / per-source tokenization for multinomial mixes.
# The training-time multinomial pipeline upsamples by repeating row indices
# rather than duplicating tokenized rows. The pieces below implement that:
#
#   <cache_dir>/<source_id>/untokenized/                      (text, mix-agnostic)
#   <cache_dir>/<source_id>/tokenized_<tok>_ml<L>_{labels,nolabels}/  (mix-agnostic)
#   <mix_dir>/train_plan.npz                                  (shuffled global indices)
#   <mix_dir>/dev/                                            (per-source tokenized dev)
#
# The training Dataset is built as
# `concatenate_datasets(per_source_tokenized).select(global_indices)`,
# which produces an Arrow indices-mapped view; no rows are duplicated.

TOKENIZED_SOURCE_CONFIG_FILENAME = "tokenized_config.yaml"
TRAIN_PLAN_FILENAME = "train_plan.npz"
DEV_SUBDIR = "dev"


def _tokenized_source_dirname(
    tokenizer_id: str,
    max_length: int,
    add_labels: bool,
    variant_suffix: str = "",
) -> str:
    """
    Build the per-source tokenized cache directory name.

    The optional ``variant_suffix`` carries the untokenized variant (e.g.
    ``_sub_<hash>`` for a substituted source) so a substituted source tokenizes
    into a distinct cache rather than colliding with the raw one. An empty
    suffix reproduces the original ``tokenized_<id>_ml<L>_<labels>`` name, so
    pre-existing raw caches stay valid.
    """
    label_suffix = "labels" if add_labels else "nolabels"
    return f"tokenized{variant_suffix}_{tokenizer_id}_ml{max_length}_{label_suffix}"


def _dev_splits_dirname(
    tokenizer_id: str,
    max_length: int,
    add_labels: bool,
) -> str:
    """
    Build the mix-level tokenized dev-splits cache directory name.

    The dev splits hold token ids, so the cache must be keyed by the same
    parameters as the per-source tokenized caches. The mix slug that names the
    parent directory deliberately excludes the tokenizer (sources and plans are
    shared across tokenizers), so without this suffix a dev cache written by one
    model's tokenizer would be silently reused by a model with a different
    tokenizer.
    """
    label_suffix = "labels" if add_labels else "nolabels"
    return f"{DEV_SUBDIR}_{tokenizer_id}_ml{max_length}_{label_suffix}"


def _source_has_instruction_columns(untokenized_path: str) -> bool:
    """Return True if the source's untokenized 'train' split has prompt/response columns."""
    data = load_from_disk(untokenized_path)
    if isinstance(data, DatasetDict):
        split = data['train'] if 'train' in data else data[list(data.keys())[0]]
    else:
        split = data
    cols = split.column_names
    return 'prompt' in cols and 'response' in cols


def tokenize_source(
    untokenized_path: str,
    tokenizer: PreTrainedTokenizer,
    tokenizer_id: str,
    max_length: int,
    add_labels: bool,
) -> str:
    """
    Tokenize a single source's untokenized rows once and cache them.

    Detects instruction vs plaintext from the source's column schema:
    - Instruction sources (prompt/response columns) always produce masked labels.
    - Plaintext sources produce labels=input_ids when add_labels is True, otherwise
      only input_ids and attention_mask.

    The tokenized cache holds ALL rows of the source as a single (non-split) Dataset.
    Train/dev partitioning is a mix-time concern and does not affect this cache.

    Args:
        untokenized_path: Path to the source's untokenized dataset, as returned by
            load_untokenized_dataset. This is the raw 'untokenized/' dir for an
            unmodified source, or an 'untokenized_sub_<hash>/' sibling when the
            source declares regex substitutions; the tokenized output is written
            as a sibling whose name encodes that variant.
        tokenizer: Tokenizer to use.
        tokenizer_id: Stable identifier embedded in the cache directory name.
        max_length: Truncation length for tokenization.
        add_labels: Whether to materialize a labels column for plaintext sources.
            Required when the source will be mixed alongside an instruction source so
            that the data collator sees a uniform schema.

    Returns:
        Path to the tokenized cache directory.
    """
    if not os.path.exists(untokenized_path):
        raise FileNotFoundError(
            f"Source untokenized cache missing at {untokenized_path}; "
            "load the source via load_untokenized_dataset first."
        )

    source_cache_dir = os.path.dirname(untokenized_path)
    # Derive the variant suffix from the untokenized dir name so substituted
    # sources ('untokenized_sub_<hash>') tokenize into their own cache and don't
    # collide with the raw 'untokenized' cache.
    untokenized_name = os.path.basename(untokenized_path)
    variant_suffix = untokenized_name[len("untokenized"):]
    dirname = _tokenized_source_dirname(
        tokenizer_id, max_length, add_labels, variant_suffix
    )
    tokenized_path = os.path.join(source_cache_dir, dirname)

    tracked = {
        'tokenizer_id': tokenizer_id,
        'max_length': max_length,
        'add_labels': add_labels,
    }
    if os.path.exists(tokenized_path):
        _validate_tokenized_source_cache(tokenized_path, tracked)
        return tokenized_path

    print(
        f"Tokenizing source at {source_cache_dir} -> {dirname} "
        f"(tokenizer={tokenizer_id}, max_length={max_length}, add_labels={add_labels})",
        file=sys.stderr,
    )

    ds = load_from_disk(untokenized_path)
    # Per-source untokenized caches are stored as DatasetDict with a single 'train' split.
    if isinstance(ds, DatasetDict):
        if 'train' not in ds:
            raise ValueError(
                f"Source untokenized cache at {untokenized_path} has no 'train' split "
                f"(found splits: {list(ds.keys())})"
            )
        data = ds['train']
    else:
        data = ds

    cols = data.column_names
    has_instruction = 'prompt' in cols and 'response' in cols
    has_text = 'text' in cols

    if has_instruction:
        cols_to_remove = ['prompt', 'response'] + (['text'] if has_text else [])
        tokenized = data.map(
            lambda examples: _tokenize_instruction_examples(
                examples, tokenizer, max_length
            ),
            batched=True,
            remove_columns=cols_to_remove,
        )
    elif has_text:
        if add_labels:
            tokenized = data.map(
                lambda examples: _tokenize_plaintext_with_labels(
                    examples, tokenizer, max_length
                ),
                batched=True,
                remove_columns='text',
            )
        else:
            tokenized = data.map(
                lambda examples: tokenizer(
                    examples['text'], max_length=max_length, truncation=True
                ),
                batched=True,
                remove_columns='text',
            )
    else:
        raise ValueError(
            f"Source at {untokenized_path} has neither instruction columns "
            f"(prompt/response) nor a 'text' column. Found: {cols}"
        )

    tokenized.save_to_disk(tokenized_path)
    _save_tokenized_source_cache_config(tokenized_path, tracked)
    print(
        f"  Tokenized {len(tokenized)} rows -> {tokenized_path}",
        file=sys.stderr,
    )
    return tokenized_path


def _validate_tokenized_source_cache(tokenized_path: str, current: dict) -> None:
    """Validate that a cached per-source tokenized dataset matches current params."""
    config_path = os.path.join(tokenized_path, TOKENIZED_SOURCE_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        print(
            f"Note: tokenized source cache at {tokenized_path} has no config "
            "tracking file; assuming it matches current parameters.",
            file=sys.stderr,
        )
        return
    with open(config_path) as f:
        cached = yaml.safe_load(f) or {}
    diffs = dict_diff(cached, current)
    if not diffs:
        return
    raise ValueError(
        f"\nTOKENIZED SOURCE CACHE MISMATCH: {tokenized_path}\n"
        + "\n".join(f"  {diff}" for diff in diffs)
        + "\nRemove this cache directory or change cache parameters to resolve."
    )


def _save_tokenized_source_cache_config(tokenized_path: str, current: dict) -> None:
    """Write the per-source tokenized cache config file."""
    config_path = os.path.join(tokenized_path, TOKENIZED_SOURCE_CONFIG_FILENAME)
    with open(config_path, 'w') as f:
        yaml.dump(current, f, default_flow_style=False, sort_keys=False)


def _partition_source_indices(
    num_rows: int,
    dev_size: float,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the train/dev row partition for a single source.

    Uses a numpy RNG seeded deterministically so the partition is reproducible
    and independent of HuggingFace's internal split implementation.

    Args:
        num_rows: Size of the source's tokenized dataset.
        dev_size: -1 to skip dev split, fractional (0 < x < 1) for proportion,
            or absolute (>= 1) for an explicit row count.
        seed: RNG seed for the permutation.

    Returns:
        (train_indices, dev_indices) as uint32 numpy arrays. dev_indices is empty
        when dev_size == -1.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_rows).astype(np.uint32)
    if dev_size == -1:
        return perm, np.empty(0, dtype=np.uint32)
    if dev_size >= 1:
        dev_count = int(dev_size)
    elif 0 < dev_size < 1:
        dev_count = int(round(dev_size * num_rows))
    else:
        raise ValueError(f"Invalid dev_size {dev_size}")
    dev_count = min(dev_count, num_rows)
    dev_indices = perm[:dev_count]
    train_indices = perm[dev_count:]
    return train_indices, dev_indices


def load_tokenized_multinomial_dataset(
    sources: list,
    alpha: float,
    total_samples: int,
    dev_size: float,
    base_cache_dir: str,
    tokenizer: PreTrainedTokenizer,
    tokenizer_id: str,
    max_length: int,
    shuffle_seed: int = 1,
) -> DatasetDict:
    """
    Build a tokenized multinomial mix without materializing the upsampled train set.

    Each source's rows are tokenized exactly once and stored in a mix-agnostic
    cache. The training mix is represented as a tiny plan file of shuffled global
    indices into the concatenation of per-source tokenized datasets; at load time
    the train split is realized as an Arrow indices-mapped view (no row duplication).
    Per-source dev splits are tokenized once and persisted under the mix directory.

    Args:
        sources: List of source configuration dicts (same schema as the existing
            multinomial path; supports per-source dev_size, sampling_prob, and
            upsampling_factor overrides).
        alpha: Temperature parameter for sampling among unpinned sources.
        total_samples: Total training rows to sample (with repetition for upsampling).
        dev_size: Global default dev split fraction (-1 to skip globally).
        base_cache_dir: Parent cache directory holding per-source subdirs.
        tokenizer: Tokenizer used to materialize the tokenized caches.
        tokenizer_id: Stable identifier for the tokenizer (drives cache directory naming).
        max_length: Truncation length.
        shuffle_seed: Seed for shuffling the global training plan.

    Returns:
        DatasetDict with:
            - 'train': indices-mapped virtual view of the upsampled training mix.
            - One entry per source-with-dev (keyed by source id), holding the
              tokenized dev split for that source.
    """
    if not sources:
        raise ValueError("Cannot sample from datasets: sources list is empty")
    if total_samples <= 0:
        raise ValueError(f"total_samples must be positive, got {total_samples}")
    if alpha is not None and alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if dev_size is None:
        raise ValueError("dev_size must be provided")

    mix_config = {
        'alpha': alpha,
        'total_samples': total_samples,
        'dev_size': dev_size,
        'sources': [
            OmegaConf.to_container(DictConfig(s), resolve=True) for s in sources
        ],
    }
    mix_dir = os.path.join(base_cache_dir, multinomial_mix_slug(mix_config))
    os.makedirs(mix_dir, exist_ok=True)

    # Step 1: ensure per-source untokenized caches exist.
    source_ids = []
    source_untokenized_paths = []
    source_dev_sizes = []
    for idx, source_config in enumerate(sources):
        source_dict = DictConfig(source_config)
        source_id = _get_source_id(source_dict, fallback=f"source_{idx}")
        source_cache = os.path.join(base_cache_dir, source_id)
        untokenized_path = load_untokenized_dataset(
            dataset_config=source_dict,
            cache_dir=source_cache,
        )
        source_ids.append(source_id)
        source_untokenized_paths.append(untokenized_path)
        per_source_dev_size = getattr(source_dict, 'dev_size', dev_size)
        source_dev_sizes.append(per_source_dev_size)

    # Step 2: decide labels schema. Any instruction source forces add_labels for all.
    any_instruction = any(
        _source_has_instruction_columns(p) for p in source_untokenized_paths
    )
    add_labels = any_instruction
    if any_instruction:
        print(
            "Mix contains instruction data; tokenizing all sources with labels.",
            file=sys.stderr,
        )

    # Step 3: tokenize each source (cache-aware). Tokenize the exact untokenized
    # path returned above, which is the substituted variant when a source declares
    # regex substitutions (not the raw 'untokenized' dir).
    source_tokenized_paths = [
        tokenize_source(
            untokenized_path=untokenized_path,
            tokenizer=tokenizer,
            tokenizer_id=tokenizer_id,
            max_length=max_length,
            add_labels=add_labels,
        )
        for untokenized_path in source_untokenized_paths
    ]
    per_source_tokenized = [load_from_disk(p) for p in source_tokenized_paths]
    source_sizes = [len(d) for d in per_source_tokenized]

    # Step 4: partition each source's rows into train pool and dev rows.
    train_pools = []
    dev_indices_per_source = []
    for size, src_dev in zip(source_sizes, source_dev_sizes):
        train_idx, dev_idx = _partition_source_indices(size, src_dev, seed=1)
        train_pools.append(train_idx)
        dev_indices_per_source.append(dev_idx)

    # Step 5: compute per-source training sample counts using existing logic.
    train_pool_sizes = [len(p) for p in train_pools]
    if all(s == 0 for s in train_pool_sizes):
        raise ValueError("Cannot sample: every source has an empty train pool")
    sampling_probs = _compute_sampling_probs(sources, train_pool_sizes, alpha)
    samples_per_source = [int(p * total_samples) for p in sampling_probs]
    remaining = total_samples - sum(samples_per_source)
    for i in range(remaining):
        samples_per_source[i % len(sources)] += 1

    print("Train sampling distribution (plan-based):", file=sys.stderr)
    for idx, count in enumerate(samples_per_source):
        pct = 100 * count / total_samples
        pinned = sources[idx].get('sampling_prob') is not None
        marker = " (pinned)" if pinned else ""
        print(
            f"  {source_ids[idx]}: {count} samples ({pct:.2f}%){marker}",
            file=sys.stderr,
        )

    # Step 6: build the plan if missing.
    plan_path = os.path.join(mix_dir, TRAIN_PLAN_FILENAME)
    offsets = np.zeros(len(sources) + 1, dtype=np.int64)
    for i, sz in enumerate(source_sizes):
        offsets[i + 1] = offsets[i] + sz

    if not os.path.exists(plan_path):
        rng = np.random.default_rng(shuffle_seed)
        global_indices_chunks = []
        for src_idx, (pool, n_samples) in enumerate(
            zip(train_pools, samples_per_source)
        ):
            if n_samples == 0:
                continue
            pool_size = len(pool)
            if n_samples <= pool_size:
                # Sample without replacement within the source's train pool.
                local_indices = rng.choice(pool_size, size=n_samples, replace=False)
            else:
                # Exhaust-first: every pool row at least once, then sample remainder.
                extra = rng.integers(0, pool_size, size=n_samples - pool_size)
                local_indices = np.concatenate(
                    [np.arange(pool_size, dtype=np.int64), extra]
                )
            global_chunk = pool[local_indices].astype(np.int64) + offsets[src_idx]
            global_indices_chunks.append(global_chunk)

        global_indices = np.concatenate(global_indices_chunks)
        rng.shuffle(global_indices)
        np.savez(
            plan_path,
            global_indices=global_indices.astype(np.int64),
            source_ids=np.array(source_ids, dtype=object),
            source_sizes=np.array(source_sizes, dtype=np.int64),
        )
        print(
            f"Saved train plan ({len(global_indices)} samples) to {plan_path}",
            file=sys.stderr,
        )
    else:
        print(f"Loading existing train plan from {plan_path}", file=sys.stderr)

    plan = np.load(plan_path, allow_pickle=True)
    cached_source_ids = list(plan['source_ids'])
    cached_source_sizes = list(plan['source_sizes'])
    if cached_source_ids != source_ids or cached_source_sizes != source_sizes:
        raise ValueError(
            f"Train plan at {plan_path} is inconsistent with current per-source "
            f"tokenized caches (source ids or row counts changed). Delete the plan "
            f"file and re-run to rebuild."
        )
    global_indices = plan['global_indices']

    # Step 7: build or load the dev DatasetDict. The dev cache is keyed by
    # tokenizer/max_length/labels because it stores token ids, unlike the
    # tokenizer-agnostic mix slug naming the parent directory.
    dev_path = os.path.join(
        mix_dir, _dev_splits_dirname(tokenizer_id, max_length, add_labels)
    )
    legacy_dev_path = os.path.join(mix_dir, DEV_SUBDIR)
    if os.path.exists(legacy_dev_path):
        print(
            f"Warning: ignoring legacy dev cache at {legacy_dev_path}; it was written "
            f"without a tokenizer key and may hold token ids from a different "
            f"tokenizer. Delete it once no runs depend on it.",
            file=sys.stderr,
        )
    if not os.path.exists(dev_path):
        dev_dict = {}
        for src_id, ds, dev_idx in zip(
            source_ids, per_source_tokenized, dev_indices_per_source
        ):
            if len(dev_idx) == 0:
                continue
            dev_dict[src_id] = ds.select(dev_idx.tolist()).flatten_indices()
        if dev_dict:
            DatasetDict(dev_dict).save_to_disk(dev_path)
            print(
                f"Saved tokenized dev splits ({list(dev_dict.keys())}) to {dev_path}",
                file=sys.stderr,
            )

    dev_dict = (
        load_from_disk(dev_path) if os.path.exists(dev_path) else DatasetDict()
    )

    # Step 8: assemble the virtual training view. concatenate + select store an
    # Arrow indices map; no row duplication occurs on disk or in memory.
    concat = concatenate_datasets(per_source_tokenized)
    train = concat.select(global_indices.tolist())

    result = {'train': train}
    for k, v in dev_dict.items():
        result[k] = v
    return DatasetDict(result)


def load_tokenized_dataset(
    untokenized_path: str,
    tokenized_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    dev_size: float,
) -> DatasetDict:
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
        Dataset dictionary with 'train' and dev splits. Simple datasets give
        ``{'train': ..., 'test': ...}``; multinomial datasets give one dev split
        per source, e.g. ``{'train': ..., 'got': ..., 'ang': ..., 'non': ...}``;
        instruction datasets use the same structure with a 'labels' field added
        for loss masking.

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
        # Always check 'train' split since multinomial datasets may have per-source dev splits
        # with different column schemas (e.g., 'eng' dev has only 'text', 'train' has mixed)
        sample_split = 'train' if 'train' in dataset else list(dataset.keys())[0]
        has_instruction_data = (
            'prompt' in dataset[sample_split].column_names
            and 'response' in dataset[sample_split].column_names
        )

        if has_instruction_data:
            print(
                "Detected instruction dataset format, tokenizing with label masking",
                file=sys.stderr
            )

        # Process each split individually since they may have different column schemas
        # (e.g., multinomial datasets with per-source dev splits)
        tokenized_splits = {}
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            split_columns = split_data.column_names

            # Check what type of data this split has
            split_has_instruction = 'prompt' in split_columns and 'response' in split_columns
            split_has_text = 'text' in split_columns

            if split_has_instruction:
                # Instruction data (possibly mixed with plaintext)
                columns_to_remove = ['prompt', 'response']
                if split_has_text:
                    columns_to_remove.append('text')
                tokenized_splits[split_name] = split_data.map(
                    lambda examples: _tokenize_instruction_examples(
                        examples, tokenizer, max_length
                    ),
                    batched=True,
                    remove_columns=columns_to_remove
                )
            elif split_has_text:
                # Standard plaintext data
                # If the overall dataset has instruction data, add labels for collator compatibility
                if has_instruction_data:
                    tokenized_splits[split_name] = split_data.map(
                        lambda examples: _tokenize_plaintext_with_labels(
                            examples, tokenizer, max_length
                        ),
                        batched=True,
                        remove_columns='text'
                    )
                else:
                    tokenized_splits[split_name] = split_data.map(
                        lambda examples: tokenizer(
                            examples['text'], max_length=max_length, truncation=True
                        ),
                        batched=True,
                        remove_columns='text'
                    )
            else:
                raise ValueError(
                    f"Split '{split_name}' has neither instruction columns (prompt/response) "
                    f"nor text column. Found columns: {split_columns}"
                )

        dataset = DatasetDict(tokenized_splits)

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
    max_length: int,
    add_labels: bool = False,
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
        add_labels: If True, add a 'labels' column (=input_ids) for plaintext/jsonl
            formats so the set is compatible with DataCollatorForInstructionTuning.
            instruction_jsonl format always includes masked labels regardless.

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
        with open(path, encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        dataset = Dataset.from_dict({'text': lines})
        is_instruction = False

    elif file_format == 'jsonl':
        # Load JSONL file
        data = []
        with open(path, encoding='utf-8') as f:
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
        prompts, responses = read_instruction_jsonl(path)
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
        if add_labels:
            tokenize_fn = lambda examples: _tokenize_plaintext_with_labels(
                examples, tokenizer, max_length
            )
        else:
            tokenize_fn = lambda examples: tokenizer(
                examples['text'], max_length=max_length, truncation=True
            )
        dataset = dataset.map(
            tokenize_fn,
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

        # If the existing eval sets carry labels (instruction-tuning run), plaintext
        # external eval sets must also have labels to be compatible with
        # DataCollatorForInstructionTuning.
        existing_has_labels = any(
            'labels' in ds.column_names for ds in eval_dataset.values()
        )

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
                max_length=max_length,
                add_labels=existing_has_labels,
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
