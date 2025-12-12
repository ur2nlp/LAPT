"""
Utilities for saving and loading configuration metadata alongside cached artifacts.

This module provides functions to:
1. Extract relevant config subsets for each artifact type
2. Save config metadata alongside cached artifacts
3. Load and compare configs to detect cache invalidation
"""

import os
import sys
import warnings
from typing import Optional

from omegaconf import DictConfig, OmegaConf
import yaml


def extract_dataset_config(args: DictConfig) -> dict:
    """
    Extract the configuration subset that affects untokenized dataset caching.

    This should mirror all parameters used by load_untokenized_dataset() in dataset_utils.py
    that affect the dataset artifact (not just input/output paths).

    Args:
        args: Full Hydra configuration

    Returns:
        Dictionary containing only parameters that affect dataset loading
    """
    config = {
        'type': args.dataset.type,
        'seed': args.seed,
    }

    # Type-specific parameters
    if args.dataset.type == 'oscar':
        config['language'] = args.dataset.language
    elif args.dataset.type == 'plaintext':
        config['path'] = args.dataset.path
    elif args.dataset.type == 'plaintext_dir':
        config['directory'] = args.dataset.directory
        config['pattern'] = getattr(args.dataset, 'pattern', '*.txt')
    elif args.dataset.type == 'concat':
        config['sources'] = OmegaConf.to_container(args.dataset.sources, resolve=True)
    elif args.dataset.type == 'multinomial':
        config['sources'] = OmegaConf.to_container(args.dataset.sources, resolve=True)
        config['alpha'] = args.dataset.alpha
        config['total_samples'] = args.dataset.total_samples
        config['dev_size'] = args.training.dev_size

    return config


def extract_tokenizer_config(args: DictConfig) -> Optional[dict]:
    """
    Extract the configuration subset that affects tokenizer training (FOCUS only).

    This should mirror all parameters used by train_new_tokenizer() in tokenizer_utils.py
    that affect the tokenizer artifact (not just input/output paths).

    Args:
        args: Full Hydra configuration

    Returns:
        Dictionary containing only parameters that affect tokenizer training,
        or None if FOCUS is not enabled
    """
    if not args.focus.enabled:
        return None

    config = {
        'hf_model': args.hf_model,
        'vocab_size': args.focus.vocab_size,
        'num_samples': args.focus.num_samples,
        'inherit_additional_special_tokens': args.focus.get('inherit_additional_special_tokens', True),
        'character_coverage': args.focus.get('character_coverage', 1.0),
        'use_seed_vocabulary': args.focus.get('use_seed_vocabulary', False),
        'seed_min_frequency': args.focus.get('seed_min_frequency', 1),
        'seed_lambda': args.focus.get('seed_lambda', 0.5),
        'seed_round_mode': args.focus.get('seed_round_mode', 'ceil'),
        'seed_vocab_multiplier': args.focus.get('seed_vocab_multiplier', 5.0),
        'seed_target_mass': args.focus.get('seed_target_mass', 10_000_000),
        'seed': args.seed,
    }

    # Include either training dataset or separate FOCUS dataset config
    if hasattr(args.focus, 'dataset') and args.focus.dataset is not None:
        # Use separate FOCUS dataset
        focus_dataset_config = OmegaConf.to_container(args.focus.dataset, resolve=True)
        config['focus_dataset'] = focus_dataset_config
    else:
        # Use training dataset - just reference the cache_dir
        config['train_dataset_cache'] = args.dataset.cache_dir

    return config


def extract_tokenized_config(args: DictConfig) -> dict:
    """
    Extract the configuration subset that affects tokenized dataset caching.

    This should mirror all parameters used by load_or_tokenize_dataset() in dataset_utils.py
    that affect the tokenized dataset artifact, plus all upstream dependencies.

    Args:
        args: Full Hydra configuration

    Returns:
        Dictionary containing only parameters that affect tokenization
    """
    config = {
        'max_length': args.training.max_length,
        'dev_size': args.training.dev_size,
    }

    # Include untokenized dataset config
    config['dataset'] = extract_dataset_config(args)

    # Include tokenizer config if FOCUS
    tokenizer_config = extract_tokenizer_config(args)
    if tokenizer_config is not None:
        config['tokenizer'] = tokenizer_config

    return config


def extract_model_config(args: DictConfig) -> dict:
    """
    Extract the full configuration for model training.

    Args:
        args: Full Hydra configuration

    Returns:
        Dictionary containing all training configuration
    """
    # For model outputs, we save the full config
    # Hydra already does this in outputs/, but we want it with the model too
    return OmegaConf.to_container(args, resolve=True)


def save_config(config: dict, output_path: str):
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        output_path: Path where config should be saved (will create parent dirs)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(config_path: str) -> Optional[dict]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary, or None if file doesn't exist
    """
    if not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _dict_diff(dict1: dict, dict2: dict, path: str = "") -> list[str]:
    """
    Recursively find differences between two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        path: Current path in nested structure (for error messages)

    Returns:
        List of difference descriptions
    """
    diffs = []

    # Keys only in dict1
    only_in_1 = set(dict1.keys()) - set(dict2.keys())
    for key in only_in_1:
        diffs.append(f"{path}.{key}" if path else f"{key}: present in cached config but not in current")

    # Keys only in dict2
    only_in_2 = set(dict2.keys()) - set(dict1.keys())
    for key in only_in_2:
        diffs.append(f"{path}.{key}" if path else f"{key}: present in current config but not in cached")

    # Keys in both - check values
    for key in set(dict1.keys()) & set(dict2.keys()):
        val1 = dict1[key]
        val2 = dict2[key]
        current_path = f"{path}.{key}" if path else key

        if isinstance(val1, dict) and isinstance(val2, dict):
            # Recurse into nested dicts
            diffs.extend(_dict_diff(val1, val2, current_path))
        elif val1 != val2:
            diffs.append(f"{current_path}: {val1} (cached) != {val2} (current)")

    return diffs


def check_config_match(
    cached_config: Optional[dict],
    current_config: dict,
    artifact_name: str,
    error_on_mismatch: bool = True
) -> bool:
    """
    Check if cached config matches current config, raising error if different.

    Args:
        cached_config: Configuration saved with cached artifact (None if not found)
        current_config: Current configuration
        artifact_name: Name of artifact for error messages
        error_on_mismatch: Whether to raise error on mismatch (vs returning False)

    Returns:
        True if configs match, False otherwise (if error_on_mismatch=False)

    Raises:
        ValueError: If configs don't match and error_on_mismatch=True
    """
    if cached_config is None:
        # No cached config found - this is expected for old caches
        return True

    diffs = _dict_diff(cached_config, current_config)

    if diffs:
        error_msg = (
            f"\n{'=' * 70}\n"
            f"CONFIG MISMATCH: {artifact_name}\n"
            f"{'=' * 70}\n"
            f"Cached artifact was created with different parameters:\n\n"
            + "\n".join(f"  {diff}" for diff in diffs)
            + f"\n\n"
            f"The cached config file represents what actually created this artifact.\n"
            f"To proceed, either:\n\n"
            f"  1. Regenerate with current config:\n"
            f"     Add fresh_dataset=true (dataset), fresh_tokenizer=true (tokenizer),\n"
            f"     or fresh_model=true (model) to your command\n\n"
            f"  2. Update your config to match the cached version\n"
            f"{'=' * 70}\n"
        )

        if error_on_mismatch:
            raise ValueError(error_msg)
        else:
            print(error_msg, file=sys.stderr)
            return False

    return True


def save_dataset_config(args: DictConfig, cache_dir: str):
    """Save dataset config alongside untokenized dataset."""
    config = extract_dataset_config(args)
    config_path = os.path.join(cache_dir, "untokenized", "config.yaml")
    save_config(config, config_path)
    print(f"Saved dataset config to {config_path}", file=sys.stderr)


def save_tokenizer_config(args: DictConfig, tokenizer_dir: str):
    """Save tokenizer config alongside trained tokenizer (FOCUS only)."""
    config = extract_tokenizer_config(args)
    if config is None:
        return

    config_path = os.path.join(tokenizer_dir, "training_config.yaml")
    save_config(config, config_path)
    print(f"Saved tokenizer config to {config_path}", file=sys.stderr)


def save_tokenized_config(args: DictConfig, tokenized_dir: str):
    """Save tokenized dataset config."""
    config = extract_tokenized_config(args)
    config_path = os.path.join(tokenized_dir, "config.yaml")
    save_config(config, config_path)
    print(f"Saved tokenized dataset config to {config_path}", file=sys.stderr)


def save_model_config(args: DictConfig, output_dir: str):
    """Save full config alongside model checkpoints."""
    config = extract_model_config(args)
    config_path = os.path.join(output_dir, "training_config.yaml")
    save_config(config, config_path)
    print(f"Saved model training config to {config_path}", file=sys.stderr)


def check_dataset_config(args: DictConfig, cache_dir: str) -> bool:
    """Check if cached dataset config matches current config."""
    config_path = os.path.join(cache_dir, "untokenized", "config.yaml")
    cached_config = load_config(config_path)
    current_config = extract_dataset_config(args)
    return check_config_match(cached_config, current_config, "Untokenized Dataset")


def check_tokenizer_config(args: DictConfig, tokenizer_dir: str) -> bool:
    """Check if cached tokenizer config matches current config (FOCUS only)."""
    if not args.focus.enabled:
        return True

    config_path = os.path.join(tokenizer_dir, "training_config.yaml")
    cached_config = load_config(config_path)
    current_config = extract_tokenizer_config(args)
    return check_config_match(cached_config, current_config, "Tokenizer")


def check_tokenized_config(args: DictConfig, tokenized_dir: str) -> bool:
    """Check if cached tokenized dataset config matches current config."""
    config_path = os.path.join(tokenized_dir, "config.yaml")
    cached_config = load_config(config_path)
    current_config = extract_tokenized_config(args)
    return check_config_match(cached_config, current_config, "Tokenized Dataset")
