"""
Utilities for model and tokenizer initialization.

This module handles both standard model loading and FOCUS-based vocabulary
specialization workflows.
"""

import os
import random
import sys

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tokenizer_utils import (
    apply_focus_initialization,
    prepare_focus_training_data,
    train_new_tokenizer
)
from config_utils import check_tokenizer_config, save_tokenizer_config


def format_number(n: int) -> str:
    """
    Format large numbers with k/m suffix for directory names.

    Args:
        n: Number to format

    Returns:
        Formatted string (e.g., 50000 -> "50k", 1000000 -> "1m")
    """
    if n >= 1_000_000:
        return f"{n // 1_000_000}m"
    elif n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def get_model_shortname(hf_model: str) -> str:
    """
    Extract a short identifier from HuggingFace model name.

    Args:
        hf_model: Full HuggingFace model name (e.g., "facebook/xglm-564M")

    Returns:
        Short identifier (e.g., "xglm564m")
    """
    # Take the part after "/" if present, otherwise use full name
    model_name = hf_model.split('/')[-1]
    # Remove dots and dashes, lowercase, make compact
    shortname = model_name.replace('-', '').replace('.', '').lower()
    return shortname


def is_local_model_path(hf_model: str) -> bool:
    """
    Check if hf_model refers to a local path rather than a HuggingFace model.

    Args:
        hf_model: Model identifier (HF name or local path)

    Returns:
        True if this appears to be a local path, False if HuggingFace model name
    """
    # Check for path-like patterns
    if hf_model.startswith(('.', '/', '~')):
        return True
    # Check if it exists as a local directory
    if os.path.isdir(hf_model):
        return True
    return False


def get_init_model_identifier(args: DictConfig) -> str:
    """
    Get the short identifier for the base model.

    Uses init_model_id if provided, otherwise derives from hf_model name.
    Validation in __main__ ensures init_model_id is present when hf_model is a local path.

    Args:
        args: Hydra configuration object

    Returns:
        Short identifier for the model (e.g., "v81" or "xglm564m")
    """
    init_model_id = getattr(args, 'init_model_id', None)
    if init_model_id:
        return init_model_id
    return get_model_shortname(args.hf_model)


def get_tokenized_path(args: DictConfig) -> str:
    """
    Compute the canonical tokenized dataset path based on configuration.

    This is the single source of truth for tokenized dataset paths.

    Args:
        args: Hydra configuration object

    Returns:
        Path to tokenized dataset directory
    """
    init_model_identifier = get_init_model_identifier(args)

    if args.focus.enabled:
        tokenizer_suffix = get_tokenizer_suffix(args)
        return f"{args.dataset.cache_dir}/tokenized_focus_{init_model_identifier}_{tokenizer_suffix}"
    else:
        return f"{args.dataset.cache_dir}/tokenized_{init_model_identifier}"


def get_seed_tokenizer_suffix(
    vocab_size: int,
    num_samples: int,
    seed_vocab_multiplier: float
) -> str:
    """
    Build tokenizer suffix for seed tokenizer used in hybrid seed vocabulary approach.

    The seed tokenizer is the intermediate large tokenizer trained to extract target
    vocabulary. It can be shared across different lambda values since it doesn't depend
    on the merging parameters (lambda, round_mode).

    Args:
        vocab_size: Target vocabulary size (not the intermediate size)
        num_samples: Number of training samples
        seed_vocab_multiplier: Multiplier for seed tokenizer size

    Returns:
        Suffix like "focus-v16k-s200k_seed-5.0x"
    """
    vocab_str = format_number(vocab_size)
    samples_str = format_number(num_samples)

    return f"focus-v{vocab_str}-s{samples_str}_seed-{seed_vocab_multiplier}x"


def get_tokenizer_suffix(args: DictConfig) -> str:
    """
    Build tokenizer suffix encoding vocab size, num samples, and other tokenizer parameters.

    Args:
        args: Hydra configuration object

    Returns:
        Suffix string like "focus-v16k-s200k", "focus-v16k-s200k_seeded-5.0x-lambda0.5",
        "focus-v16k-s200k_customdata", or "focus-v16k-s200k_no-additional_seeded-5.0x-lambda0.7-min2"
    """
    vocab_str = format_number(args.focus.vocab_size)
    samples_str = format_number(args.focus.num_samples)
    suffix = f"focus-v{vocab_str}-s{samples_str}"

    if not args.focus.get('inherit_additional_special_tokens', True):
        suffix += "_no-additional"

    # Indicate if using separate FOCUS dataset (not training dataset)
    if hasattr(args.focus, 'dataset') and args.focus.dataset is not None:
        suffix += "_customdata"

    # Include seed vocabulary status in path to avoid cache collision
    if args.focus.get('use_seed_vocabulary', False):
        suffix += "_seeded"
        # Add multiplier (modifies seeded approach)
        vocab_multiplier = args.focus.get('seed_vocab_multiplier', 5.0)
        suffix += f"-{vocab_multiplier}x"
        # Add lambda (the varying parameter in sweeps)
        seed_lambda = args.focus.get('seed_lambda', 0.5)
        suffix += f"-lambda{seed_lambda}"
        # Add other non-default parameters
        min_freq = args.focus.get('seed_min_frequency', 1)
        if min_freq > 1:
            suffix += f"-min{min_freq}"
        score_mode = args.focus.get('seed_score_mode', 'count')
        if score_mode != 'count':
            suffix += f"-{score_mode}"

    return suffix


def set_random_seeds(seed: int):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def initialize_model_and_tokenizer(args: DictConfig):
    """
    Initialize model and tokenizer with optional FOCUS vocabulary specialization.

    Args:
        args: Hydra configuration object

    Returns:
        Tuple of (model, tokenizer, tokenized_path)
    """
    if args.focus.enabled:
        return _initialize_focus_model(args)
    else:
        return _initialize_standard_model(args)


def _initialize_focus_model(args: DictConfig):
    """
    Initialize model with FOCUS vocabulary specialization.

    Args:
        args: Hydra configuration object

    Returns:
        Tuple of (model, tokenizer, tokenized_path)
    """
    print("=" * 60, file=sys.stderr)
    print("FOCUS MODE ENABLED", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Build directory paths with formatted vocab size and sample count
    init_model_identifier = get_init_model_identifier(args)
    tokenizer_suffix = get_tokenizer_suffix(args)
    focus_suffix = f"{init_model_identifier}_{tokenizer_suffix}"

    # Prepare JSONL training data for FOCUS
    # Store FOCUS training data alongside the dataset it's sampled from
    if hasattr(args.focus, 'dataset') and args.focus.dataset is not None:
        # Using separate FOCUS dataset - store in that dataset's cache dir
        focus_data_cache = args.focus.dataset.cache_dir
        jsonl_path = prepare_focus_training_data(
            num_samples=args.focus.num_samples,
            output_jsonl_path=f"{focus_data_cache}/{focus_suffix}/training_subset.jsonl",
            seed=args.seed,
            dataset_config=args.focus.dataset
        )
    else:
        # Using training dataset - store in training dataset's cache dir
        jsonl_path = prepare_focus_training_data(
            num_samples=args.focus.num_samples,
            output_jsonl_path=f"{args.dataset.cache_dir}/{focus_suffix}/training_subset.jsonl",
            seed=args.seed,
            train_dataset_cache=args.dataset.cache_dir
        )

    # Load existing tokenizer or train a new one
    if args.focus.tokenizer_path:
        print(f"Loading tokenizer from {args.focus.tokenizer_path}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.focus.tokenizer_path)
    else:
        tokenizer_output_dir = f"tokenizers/{args.dataset.language}/{focus_suffix}"

        # Check if tokenizer cache and config exist
        tokenizer_cache_exists = os.path.exists(
            os.path.join(tokenizer_output_dir, "tokenizer.json")
        )
        tokenizer_config_exists = os.path.exists(
            os.path.join(tokenizer_output_dir, "training_config.yaml")
        )

        # Verify config matches if both cache and config exist
        if tokenizer_cache_exists and tokenizer_config_exists:
            check_tokenizer_config(args, tokenizer_output_dir)
        elif tokenizer_cache_exists and not tokenizer_config_exists:
            print(
                f"Note: Using cached tokenizer at {tokenizer_output_dir} without config tracking\n"
                f"      (artifact was created before config tracking was implemented)",
                file=sys.stderr
            )

        tokenizer = train_new_tokenizer(
            jsonl_path=jsonl_path,
            base_tokenizer_name=args.hf_model,
            vocab_size=args.focus.vocab_size,
            output_path=tokenizer_output_dir,
            num_samples=args.focus.num_samples,
            inherit_additional_special_tokens=args.focus.get('inherit_additional_special_tokens', True),
            character_coverage=args.focus.get('character_coverage', 1.0),
            use_seed_vocabulary=args.focus.get('use_seed_vocabulary', False),
            seed_min_frequency=args.focus.get('seed_min_frequency', 1),
            seed_lambda=args.focus.get('seed_lambda', 0.5),
            seed_round_mode=args.focus.get('seed_round_mode', 'round'),
            seed_vocab_multiplier=args.focus.get('seed_vocab_multiplier', 5.0),
            seed_score_mode=args.focus.get('seed_score_mode', 'count'),
        )

        # Save config if we just created the tokenizer
        if not tokenizer_cache_exists:
            save_tokenizer_config(args, tokenizer_output_dir)

    # Load model and apply FOCUS
    print(f"Loading model: {args.hf_model}", file=sys.stderr)

    # Load config and override dropout if specified
    config = AutoConfig.from_pretrained(args.hf_model)
    if hasattr(args.training, 'dropout'):
        config.dropout = args.training.dropout
        print(f"  Overriding dropout: {config.dropout}", file=sys.stderr)
    if hasattr(args.training, 'attention_dropout'):
        config.attention_dropout = args.training.attention_dropout
        print(f"  Overriding attention_dropout: {config.attention_dropout}", file=sys.stderr)
    if hasattr(args.training, 'activation_dropout'):
        config.activation_dropout = args.training.activation_dropout
        print(f"  Overriding activation_dropout: {config.activation_dropout}", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(args.hf_model, config=config)
    source_tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    # Cache embeddings in tokenizer directory
    # If using custom tokenizer path, cache there; otherwise use the trained tokenizer directory
    embedding_cache_dir = args.focus.tokenizer_path if args.focus.tokenizer_path else tokenizer_output_dir

    new_input_embeddings, new_output_embeddings = apply_focus_initialization(
        source_model=model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=tokenizer,
        training_data_path=jsonl_path,
        fasttext_model_min_count=args.focus.get('fasttext_model_min_count', 4),
        cache_dir=embedding_cache_dir
    )

    # Resize model vocabulary and replace embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Set new input embeddings
    new_input_embedding_layer = torch.nn.Embedding.from_pretrained(
        new_input_embeddings,
        padding_idx=tokenizer.pad_token_id
    )
    model.set_input_embeddings(new_input_embedding_layer)

    # Set new output embeddings if model doesn't tie weights
    if hasattr(model.config, 'tie_word_embeddings') and not model.config.tie_word_embeddings:
        if new_output_embeddings is not None:
            model.get_output_embeddings().weight.data = new_output_embeddings  # type: ignore
    else:
        # Tie weights for models that use tied embeddings
        model.tie_weights()

    del source_tokenizer
    print("FOCUS initialization complete", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Use canonical path function for consistency
    tokenized_path = get_tokenized_path(args)

    return model, tokenizer, tokenized_path


def _initialize_standard_model(args: DictConfig):
    """
    Initialize model with standard pretrained tokenizer and weights.

    Args:
        args: Hydra configuration object

    Returns:
        Tuple of (model, tokenizer, tokenized_path)
    """
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    # Load config and override dropout if specified
    config = AutoConfig.from_pretrained(args.hf_model)
    if hasattr(args.training, 'dropout'):
        config.dropout = args.training.dropout
        print(f"  Overriding dropout: {config.dropout}", file=sys.stderr)
    if hasattr(args.training, 'attention_dropout'):
        config.attention_dropout = args.training.attention_dropout
        print(f"  Overriding attention_dropout: {config.attention_dropout}", file=sys.stderr)
    if hasattr(args.training, 'activation_dropout'):
        config.activation_dropout = args.training.activation_dropout
        print(f"  Overriding activation_dropout: {config.activation_dropout}", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(args.hf_model, config=config)
    tokenized_path = get_tokenized_path(args)

    return model, tokenizer, tokenized_path
