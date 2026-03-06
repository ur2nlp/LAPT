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
from config_utils import save_config, load_config, check_config_match
from artifact_configs import TokenizerConfig


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
    tokenizer_config = TokenizerConfig.from_args(args)

    if tokenizer_config is not None:
        return f"{args.dataset.cache_dir}/tokenized_{tokenizer_config.focus_suffix()}"
    else:
        init_model_identifier = get_init_model_identifier(args)
        return f"{args.dataset.cache_dir}/tokenized_{init_model_identifier}"



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

    # Extract tokenizer config (handles path generation and validation)
    tokenizer_config = TokenizerConfig.from_args(args)
    focus_suffix = tokenizer_config.focus_suffix()

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
        tokenizer_output_dir = tokenizer_config.cache_dir(args.dataset.language)

        # Check if tokenizer cache and config exist
        tokenizer_cache_exists = os.path.exists(
            os.path.join(tokenizer_output_dir, "tokenizer.json")
        )
        config_path = os.path.join(tokenizer_output_dir, "training_config.yaml")
        tokenizer_config_exists = os.path.exists(config_path)

        # Verify config matches if both cache and config exist
        if tokenizer_cache_exists and tokenizer_config_exists:
            cached_config = load_config(config_path)
            check_config_match(cached_config, tokenizer_config.to_dict(), "Tokenizer")
        elif tokenizer_cache_exists and not tokenizer_config_exists:
            print(
                f"Note: Using cached tokenizer at {tokenizer_output_dir} without config tracking\n"
                f"      (artifact was created before config tracking was implemented)",
                file=sys.stderr
            )

        tokenizer = train_new_tokenizer(
            config=tokenizer_config,
            jsonl_path=jsonl_path,
            output_path=tokenizer_output_dir
        )

        # Save config if we just created the tokenizer
        if not tokenizer_cache_exists:
            save_config(tokenizer_config.to_dict(), config_path)
            print(f"Saved tokenizer config to {config_path}", file=sys.stderr)

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
        fasttext_model_min_count=tokenizer_config.fasttext_model_min_count,
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
