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
from transformers import AutoModelForCausalLM, AutoTokenizer

from focus_utils import (
    apply_focus_initialization,
    prepare_focus_training_data,
    train_new_tokenizer
)


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
    vocab_str = format_number(args.focus.vocab_size)
    samples_str = format_number(args.focus.num_samples)
    focus_suffix = f"vocab{vocab_str}_samples{samples_str}"

    # Prepare JSONL training data for FOCUS
    training_data_output = f"data/{args.language_code}_focus/{focus_suffix}/training_subset.jsonl"
    jsonl_path = prepare_focus_training_data(
        dataset_path=args.dataset_path,
        num_samples=args.focus.num_samples,
        output_jsonl_path=training_data_output,
        seed=args.seed
    )

    # Load existing tokenizer or train a new one
    if args.focus.tokenizer_path:
        print(f"Loading tokenizer from {args.focus.tokenizer_path}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.focus.tokenizer_path)
    else:
        tokenizer_output_dir = f"tokenizers/{args.language_code}/{focus_suffix}"
        tokenizer = train_new_tokenizer(
            jsonl_path=jsonl_path,
            base_tokenizer_name=args.hf_model,
            vocab_size=args.focus.vocab_size,
            output_path=tokenizer_output_dir
        )

    # Load model and apply FOCUS
    print(f"Loading model: {args.hf_model}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model)
    source_tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    new_input_embeddings, new_output_embeddings = apply_focus_initialization(
        source_model=model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=tokenizer,
        training_data_path=jsonl_path
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

    # Determine tokenized dataset path for FOCUS (separate from standard tokenized data)
    tokenized_path = f"{args.dataset_path}/tokenized_focus_{focus_suffix}"

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
    model = AutoModelForCausalLM.from_pretrained(args.hf_model)
    tokenized_path = args.dataset_path + "/tokenized"

    return model, tokenizer, tokenized_path
