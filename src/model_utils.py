"""
Utilities for model and tokenizer initialization.

This module handles both standard model loading and FOCUS-based vocabulary
specialization workflows.
"""

import os
import random
import sys
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tokenizer_utils import (
    LEGACY_INPUT_NAME,
    apply_focus_initialization,
    prepare_focus_training_data,
    resolve_cached_embedding_paths,
    train_new_tokenizer,
)
from artifact_configs import (
    TokenizerConfig, get_model_shortname, format_number, effective_dataset_cache_dir,
    focus_embedding_hash,
)


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
    cache_dir = effective_dataset_cache_dir(args)

    if tokenizer_config is not None:
        return f"{cache_dir}/tokenized_{tokenizer_config.tokenizer_id()}"
    else:
        init_model_identifier = get_init_model_identifier(args)
        return f"{cache_dir}/tokenized_{init_model_identifier}"



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
    tokenizer_id = tokenizer_config.tokenizer_id()

    # Hash of the inputs that determine FOCUS embedding values (mix + num_samples
    # + seed + fasttext_min_count). Independent of tokenizer hyperparameters.
    emb_hash = focus_embedding_hash(args)
    reuse_policy = args.focus.get('reuse_embeddings', None)

    # Pre-compute target paths so we can decide whether the JSONL is needed at
    # all. The JSONL is consumed by (a) tokenizer training (SentencePiece) and
    # (b) FOCUS's fastText fit on the target side; if both the tokenizer and
    # the per-mix embeddings are already cached, nothing reads it.
    if args.focus.tokenizer_path:
        embedding_cache_dir = args.focus.tokenizer_path
        tokenizer_cache_exists = True
    else:
        embedding_cache_dir = tokenizer_config.cache_dir(args.dataset.language)
        tokenizer_cache_exists = os.path.exists(
            os.path.join(embedding_cache_dir, "tokenizer.json")
        )

    resolved_emb_paths = resolve_cached_embedding_paths(
        embedding_cache_dir, emb_hash, reuse_policy
    )
    embeddings_cache_hit = resolved_emb_paths is not None

    # When reuse_policy == 'any' resolves to a unique cached set, record the
    # specific hash back into args so the saved training_config.yaml documents
    # exactly which embeddings this run used. Legacy unhashed sidecars have no
    # hash; leave 'any' in place rather than fabricate one.
    if reuse_policy == 'any' and resolved_emb_paths is not None:
        input_pt = resolved_emb_paths[0]
        basename = os.path.basename(input_pt)
        if basename.endswith('.input.pt') and basename != LEGACY_INPUT_NAME:
            resolved_hash = basename[: -len('.input.pt')]
            args.focus.reuse_embeddings = resolved_hash
            print(
                f"Resolved focus.reuse_embeddings='any' -> '{resolved_hash}'",
                file=sys.stderr,
            )

    jsonl_needed = not (tokenizer_cache_exists and embeddings_cache_hit)
    jsonl_path: Optional[str] = None
    if jsonl_needed:
        # Subset name depends only on source data + num_samples + seed and is
        # therefore shared across FOCUS runs on the same mix.
        subset_filename = (
            f"training_subset_s{format_number(args.focus.num_samples)}"
            f"_seed{args.seed}.jsonl"
        )
        if hasattr(args.focus, 'dataset') and args.focus.dataset is not None:
            focus_data_cache = args.focus.dataset.cache_dir
            jsonl_path = prepare_focus_training_data(
                num_samples=args.focus.num_samples,
                output_jsonl_path=f"{focus_data_cache}/{subset_filename}",
                seed=args.seed,
                dataset_config=args.focus.dataset
            )
        else:
            training_cache_dir = effective_dataset_cache_dir(args)
            jsonl_path = prepare_focus_training_data(
                num_samples=args.focus.num_samples,
                output_jsonl_path=f"{training_cache_dir}/{subset_filename}",
                seed=args.seed,
                train_dataset_cache=training_cache_dir
            )
    else:
        print(
            f"Skipping FOCUS JSONL materialization: tokenizer cache present "
            f"and embeddings cache hit for hash {emb_hash}.",
            file=sys.stderr,
        )

    # Load existing tokenizer or train a new one
    if args.focus.tokenizer_path:
        print(f"Loading tokenizer from {args.focus.tokenizer_path}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.focus.tokenizer_path)
    else:
        tokenizer_output_dir = embedding_cache_dir
        config_path = os.path.join(tokenizer_output_dir, "training_config.yaml")

        # Verify config matches if cache exists
        if tokenizer_cache_exists:
            if os.path.exists(config_path):
                tokenizer_config.check_cached(config_path)
            else:
                print(
                    f"Note: Using cached tokenizer at {tokenizer_output_dir}"
                    f" without config tracking\n"
                    f"      (artifact was created before config tracking was implemented)",
                    file=sys.stderr
                )

        # train_new_tokenizer is cache-aware: returns the existing tokenizer
        # without re-training when the cache is populated. jsonl_path may be
        # None in that case since it is not consulted.
        tokenizer = train_new_tokenizer(
            config=tokenizer_config,
            jsonl_path=jsonl_path,
            output_path=tokenizer_output_dir
        )

        if not tokenizer_cache_exists:
            tokenizer_config.save(config_path)

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

    # Build provenance meta for the embedding sidecar.
    if hasattr(args.focus, 'dataset') and args.focus.dataset is not None:
        data_spec = OmegaConf.to_container(args.focus.dataset, resolve=True)
    else:
        data_spec = None  # uses training dataset; recorded via effective cache dir below
    embedding_meta = {
        'embedding_hash': emb_hash,
        'tokenizer_id': tokenizer_id,
        'num_samples': args.focus.num_samples,
        'seed': args.seed,
        'fasttext_model_min_count': tokenizer_config.fasttext_model_min_count,
        'focus_dataset': data_spec,
        'train_dataset_cache': (
            effective_dataset_cache_dir(args) if data_spec is None else None
        ),
    }

    new_input_embeddings, new_output_embeddings = apply_focus_initialization(
        source_model=model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=tokenizer,
        training_data_path=jsonl_path,
        fasttext_model_min_count=tokenizer_config.fasttext_model_min_count,
        cache_dir=embedding_cache_dir,
        embedding_hash=emb_hash,
        embedding_meta=embedding_meta,
        reuse_policy=reuse_policy,
    )

    # Resize the existing embedding module in place, then overwrite its weight
    # data with the new rows. Replacing the module via set_input_embeddings
    # would drop the XGLMScaledWordEmbedding subclass (with its sqrt(d_model)
    # forward-time scale), silently shrinking hidden states by ~32x.
    model.resize_token_embeddings(len(tokenizer))

    current_input_embed = model.get_input_embeddings()
    with torch.no_grad():
        current_input_embed.weight.data.copy_(
            new_input_embeddings.to(
                device=current_input_embed.weight.device,
                dtype=current_input_embed.weight.dtype,
            )
        )
    if tokenizer.pad_token_id is not None:
        current_input_embed.padding_idx = tokenizer.pad_token_id

    # Set new output embeddings if model doesn't tie weights
    if hasattr(model.config, 'tie_word_embeddings') and not model.config.tie_word_embeddings:
        if new_output_embeddings is not None:
            current_output_embed = model.get_output_embeddings()
            with torch.no_grad():
                current_output_embed.weight.data.copy_(
                    new_output_embeddings.to(
                        device=current_output_embed.weight.device,
                        dtype=current_output_embed.weight.dtype,
                    )
                )
    else:
        # Tie weights for models that use tied embeddings
        model.tie_weights()

    # Sync model config with the new tokenizer's special token IDs.
    # The PTEx tokenizer reorders pad/eos/unk relative to XGLM, so without
    # this update model.config.pad_token_id would point to the wrong token.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # The generation config is a separate object that takes precedence over
    # model.config during generation. It is derived from the base model and so
    # retains the base tokenizer's special-token IDs; it must be synced too, or
    # generation halts on the wrong EOS id and never stops on the real one.
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

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
