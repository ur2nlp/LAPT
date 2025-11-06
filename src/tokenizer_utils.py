"""
Utilities for FOCUS-based vocabulary and embedding reinitialization.

This module provides functions to:
1. Prepare training data in JSONL format for FOCUS
2. Train new language-specific tokenizers
3. Apply FOCUS to initialize embeddings for the new vocabulary
"""

import json
import os
import random
import sys
from typing import Optional

import sentencepiece as spm
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast


def prepare_focus_training_data(
    num_samples: int,
    output_jsonl_path: str,
    seed: int = 1,
    train_dataset_cache: str = None,
    dataset_config = None
) -> str:
    """
    Extract a random subset of untokenized data and convert to JSONL format.

    Args:
        num_samples: Number of samples to extract
        output_jsonl_path: Path where JSONL file will be saved
        seed: Random seed for reproducible sampling
        train_dataset_cache: Path to training dataset cache directory (for reusing training data)
        dataset_config: Optional separate dataset configuration for FOCUS

    Returns:
        Path to the created JSONL file
    """
    if os.path.exists(output_jsonl_path):
        print(f"JSONL data already exists at {output_jsonl_path}, skipping generation", file=sys.stderr)
        return output_jsonl_path

    print(f"Preparing FOCUS training data: {num_samples} samples", file=sys.stderr)

    # If dataset_config provided, load that dataset; otherwise use training dataset
    if dataset_config is not None:
        # Import here to avoid circular dependency (dataset_utils imports tokenizer_utils)
        from dataset_utils import load_untokenized_dataset
        # Use the JSONL output directory as the cache for the FOCUS dataset
        focus_cache = os.path.dirname(output_jsonl_path)
        untokenized_path = load_untokenized_dataset(
            dataset_config=dataset_config,
            cache_dir=focus_cache
        )
        dataset = load_from_disk(untokenized_path)
    else:
        if train_dataset_cache is None:
            raise ValueError("Either train_dataset_cache or dataset_config must be provided")
        untokenized_path = os.path.join(train_dataset_cache, "untokenized")
        if os.path.exists(untokenized_path):
            dataset = load_from_disk(untokenized_path)
        else:
            raise FileNotFoundError(
                f"Untokenized dataset not found at {untokenized_path}. "
                "Please ensure the dataset is loaded first."
            )

    train_data = dataset['train']
    total_samples = len(train_data)

    if num_samples > total_samples:
        print(
            f"Warning: Requested {num_samples} samples but dataset only has {total_samples}. "
            f"Using all available samples.",
            file=sys.stderr
        )
        num_samples = total_samples

    random.seed(seed)
    indices = random.sample(range(total_samples), num_samples)
    # Sort indices for efficient sequential access to memory-mapped dataset
    indices.sort()

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx in indices:
            text = train_data[idx]['text']
            json.dump({'text': text}, f, ensure_ascii=False)
            f.write('\n')

    print(f"JSONL data saved to {output_jsonl_path}", file=sys.stderr)
    return output_jsonl_path


def train_new_tokenizer(
    jsonl_path: str,
    base_tokenizer_name: str,
    vocab_size: int,
    output_path: str,
    inherit_additional_special_tokens: bool = True
) -> PreTrainedTokenizerFast:
    """
    Train a new tokenizer on JSONL data using SentencePiece library.

    The tokenizer will use the same algorithm (BPE, Unigram, etc.) as the base tokenizer.

    Args:
        jsonl_path: Path to JSONL file with training data
        base_tokenizer_name: Name of base model tokenizer (for special tokens and algorithm)
        vocab_size: Target vocabulary size
        output_path: Directory where trained tokenizer will be saved
        inherit_additional_special_tokens: Whether to inherit additional special tokens
            (e.g., <madeupword0-6>) from base tokenizer (default: True for compatibility)

    Returns:
        Trained tokenizer
    """
    # Check if tokenizer already trained and cached
    if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "tokenizer.json")):
        print(f"Tokenizer already exists at {output_path}, loading it", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)
        _validate_tokenizer(tokenizer, vocab_size)
        return tokenizer

    print(f"Training new tokenizer with vocab size {vocab_size}", file=sys.stderr)

    # Inspect base tokenizer to determine algorithm and special tokens to inherit
    # Force Fast tokenizer since we need to access backend_tokenizer for algorithm detection
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, use_fast=True)

    model_type = _detect_tokenizer_algorithm(base_tokenizer)
    print(f"Detected tokenizer algorithm: {model_type}", file=sys.stderr)

    special_tokens_config = _extract_special_tokens(
        base_tokenizer,
        inherit_additional=inherit_additional_special_tokens
    )

    # Convert JSONL to plain text for SentencePiece training (cached alongside JSONL)
    # We keep the JSONL for FOCUS which needs that format later
    text_file_path = jsonl_path.replace('.jsonl', '_spm.txt')
    if not os.path.exists(text_file_path):
        print(f"Creating SentencePiece training file: {text_file_path}", file=sys.stderr)
        with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                for line in jsonl_file:
                    data = json.loads(line)
                    text_file.write(data['text'] + '\n')
    else:
        print(f"SentencePiece training file already exists: {text_file_path}", file=sys.stderr)

    os.makedirs(output_path, exist_ok=True)

    # Train SentencePiece model
    sp_model = _train_sentencepiece_model(
        text_file_path=text_file_path,
        model_type=model_type,
        vocab_size=vocab_size,
        special_tokens_config=special_tokens_config,
        output_path=output_path
    )

    # Extract vocabulary with scores for HuggingFace tokenizer initialization
    actual_vocab_size = sp_model.get_piece_size()
    vocab_with_scores = [
        (sp_model.id_to_piece(i), sp_model.get_score(i))
        for i in range(actual_vocab_size)
    ]

    # Convert SentencePiece model to HuggingFace tokenizer backend
    # Note: Asymmetric API - BPE can load from file, Unigram must be built manually
    if model_type == 'bpe':
        # SentencePieceBPETokenizer has .from_file() - can directly load .model file
        from tokenizers import SentencePieceBPETokenizer
        model_file = os.path.join(output_path, 'spm.model')
        backend_tokenizer = SentencePieceBPETokenizer.from_file(
            vocab=model_file,
            replacement="▁",
            add_prefix_space=True
        )
    else:
        # Unigram lacks .from_file() - must manually construct from vocab+scores
        unk_id = special_tokens_config.get('unk_id', 0)
        backend_tokenizer = _create_unigram_tokenizer(vocab_with_scores, unk_id=unk_id)

    # Wrap in PreTrainedTokenizerFast with special tokens from base model
    new_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        bos_token=base_tokenizer.bos_token,
        eos_token=base_tokenizer.eos_token,
        unk_token=base_tokenizer.unk_token,
        pad_token=base_tokenizer.pad_token,
    )

    # Add additional special tokens ONLY if we inherited them
    # (They're already in the SentencePiece vocab via user_defined_symbols,
    #  but PreTrainedTokenizerFast needs to know about them explicitly.
    #  This is REGISTRATION not ADDITION - we're just setting the
    #  additional_special_tokens attribute, not increasing vocab size)
    if inherit_additional_special_tokens:
        if hasattr(base_tokenizer, 'additional_special_tokens') and base_tokenizer.additional_special_tokens:
            new_tokenizer.add_special_tokens({
                'additional_special_tokens': base_tokenizer.additional_special_tokens
            })

    new_tokenizer.save_pretrained(output_path)
    print(f"Tokenizer saved to {output_path}", file=sys.stderr)

    # Validate vocab size and token ID contiguity
    _validate_tokenizer(new_tokenizer, vocab_size)

    return new_tokenizer


def _train_sentencepiece_model(
    text_file_path: str,
    model_type: str,
    vocab_size: int,
    special_tokens_config: dict,
    output_path: str
) -> spm.SentencePieceProcessor:
    """
    Train a SentencePiece model and return the loaded processor.

    Args:
        text_file_path: Path to plain text training file (one sentence per line)
        model_type: 'bpe' or 'unigram'
        vocab_size: Target vocabulary size
        special_tokens_config: Dict of special token configs (from _extract_special_tokens)
        output_path: Directory where model files will be saved

    Returns:
        Loaded SentencePieceProcessor with the trained model
    """
    model_prefix = os.path.join(output_path, 'spm')

    # Train SentencePiece model
    # character_coverage=1.0: include all characters (no sampling for rare chars)
    # normalization_rule_name='identity': no text normalization
    # hard_vocab_limit=True: strictly enforce vocab_size (not a soft target)
    train_args = {
        'input': text_file_path,
        'model_prefix': model_prefix,
        'model_type': model_type,
        'vocab_size': vocab_size,
        'character_coverage': 1.0,
        'normalization_rule_name': 'identity',
        'hard_vocab_limit': True,
    }

    train_args.update(special_tokens_config)

    train_args_str = ' '.join([f'--{k}={v}' for k, v in train_args.items()])
    print(f"Training SentencePiece with args: {train_args_str}", file=sys.stderr)

    spm.SentencePieceTrainer.Train(train_args_str)

    # Load the trained model and validate vocab size
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f'{model_prefix}.model')

    actual_vocab_size = sp_model.get_piece_size()
    print(f"SentencePiece model trained. Vocab size: {actual_vocab_size}", file=sys.stderr)

    if actual_vocab_size != vocab_size:
        raise ValueError(
            f"Trained SentencePiece model has vocab size {actual_vocab_size}, "
            f"but expected {vocab_size}. This may indicate a SentencePiece training issue."
        )

    return sp_model


def _detect_tokenizer_algorithm(tokenizer: PreTrainedTokenizerFast) -> str:
    """
    Detect whether a tokenizer uses BPE or Unigram algorithm.

    Requires a Fast tokenizer (PreTrainedTokenizerFast) to access backend_tokenizer.

    Args:
        tokenizer: HuggingFace Fast tokenizer to inspect

    Returns:
        'bpe' or 'unigram'
    """
    backend_model = tokenizer.backend_tokenizer.model
    model_type_str = str(type(backend_model).__name__).lower()

    if 'bpe' in model_type_str:
        return 'bpe'
    elif 'unigram' in model_type_str:
        return 'unigram'
    else:
        raise ValueError(
            f"Unknown tokenizer algorithm: {type(backend_model)}. "
            "Expected BPE or Unigram."
        )
    

def _create_unigram_tokenizer(vocab_scores: list[tuple[str, float]], unk_id: int = 0):
    """
    Create a HuggingFace Tokenizer with Unigram model from SentencePiece vocabulary.

    Builds a complete tokenization pipeline with:
    - Unigram model initialized with vocab and scores
    - Empty normalizer (no text normalization)
    - Metaspace pre-tokenizer for SentencePiece-style space handling

    Args:
        vocab_scores: List of (token, score) tuples from SentencePiece model
        unk_id: Token ID for unknown tokens (default: 0)

    Returns:
        Configured Tokenizer object ready for use with PreTrainedTokenizerFast
    """
    from tokenizers import Tokenizer, normalizers
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Metaspace

    # Initialize Unigram model with vocabulary and scores from SentencePiece
    # byte_fallback=False: use <unk> for unknown chars (matches SentencePiece training)
    unigram_model = Unigram(vocab_scores, unk_id=unk_id, byte_fallback=False)
    backend_tokenizer = Tokenizer(unigram_model)

    # Configure tokenization pipeline to match SentencePiece behavior:
    # - Empty normalizer: no text transformations (matches normalization_rule_name='identity')
    # - Metaspace: handle spaces as ▁ tokens (SentencePiece convention)
    backend_tokenizer.normalizer = normalizers.Sequence(normalizers=[])  # type: ignore
    backend_tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")

    return backend_tokenizer


def _validate_tokenizer(tokenizer: PreTrainedTokenizerBase, expected_vocab_size: int):
    """
    Validate that a tokenizer has the expected vocab size and contiguous token IDs.

    Args:
        tokenizer: Tokenizer to validate
        expected_vocab_size: Expected vocabulary size

    Raises:
        ValueError: If validation fails
    """
    actual_vocab_size = len(tokenizer)

    if actual_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer has vocab size {actual_vocab_size}, "
            f"but expected {expected_vocab_size}"
        )

    # Check that token IDs are contiguous from 0 to vocab_size-1
    # HuggingFace's .train_new_from_iterator() had a bug where it would skip ID 0
    # at larger vocab sizes (e.g., creating IDs 1-4095 instead of 0-4095 for vocab_size=4096)
    vocab = tokenizer.get_vocab()
    all_token_ids = list(vocab.values())

    if len(all_token_ids) != actual_vocab_size:
        raise ValueError(
            f"Vocab has {len(all_token_ids)} entries but vocab_size is {actual_vocab_size}"
        )

    min_id = min(all_token_ids)
    max_id = max(all_token_ids)

    if min_id != 0 or max_id != actual_vocab_size - 1:
        raise ValueError(
            f"Token IDs are not contiguous! Range is {min_id}-{max_id}, "
            f"expected 0-{actual_vocab_size - 1}"
        )

    # Check for duplicates or gaps in token IDs
    unique_ids = set(all_token_ids)
    if len(unique_ids) != actual_vocab_size:
        raise ValueError(
            f"Token IDs have duplicates or gaps! "
            f"Found {len(unique_ids)} unique IDs but expected {actual_vocab_size}"
        )

    print(f"Tokenizer validation passed: vocab_size={actual_vocab_size}, token IDs: {min_id}-{max_id}", file=sys.stderr)


def _extract_special_tokens(tokenizer: PreTrainedTokenizerBase, inherit_additional: bool = True) -> dict:
    """
    Extract special token configuration from a tokenizer for SentencePiece training.

    Args:
        tokenizer: HuggingFace tokenizer to extract special tokens from
        inherit_additional: Whether to inherit additional special tokens (e.g., <madeupword0-6>)
            from the base tokenizer (default: True)

    Returns:
        Dictionary of SentencePiece training arguments for special tokens
    """
    config = {}

    user_defined_symbols = []

    if tokenizer.unk_token is not None:
        config['unk_piece'] = tokenizer.unk_token
        config['unk_id'] = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    if tokenizer.bos_token is not None:
        config['bos_piece'] = tokenizer.bos_token
        config['bos_id'] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1

    if tokenizer.eos_token is not None:
        config['eos_piece'] = tokenizer.eos_token
        config['eos_id'] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    if tokenizer.pad_token is not None:
        config['pad_piece'] = tokenizer.pad_token
        # Default to -1 if pad_token_id is None (SentencePiece convention for "no padding")
        config['pad_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

    # Optionally inherit additional special tokens like <madeupword0-6>
    # These are vocabulary reservations from the base model that may be unused
    if inherit_additional:
        if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
            user_defined_symbols.extend(tokenizer.additional_special_tokens)

    if user_defined_symbols:
        config['user_defined_symbols'] = ','.join(user_defined_symbols)

    return config


def apply_focus_initialization(
    source_model,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    training_data_path: str,
    fasttext_model_min_count: int = 4
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply FOCUS to generate new input embeddings and optionally output embeddings.

    Args:
        source_model: Source pretrained model
        source_tokenizer: Tokenizer for the source model
        target_tokenizer: Target language-specific tokenizer
        training_data_path: Path to JSONL training data for FOCUS
        fasttext_model_min_count: Minimum occurrences for FastText embeddings (default: 4)

    Returns:
        Tuple of (input_embeddings, output_embeddings)
        output_embeddings will be None if model ties word embeddings
    """
    print("Applying FOCUS to initialize embeddings", file=sys.stderr)

    try:
        from deepfocus import FOCUS
    except ImportError:
        raise ImportError(
            "deepfocus package not found. Please install it with: pip install deepfocus"
        )

    source_embeddings = source_model.get_input_embeddings().weight

    new_input_embeddings = FOCUS(
        source_embeddings=source_embeddings,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        target_training_data_path=training_data_path,
        fasttext_model_min_count=fasttext_model_min_count
    )

    new_output_embeddings = None
    if hasattr(source_model.config, 'tie_word_embeddings') and not source_model.config.tie_word_embeddings:
        print("Model uses separate output embeddings, applying FOCUS to output embeddings", file=sys.stderr)
        source_output_embeddings = source_model.get_output_embeddings().weight
        new_output_embeddings = FOCUS(
            source_embeddings=source_output_embeddings,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            target_training_data_path=training_data_path,
            fasttext_model_min_count=fasttext_model_min_count
        )

    print(f"FOCUS initialization complete. New vocab size: {len(target_tokenizer)}", file=sys.stderr)
    return new_input_embeddings, new_output_embeddings
