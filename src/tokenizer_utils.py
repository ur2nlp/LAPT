"""
Utilities for FOCUS-based vocabulary and embedding reinitialization.

This module provides functions to:
1. Prepare training data in JSONL format for FOCUS
2. Train new language-specific tokenizers
3. Apply FOCUS to initialize embeddings for the new vocabulary
"""

import glob
import json
import os
import random
import sys
from typing import Optional

import sentencepiece as spm
import torch
import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from artifact_configs import TokenizerConfig


def extract_base_vocabulary_frequencies(
    text_file_path: str,
    base_tokenizer_name: str,
    output_file: str,
    filter_special_tokens: bool = True,
    min_frequency: int = 1
) -> dict[str, int]:
    """
    Tokenize corpus with base tokenizer and extract raw token frequencies.

    Raw counts are cached to disk and can be normalized later by the caller.

    Args:
        text_file_path: Path to plain text training file (one sentence per line)
        base_tokenizer_name: Name of base model tokenizer
        output_file: Path where base vocabulary counts will be saved (for caching)
        filter_special_tokens: Filter out <unk>, <s>, </s>, <pad>, and <madeupword*> tokens
        min_frequency: Minimum raw frequency threshold to include a token (default: 1)

    Returns:
        Dictionary mapping token strings to raw counts
    """
    # Check cache
    if os.path.exists(output_file):
        print(f"Base vocabulary already exists at {output_file}, loading it", file=sys.stderr)
        vocab = {}
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 2:
                    token, count_str = parts
                    vocab[token] = int(count_str)
        print(f"  Loaded {len(vocab)} tokens", file=sys.stderr)
        return vocab

    print(f"Extracting base vocabulary from {text_file_path}", file=sys.stderr)

    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, use_fast=True)

    # Count token frequencies
    from collections import Counter
    token_counts = Counter()

    with open(text_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue

            token_ids = base_tokenizer.encode(text, add_special_tokens=False)
            for token_id in token_ids:
                token_counts[token_id] += 1

    print(f"  Found {len(token_counts)} unique tokens in corpus", file=sys.stderr)

    # Convert token IDs to strings and filter
    filtered_vocab = {}
    filtered_reasons = Counter()

    for token_id, count in token_counts.items():
        if count < min_frequency:
            filtered_reasons['below_min_frequency'] += 1
            continue

        token_str = base_tokenizer.convert_ids_to_tokens(token_id)

        # Filter special tokens
        if filter_special_tokens:
            if token_str in ['<unk>', '<s>', '</s>', '<pad>', '<mask>']:
                filtered_reasons['special_token'] += 1
                continue
            if token_str.startswith('<madeupword'):
                filtered_reasons['madeupword_token'] += 1
                continue
            if token_str in base_tokenizer.all_special_tokens:
                filtered_reasons['registered_special'] += 1
                continue

        filtered_vocab[token_str] = count

    print(f"  After filtering: {len(filtered_vocab)} tokens", file=sys.stderr)
    if filtered_reasons:
        print(f"  Filtered out:", file=sys.stderr)
        for reason, count in filtered_reasons.most_common():
            print(f"    {reason}: {count}", file=sys.stderr)

    total_count = sum(filtered_vocab.values())
    print(f"  Total raw token count: {total_count:,}", file=sys.stderr)

    # Write raw counts to cache file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        # Sort by frequency (descending) for better readability
        for token_str, count in sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{token_str}\t{count}\n")

    print(f"Base vocabulary saved to {output_file}", file=sys.stderr)

    return filtered_vocab


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

    NOTE: Parameters affecting FOCUS training data (num_samples, seed, dataset source) are
    tracked via TokenizerConfig in artifact_configs.py since this data is only used
    for tokenizer training.
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
            cache_dir=focus_cache,
            dev_size=-1  # FOCUS doesn't need dev split (only uses train for tokenizer/embeddings)
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
        written_count = 0
        for idx in indices:
            text = train_data[idx]['text']
            # Skip blank lines
            if text.strip():
                json.dump({'text': text}, f, ensure_ascii=False)
                f.write('\n')
                written_count += 1

        if written_count < num_samples:
            print(
                f"Warning: Filtered out {num_samples - written_count} blank lines from FOCUS training data",
                file=sys.stderr
            )

    print(f"JSONL data saved to {output_jsonl_path}", file=sys.stderr)
    return output_jsonl_path


def extract_target_seed_vocab(
    spm_model_path: str,
    target_mass: int = 10_000_000,
    filter_special_tokens: bool = True
) -> dict[str, float]:
    """
    Extract vocabulary from trained SentencePiece model and normalize to target mass.

    Converts log probabilities to normalized counts: count = exp(log_prob) * target_mass.
    Returns exact floats — rounding is deferred to the merge step.

    Args:
        spm_model_path: Path to .model file from SentencePiece training
        target_mass: Target total count for normalization (should match base vocab total)
        filter_special_tokens: Whether to filter out special tokens

    Returns:
        Dictionary mapping token strings to normalized counts (exact floats)
    """
    import math

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spm_model_path)

    vocab = {}
    vocab_size = sp_model.get_piece_size()

    for i in range(vocab_size):
        token = sp_model.id_to_piece(i)
        log_prob = sp_model.get_score(i)

        # Skip special tokens if requested
        if filter_special_tokens:
            if log_prob == 0:  # Special tokens have score 0
                continue
            if token in ['<unk>', '<s>', '</s>', '<pad>', '<mask>']:
                continue
            if token.startswith('<madeupword'):
                continue

        # Convert log prob to normalized count (exact float, no rounding)
        # prob = exp(log_prob), count = prob * target_mass
        count = math.exp(log_prob) * target_mass

        if count > 0:
            vocab[token] = count

    return vocab


def apply_character_weighting(vocab: dict[str, int | float]) -> dict[str, float]:
    """Weight token counts by character length.

    Transforms raw token counts into character-coverage scores:
    `score[t] = count[t] * len(t)`. This measures how many characters of text
    each token handles, providing a more principled basis for comparing tokens
    across vocabularies with different size/shape distributions.

    Args:
        vocab: Dictionary mapping token strings to counts.

    Returns:
        New dictionary with counts multiplied by token character length.
    """
    return {token: count * len(token) for token, count in vocab.items()}


def normalize_vocab_mass(
    vocab: dict[str, int | float],
    target_mass: int,
) -> dict[str, float]:
    """Scale vocabulary counts so they sum to target_mass.

    Args:
        vocab: Dictionary mapping token strings to counts.
        target_mass: Desired total mass after normalization.

    Returns:
        New dictionary with counts scaled to sum to target_mass.
    """
    current_mass = sum(vocab.values())
    if current_mass == 0:
        return {token: 0.0 for token in vocab}
    scale = target_mass / current_mass
    return {token: count * scale for token, count in vocab.items()}


def merge_vocabularies(
    base_vocab: dict[str, int | float],
    target_vocab: dict[str, int | float],
    lambda_weight: float = 0.5,
    round_mode: str = "round"
) -> dict[str, int]:
    """
    Merge base and target vocabularies with lambda interpolation.

    Combined weight for each token:
    - If only in base: base_count * lambda
    - If only in target: target_count * (1 - lambda)
    - If in both: base_count * lambda + target_count * (1 - lambda)

    The lambda parameter interpolates between base and target vocabularies:
    - lambda=0.0: Pure target vocabulary
    - lambda=0.5: Equal interpolation (balanced)
    - lambda=1.0: Pure base vocabulary

    Args:
        base_vocab: Raw counts from base tokenizer corpus analysis
        target_vocab: Normalized counts from target-trained tokenizer
        lambda_weight: Interpolation weight (0=target, 1=base, default: 0.5)
        round_mode: Rounding method: "ceil", "floor", or "round"

    Returns:
        Merged vocabulary with combined weights
    """
    import math

    if round_mode == "ceil":
        round_func = math.ceil
    elif round_mode == "floor":
        round_func = math.floor
    elif round_mode == "round":
        round_func = round
    else:
        raise ValueError(f"Invalid round_mode: {round_mode}. Must be 'ceil', 'floor', or 'round'")

    all_tokens = set(base_vocab.keys()) | set(target_vocab.keys())
    merged = {}

    for token in all_tokens:
        base_count = base_vocab.get(token, 0)
        target_count = target_vocab.get(token, 0)

        combined_count = base_count * lambda_weight + target_count * (1 - lambda_weight)

        combined_count = round_func(combined_count)

        if combined_count > 0:
            merged[token] = combined_count

    return merged


def train_new_tokenizer(
    config: TokenizerConfig,
    jsonl_path: str,
    output_path: str,
) -> PreTrainedTokenizerFast:
    """
    Train a new tokenizer on JSONL data using SentencePiece library.

    The tokenizer will use the same algorithm (BPE, Unigram, etc.) as the base tokenizer.

    Args:
        config: TokenizerConfig containing all training parameters
        jsonl_path: Path to JSONL file with training data
        output_path: Directory where trained tokenizer will be saved

    Returns:
        Trained tokenizer

    NOTE: When adding parameters that affect the tokenizer artifact, update
    TokenizerConfig in artifact_configs.py to include them.
    """
    # Check if tokenizer already trained and cached
    if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "tokenizer.json")):
        print(f"Tokenizer already exists at {output_path}, loading it", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)
        _validate_tokenizer(tokenizer, config.vocab_size)
        return tokenizer

    print(f"Training new tokenizer with vocab size {config.vocab_size}", file=sys.stderr)

    # Inspect base tokenizer to determine algorithm and special tokens to inherit
    # Force Fast tokenizer since we need to access backend_tokenizer for algorithm detection
    base_tokenizer = AutoTokenizer.from_pretrained(config.hf_model, use_fast=True)

    model_type = _detect_tokenizer_algorithm(base_tokenizer)
    print(f"Detected tokenizer algorithm: {model_type}", file=sys.stderr)

    special_tokens_config = _extract_special_tokens(
        base_tokenizer,
        inherit_additional=config.inherit_additional_special_tokens
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

    # Generate seed vocabulary if enabled (hybrid approach)
    seed_file = None
    if config.use_seed_vocabulary:
        print(f"Generating hybrid seed vocabulary (lambda={config.seed_lambda})", file=sys.stderr)

        # Step 1: Train seed tokenizer to extract target-specific vocabulary
        # This is cached separately so it can be reused across different lambda values
        seed_vocab_size = int(config.vocab_size * config.seed_vocab_multiplier)

        if config.num_samples is not None:
            # Construct seed tokenizer path - saved alongside final tokenizer, not nested inside it
            # Example: tokenizers/old_germanic/xglm564m_focus-v16k-s200k_seed-5.0x/
            parent_dir = os.path.dirname(output_path)
            seed_output_path = os.path.join(parent_dir, config.seed_tokenizer_suffix())
        else:
            # Fallback: nest inside output_path (old behavior for backward compatibility)
            seed_output_path = f"{output_path}_seed"

        # Check if seed tokenizer already exists
        seed_model_path = os.path.join(seed_output_path, 'spm.model')
        if not os.path.exists(seed_model_path):
            print(f"Training seed tokenizer (vocab_size={seed_vocab_size})", file=sys.stderr)
            print(f"  Saving to: {seed_output_path}", file=sys.stderr)
            os.makedirs(seed_output_path, exist_ok=True)

            seed_sp_model = _train_sentencepiece_model(
                text_file_path=text_file_path,
                model_type=model_type,
                vocab_size=seed_vocab_size,
                special_tokens_config=special_tokens_config,
                output_path=seed_output_path,
                character_coverage=config.character_coverage,
                seed_sentencepieces_file=None  # No seeding for seed tokenizer
            )
        else:
            print(f"Seed tokenizer already exists at {seed_output_path}", file=sys.stderr)

        # Step 2: Extract raw base vocabulary counts from corpus
        # Cache in seed tokenizer dir (shared across lambda values)
        print(f"Extracting base tokenizer vocabulary", file=sys.stderr)
        base_vocab_file = os.path.join(seed_output_path, 'base_vocab_counts.txt')
        base_vocab = extract_base_vocabulary_frequencies(
            text_file_path=text_file_path,
            base_tokenizer_name=config.hf_model,
            output_file=base_vocab_file,
            filter_special_tokens=True,
            min_frequency=config.seed_min_frequency
        )
        total_base_tokens = sum(base_vocab.values())
        target_mass = total_base_tokens
        print(f"  Base vocab size: {len(base_vocab)} tokens", file=sys.stderr)
        print(f"  Total base tokens: {total_base_tokens:,}", file=sys.stderr)

        # Step 3: Extract target vocabulary from seed tokenizer, normalized to target mass scale
        # Both vocabularies are scaled to the same target_mass so they're directly comparable
        print(f"Extracting target vocabulary from seed tokenizer", file=sys.stderr)
        target_vocab = extract_target_seed_vocab(
            spm_model_path=seed_model_path,
            target_mass=target_mass,
            filter_special_tokens=True
        )
        print(f"  Target vocab size: {len(target_vocab)} tokens", file=sys.stderr)

        # Optional: convert from token-count scoring to character-length scoring
        if config.seed_score_mode == "charlength":
            print("Applying character-length weighting to vocabularies", file=sys.stderr)
            # Save original vocabs for count-based seed file output
            base_vocab_counts = dict(base_vocab)
            target_vocab_counts = dict(target_vocab)
            base_vocab = apply_character_weighting(base_vocab)
            target_vocab = apply_character_weighting(target_vocab)
            # Re-normalize target to match base character mass
            base_char_mass = int(sum(base_vocab.values()))
            target_vocab = normalize_vocab_mass(target_vocab, base_char_mass)
            print(f"  Base character mass: {base_char_mass:,}", file=sys.stderr)
        elif config.seed_score_mode != "count":
            raise ValueError(
                f"Invalid seed_score_mode: {config.seed_score_mode}. "
                "Must be 'count' or 'charlength'"
            )

        # Step 4: Merge vocabularies with lambda weighting
        print(f"Merging vocabularies with lambda={config.seed_lambda}, round_mode={config.seed_round_mode}", file=sys.stderr)
        merged_vocab = merge_vocabularies(
            base_vocab=base_vocab,
            target_vocab=target_vocab,
            lambda_weight=config.seed_lambda,
            round_mode=config.seed_round_mode
        )
        print(f"  Merged vocab size: {len(merged_vocab)} tokens", file=sys.stderr)

        # For charlength mode: also merge original counts for seed file output.
        # Character-weighted merge determines ranking (which tokens survive the
        # top-k cutoff), but the seed file should contain token counts since
        # SentencePiece uses them for initial lattice weights via ToLogProb.
        if config.seed_score_mode == "charlength":
            merged_vocab_counts = merge_vocabularies(
                base_vocab=base_vocab_counts,
                target_vocab=target_vocab_counts,
                lambda_weight=config.seed_lambda,
                round_mode=config.seed_round_mode,
            )

        # Compute overlap statistics
        base_only = set(base_vocab.keys()) - set(target_vocab.keys())
        target_only = set(target_vocab.keys()) - set(base_vocab.keys())
        both = set(base_vocab.keys()) & set(target_vocab.keys())
        print(f"  Base-only tokens: {len(base_only)}", file=sys.stderr)
        print(f"  Target-only tokens: {len(target_only)}", file=sys.stderr)
        print(f"  Shared tokens: {len(both)}", file=sys.stderr)

        # Step 5: Write merged vocabulary as seed file
        seed_file = os.path.join(output_path, 'seed_vocab.txt')
        with open(seed_file, 'w', encoding='utf-8') as f:
            if config.seed_score_mode == "charlength":
                # Rank by character-weighted score, but write token counts.
                # Pre-truncate to seed_vocab_size so SentencePiece keeps all
                # tokens (no further truncation by count values needed).
                ranked_tokens = sorted(
                    merged_vocab.items(), key=lambda x: x[1], reverse=True,
                )[:seed_vocab_size]
                for token, _ in ranked_tokens:
                    count = merged_vocab_counts.get(token, 0)
                    if count > 0:
                        f.write(f"{token}\t{count}\n")
                print(
                    f"  Pre-truncated to {len(ranked_tokens)} tokens "
                    f"(ranked by char score, values are token counts)",
                    file=sys.stderr,
                )
            else:
                # Sort by frequency (descending) for better readability
                for token, count in sorted(
                    merged_vocab.items(), key=lambda x: x[1], reverse=True,
                ):
                    f.write(f"{token}\t{count}\n")
        print(f"Hybrid seed vocabulary saved to {seed_file}", file=sys.stderr)

    # Train final SentencePiece model
    # When using seed vocabulary, set seed_sentencepiece_size to the seed
    # tokenizer's vocab size so that the merged vocabulary gets truncated to
    # that size. This makes lambda interpolation meaningful: base tokens compete
    # with corpus-derived tokens for spots in the top-k.
    # Note: if truncation removes single characters required by character_coverage,
    # SentencePiece re-adds them at finalization with penalty scores. Unlikely
    # with reasonable multiplier values since the base tokenizer covers most
    # characters already.
    sp_model = _train_sentencepiece_model(
        text_file_path=text_file_path,
        model_type=model_type,
        vocab_size=config.vocab_size,
        special_tokens_config=special_tokens_config,
        output_path=output_path,
        character_coverage=config.character_coverage,
        seed_sentencepieces_file=seed_file,
        seed_sentencepiece_size=seed_vocab_size if config.use_seed_vocabulary else None
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
        from tokenizers import SentencePieceBPETokenizer, decoders
        model_file = os.path.join(output_path, 'spm.model')
        backend_tokenizer = SentencePieceBPETokenizer.from_file(
            vocab=model_file,
            replacement="▁",
            add_prefix_space=True
        )
        # Add decoder to convert ▁ back to spaces
        backend_tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    else:
        # Unigram lacks .from_file() - must manually construct from vocab+scores
        unk_id = special_tokens_config.get('unk_id', 0)
        backend_tokenizer = _create_unigram_tokenizer(vocab_with_scores, unk_id=unk_id)

    # Copy the post-processor from base tokenizer to preserve special token behavior
    # This is critical for models like XGLM that prepend EOS to inputs
    if hasattr(base_tokenizer, '_tokenizer') and hasattr(base_tokenizer._tokenizer, 'post_processor'):
        if base_tokenizer._tokenizer.post_processor is not None:
            backend_tokenizer.post_processor = base_tokenizer._tokenizer.post_processor
            print("Copied post-processor from base tokenizer (preserves special token handling)", file=sys.stderr)

    # Wrap in PreTrainedTokenizerFast with special tokens from base model
    new_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        bos_token=base_tokenizer.bos_token,
        eos_token=base_tokenizer.eos_token,
        unk_token=base_tokenizer.unk_token,
        pad_token=base_tokenizer.pad_token,
        clean_up_tokenization_spaces=True,
    )

    # Add additional special tokens ONLY if we inherited them
    # (They're already in the SentencePiece vocab via user_defined_symbols,
    #  but PreTrainedTokenizerFast needs to know about them explicitly.
    #  This is REGISTRATION not ADDITION - we're just setting the
    #  additional_special_tokens attribute, not increasing vocab size)
    if config.inherit_additional_special_tokens:
        if hasattr(base_tokenizer, 'additional_special_tokens') and base_tokenizer.additional_special_tokens:
            new_tokenizer.add_special_tokens({
                'additional_special_tokens': base_tokenizer.additional_special_tokens
            })

    new_tokenizer.save_pretrained(output_path)
    print(f"Tokenizer saved to {output_path}", file=sys.stderr)

    # Validate vocab size and token ID contiguity
    _validate_tokenizer(new_tokenizer, config.vocab_size)

    return new_tokenizer


def _train_sentencepiece_model(
    text_file_path: str,
    model_type: str,
    vocab_size: int,
    special_tokens_config: dict,
    output_path: str,
    character_coverage: float = 1.0,
    seed_sentencepieces_file: Optional[str] = None,
    seed_sentencepiece_size: Optional[int] = None
) -> spm.SentencePieceProcessor:
    """
    Train a SentencePiece model and return the loaded processor.

    Args:
        text_file_path: Path to plain text training file (one sentence per line)
        model_type: 'bpe' or 'unigram'
        vocab_size: Target vocabulary size
        special_tokens_config: Dict of special token configs (from _extract_special_tokens)
        output_path: Directory where model files will be saved
        character_coverage: Fraction of character occurrences to cover (0-1)
        seed_sentencepieces_file: Optional path to seed vocabulary file
        seed_sentencepiece_size: Max seed pieces to keep (default: SentencePiece's 1M).
            When set, the top-k seed pieces by count are kept, forcing lambda-weighted
            counts to determine which tokens survive the cutoff.

    Returns:
        Loaded SentencePieceProcessor with the trained model
    """
    model_prefix = os.path.join(output_path, 'spm')

    # Train SentencePiece model
    # character_coverage: fraction of character occurrences to cover (rest become UNK)
    # normalization_rule_name='identity': no text normalization
    # hard_vocab_limit=True: strictly enforce vocab_size (not a soft target)
    train_args = {
        'input': text_file_path,
        'model_prefix': model_prefix,
        'model_type': model_type,
        'vocab_size': vocab_size,
        'character_coverage': character_coverage,
        'normalization_rule_name': 'identity',
        'hard_vocab_limit': True,
    }

    # Add seed vocabulary file if provided
    if seed_sentencepieces_file is not None:
        train_args['seed_sentencepieces_file'] = seed_sentencepieces_file
        if seed_sentencepiece_size is not None:
            train_args['seed_sentencepiece_size'] = seed_sentencepiece_size

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
    from tokenizers import Tokenizer, normalizers, decoders
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Metaspace

    # Initialize Unigram model with vocabulary and scores from SentencePiece
    # byte_fallback=False: use <unk> for unknown chars (matches SentencePiece training)
    unigram_model = Unigram(vocab_scores, unk_id=unk_id, byte_fallback=False)
    backend_tokenizer = Tokenizer(unigram_model)

    # Configure tokenization pipeline to match SentencePiece behavior:
    # - Empty normalizer: no text transformations (matches normalization_rule_name='identity')
    # - Metaspace pre-tokenizer: handle spaces as ▁ tokens (SentencePiece convention)
    # - Metaspace decoder: convert ▁ back to spaces when decoding
    backend_tokenizer.normalizer = normalizers.Sequence(normalizers=[])  # type: ignore
    backend_tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    backend_tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

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


FOCUS_EMBS_SUBDIR = 'focus_embs'
LEGACY_INPUT_NAME = 'focus_input_embeddings.pt'
LEGACY_OUTPUT_NAME = 'focus_output_embeddings.pt'


def _sidecar_paths(cache_dir: str, embedding_hash: str) -> tuple[str, str, str]:
    """Return (input_pt, output_pt, meta_yaml) sidecar paths for a hash."""
    sub = os.path.join(cache_dir, FOCUS_EMBS_SUBDIR)
    return (
        os.path.join(sub, f"{embedding_hash}.input.pt"),
        os.path.join(sub, f"{embedding_hash}.output.pt"),
        os.path.join(sub, f"{embedding_hash}.meta.yaml"),
    )


def _enumerate_cached_embeddings(cache_dir: str) -> list[tuple[str, Optional[str]]]:
    """
    List all cached FOCUS embedding sets under cache_dir.

    Returns a list of (input_pt_path, output_pt_path_or_None) tuples covering
    both the new focus_embs/<hash>.input.pt layout and the legacy unhashed
    files at the tokenizer-dir root.
    """
    found: list[tuple[str, Optional[str]]] = []

    sub = os.path.join(cache_dir, FOCUS_EMBS_SUBDIR)
    if os.path.isdir(sub):
        for input_pt in sorted(glob.glob(os.path.join(sub, '*.input.pt'))):
            output_pt = input_pt[: -len('.input.pt')] + '.output.pt'
            found.append((input_pt, output_pt if os.path.exists(output_pt) else None))

    legacy_input = os.path.join(cache_dir, LEGACY_INPUT_NAME)
    if os.path.exists(legacy_input):
        legacy_output = os.path.join(cache_dir, LEGACY_OUTPUT_NAME)
        found.append(
            (legacy_input, legacy_output if os.path.exists(legacy_output) else None)
        )

    return found


def resolve_cached_embedding_paths(
    cache_dir: Optional[str],
    embedding_hash: Optional[str],
    reuse_policy: Optional[str],
) -> Optional[tuple[str, Optional[str]]]:
    """
    Resolve which cached FOCUS embedding sidecar to load, if any.

    Args:
        cache_dir: Tokenizer directory containing focus_embs/ and/or legacy
            unhashed embedding files. None disables caching.
        embedding_hash: Hash for this run's mix + FOCUS knobs.
        reuse_policy: One of:
            - None: only load on exact hash match.
            - "any": accept any single cached set across both layouts; ambiguous
              if more than one exists.
            - "<hash>": load that specific sidecar; error if absent.

    Returns:
        (input_pt_path, output_pt_path_or_None) if a cache hit, else None.
    """
    if cache_dir is None:
        return None

    if reuse_policy == 'any':
        candidates = _enumerate_cached_embeddings(cache_dir)
        if len(candidates) == 0:
            return None
        if len(candidates) > 1:
            listing = "\n  ".join(p for p, _ in candidates)
            raise ValueError(
                f"focus.reuse_embeddings='any' but {len(candidates)} cached "
                f"embedding sets exist under {cache_dir}:\n  {listing}\n"
                f"Specify focus.reuse_embeddings=<hash> to disambiguate."
            )
        return candidates[0]

    if reuse_policy and reuse_policy != 'any':
        # Explicit hash request.
        input_pt, output_pt, _ = _sidecar_paths(cache_dir, reuse_policy)
        if not os.path.exists(input_pt):
            raise ValueError(
                f"focus.reuse_embeddings='{reuse_policy}' but no embeddings "
                f"found at {input_pt}"
            )
        return (input_pt, output_pt if os.path.exists(output_pt) else None)

    # Default: exact-hash match only.
    if embedding_hash is None:
        return None
    input_pt, output_pt, _ = _sidecar_paths(cache_dir, embedding_hash)
    if os.path.exists(input_pt):
        return (input_pt, output_pt if os.path.exists(output_pt) else None)
    return None


def _copy_embeddings_directly(
    source_model,
    source_token_strings: set[str],
    source_tokenizer: PreTrainedTokenizerBase,
    target_token_strings: list[str],
    has_separate_output: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build new embedding matrices by copying source embeddings for each target token.

    Used when all target tokens exist in the source vocabulary (e.g., prune-only PTEx),
    where FOCUS would crash because its novel-token matrix is empty.

    Args:
        source_model: Source pretrained model
        source_token_strings: Set of all token strings in the source vocabulary
        source_tokenizer: Tokenizer for the source model
        target_token_strings: Ordered list of token strings in the target vocabulary
        has_separate_output: Whether the model uses separate input/output embeddings

    Returns:
        Tuple of (input_embeddings, output_embeddings); output_embeddings is None
        if the model uses tied embeddings.
    """
    # Build source string → id lookup
    source_string_to_id = {
        source_tokenizer.convert_ids_to_tokens(i): i
        for i in range(len(source_tokenizer))
    }

    source_input_embeddings = source_model.get_input_embeddings().weight.detach()
    hidden_dim = source_input_embeddings.shape[1]
    target_vocab_size = len(target_token_strings)

    new_input_embeddings = torch.zeros(target_vocab_size, hidden_dim)
    missing_count = 0
    for target_id, token_str in enumerate(target_token_strings):
        source_id = source_string_to_id.get(token_str)
        if source_id is not None:
            new_input_embeddings[target_id] = source_input_embeddings[source_id]
        else:
            # Fallback to mean initialization; caller already verified this shouldn't happen
            new_input_embeddings[target_id] = source_input_embeddings.mean(dim=0)
            missing_count += 1

    if missing_count > 0:
        print(
            f"Warning: {missing_count} target tokens not found in source vocabulary; "
            f"initialized from embedding mean.",
            file=sys.stderr,
        )

    new_output_embeddings = None
    if has_separate_output:
        source_output_embeddings = source_model.get_output_embeddings().weight.detach()
        new_output_embeddings = torch.zeros(target_vocab_size, hidden_dim)
        for target_id, token_str in enumerate(target_token_strings):
            source_id = source_string_to_id.get(token_str)
            if source_id is not None:
                new_output_embeddings[target_id] = source_output_embeddings[source_id]
            else:
                new_output_embeddings[target_id] = source_output_embeddings.mean(dim=0)

    return new_input_embeddings, new_output_embeddings


def apply_focus_initialization(
    source_model,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    training_data_path: Optional[str],
    fasttext_model_min_count: int = 4,
    cache_dir: Optional[str] = None,
    embedding_hash: Optional[str] = None,
    embedding_meta: Optional[dict] = None,
    reuse_policy: Optional[str] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply FOCUS to generate new input embeddings and optionally output embeddings.

    Embedding tensors are cached as sidecars under cache_dir/focus_embs/, keyed
    by embedding_hash so that the same tokenizer can host multiple cached
    embedding sets (one per FOCUS-training mix).

    Args:
        source_model: Source pretrained model
        source_tokenizer: Tokenizer for the source model
        target_tokenizer: Target language-specific tokenizer
        training_data_path: Path to JSONL training data for FOCUS. May be None
            only when a cache hit is guaranteed by reuse_policy.
        fasttext_model_min_count: Minimum occurrences for FastText embeddings (default: 4)
        cache_dir: Directory where embeddings should be cached (typically tokenizer directory)
        embedding_hash: 8-char hash of mix + FOCUS knobs; used as sidecar key.
        embedding_meta: Provenance dict written next to a freshly-computed sidecar.
        reuse_policy: None (strict hash match), "any" (load the sole cached set
            across both new and legacy layouts), or an explicit hash string.

    Returns:
        Tuple of (input_embeddings, output_embeddings)
        output_embeddings will be None if model ties word embeddings
    """
    cached = resolve_cached_embedding_paths(cache_dir, embedding_hash, reuse_policy)
    if cached is not None:
        input_pt, output_pt = cached
        print(f"Loading cached FOCUS embeddings from {input_pt}", file=sys.stderr)
        new_input_embeddings = torch.load(input_pt, weights_only=True)

        has_separate_output = (
            hasattr(source_model.config, 'tie_word_embeddings')
            and not source_model.config.tie_word_embeddings
        )

        if has_separate_output:
            if output_pt is None:
                print(
                    f"Warning: Found cached input embeddings at {input_pt} but "
                    f"missing matching output embeddings. Regenerating both.",
                    file=sys.stderr,
                )
            else:
                new_output_embeddings = torch.load(output_pt, weights_only=True)
                print(
                    f"FOCUS embeddings loaded from cache. Vocab size: {len(target_tokenizer)}",
                    file=sys.stderr,
                )
                return new_input_embeddings, new_output_embeddings
        else:
            print(
                f"FOCUS embeddings loaded from cache. Vocab size: {len(target_tokenizer)}",
                file=sys.stderr,
            )
            return new_input_embeddings, None


    # Check whether any target tokens are absent from the source vocabulary.
    # FOCUS crashes (fastdist TypingError) when the novel-token set is empty,
    # because the cosine similarity matrix degenerates to a scalar.
    # In that case (pure prune-only tokenizer) skip FOCUS and copy directly.
    source_token_strings = {
        source_tokenizer.convert_ids_to_tokens(i)
        for i in range(len(source_tokenizer))
    }
    target_token_strings = [
        target_tokenizer.convert_ids_to_tokens(i)
        for i in range(len(target_tokenizer))
    ]
    novel_token_count = sum(1 for t in target_token_strings if t not in source_token_strings)

    has_separate_output = (
        hasattr(source_model.config, 'tie_word_embeddings')
        and not source_model.config.tie_word_embeddings
    )

    if novel_token_count == 0:
        print(
            "All target tokens exist in source vocabulary — skipping FOCUS, "
            "copying embeddings directly.",
            file=sys.stderr,
        )
        new_input_embeddings, new_output_embeddings = _copy_embeddings_directly(
            source_model=source_model,
            source_token_strings=source_token_strings,
            source_tokenizer=source_tokenizer,
            target_token_strings=target_token_strings,
            has_separate_output=has_separate_output,
        )
    else:
        if training_data_path is None:
            raise ValueError(
                "apply_focus_initialization: training_data_path is required "
                "when novel tokens are present and no cached embeddings were loaded."
            )
        print(
            f"Applying FOCUS to initialize embeddings "
            f"({novel_token_count} novel tokens)",
            file=sys.stderr,
        )

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
        if has_separate_output:
            print("Model uses separate output embeddings, applying FOCUS to output embeddings", file=sys.stderr)
            source_output_embeddings = source_model.get_output_embeddings().weight
            new_output_embeddings = FOCUS(
                source_embeddings=source_output_embeddings,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                target_training_data_path=training_data_path,
                fasttext_model_min_count=fasttext_model_min_count
            )

    print(f"Embedding initialization complete. New vocab size: {len(target_tokenizer)}", file=sys.stderr)

    # Cache the embeddings if cache_dir + embedding_hash provided
    if cache_dir is not None and embedding_hash is not None:
        input_pt, output_pt, meta_yaml = _sidecar_paths(cache_dir, embedding_hash)
        os.makedirs(os.path.dirname(input_pt), exist_ok=True)

        print(f"Saving FOCUS embeddings to {input_pt}", file=sys.stderr)
        torch.save(new_input_embeddings, input_pt)
        if new_output_embeddings is not None:
            torch.save(new_output_embeddings, output_pt)
        if embedding_meta is not None:
            with open(meta_yaml, 'w') as f:
                yaml.dump(embedding_meta, f, default_flow_style=False, sort_keys=False)

        print("FOCUS embeddings cached", file=sys.stderr)

    return new_input_embeddings, new_output_embeddings
