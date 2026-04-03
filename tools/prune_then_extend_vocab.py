"""
Prune-then-Extend (PTEx) vocabulary construction.

Instead of training one tokenizer on the full multilingual corpus (where base
and novel languages compete for vocabulary slots), this tool constructs a tokenizer
from two separate sources:

1. Prune: Select base tokens from the pre-trained model by frequency coverage on
   base-language text. These tokens keep their original Unigram scores, so base-
   language text is tokenized identically to the pre-trained model.

2. Extend: Train a SentencePiece Unigram model on novel-language data only, without
   the base language competing for vocabulary slots. Combine the novel tokens with
   the pruned base tokens.

The resulting tokenizer can be used with FOCUS (via focus.tokenizer_path) for
embedding initialization: base tokens get their original embeddings copied directly,
novel tokens get FastText-based initialization.

Usage:
    python tools/prune_then_extend_vocab.py \\
        --base-text data/english_sample.txt \\
        --novel-text data/gothic_prepared/monolingual_train.txt \\
        --output tokenizers/ptex_test \\
        --target-vocab-size 32768

    # Specify novel token budget directly instead of total target size
    python tools/prune_then_extend_vocab.py \\
        --base-text data/english_sample.txt \\
        --novel-text data/gothic_prepared/monolingual_train.txt \\
        --output tokenizers/ptex_test \\
        --novel-vocab-budget 4096

    # Multiple novel-language files (concatenated internally)
    python tools/prune_then_extend_vocab.py \\
        --base-text data/english_sample_1m.txt \\
        --novel-text \\
            data/gothic_prepared/monolingual_all-codices_both-scripts_train.txt \\
            data/sagas/sagas_clean.txt \\
            data/icecorpus-texts/icepahc_12th-18th_clean.txt \\
        --output tokenizers/ptex_test \\
        --target-vocab-size 32768

    # With score refinement on full training corpus
    python tools/prune_then_extend_vocab.py \\
        --base-text data/english_sample.txt \\
        --novel-text data/gothic_prepared/monolingual_train.txt \\
        --output tokenizers/ptex_test \\
        --target-vocab-size 32768 \\
        --refine-corpus data/full_training_corpus.txt
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tokenizer_utils import (
    _create_unigram_tokenizer,
    _detect_tokenizer_algorithm,
    _extract_special_tokens,
    _train_sentencepiece_model,
    _validate_tokenizer,
)


def extract_base_vocab_with_scores(
    base_tokenizer: PreTrainedTokenizerFast,
) -> list[tuple[str, float]]:
    """
    Extract all (token, score) pairs from a Unigram base tokenizer.

    Reads the serialized backend tokenizer JSON to access Unigram log-prob scores,
    which are not exposed through the standard HuggingFace tokenizer API.

    Args:
        base_tokenizer: HuggingFace Fast tokenizer (must be Unigram)

    Returns:
        List of (token_string, log_prob_score) in token ID order
    """
    backend_data = json.loads(base_tokenizer.backend_tokenizer.to_str())
    model_vocab = backend_data['model']['vocab']
    return [(entry[0], entry[1]) for entry in model_vocab]


def count_base_token_frequencies(
    text_path: str,
    base_tokenizer: PreTrainedTokenizerFast,
    cache_dir: str | None = None,
) -> dict[int, int]:
    """
    Count base tokenizer token frequencies on base-language text, with optional caching.

    If cache_dir is provided, saves/loads token counts as JSON to avoid
    re-tokenizing the same text across runs.

    Args:
        text_path: Path to base-language text file (one line per example)
        base_tokenizer: Base model tokenizer
        cache_dir: Directory for caching token counts (None to disable)

    Returns:
        Dict mapping token ID to frequency count
    """
    # Check for cached counts
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'base_token_counts.json')
        if os.path.isfile(cache_file):
            print(f"Loading cached base-language token counts: {cache_file}", file=sys.stderr)
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            # JSON keys are strings, convert back to int
            token_counts = {int(k): v for k, v in cached.items()}
            total = sum(token_counts.values())
            print(f"  Total token instances: {total:,}", file=sys.stderr)
            print(f"  Unique tokens seen: {len(token_counts):,}", file=sys.stderr)
            return token_counts

    print(f"Tokenizing base-language text: {text_path}", file=sys.stderr)
    token_counts_counter: Counter[int] = Counter()

    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            token_ids = base_tokenizer.encode(text, add_special_tokens=False)
            token_counts_counter.update(token_ids)

    token_counts = dict(token_counts_counter)
    total = sum(token_counts.values())
    print(f"  Total token instances: {total:,}", file=sys.stderr)
    print(f"  Unique tokens seen: {len(token_counts):,}", file=sys.stderr)

    # Cache for future runs
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'base_token_counts.json')
        with open(cache_file, 'w') as f:
            json.dump(token_counts, f)
        print(f"  Cached token counts to {cache_file}", file=sys.stderr)

    return token_counts


def select_base_tokens_by_coverage(
    token_counts: dict[int, int],
    coverage_threshold: float,
) -> set[int]:
    """
    Select the minimal set of base tokens that cover a target fraction of
    base-language token instances.

    Args:
        token_counts: Dict mapping token ID to frequency count
        coverage_threshold: Fraction of token instances to cover (e.g., 0.995)

    Returns:
        Set of selected token IDs
    """
    total_instances = sum(token_counts.values())

    # Sort by frequency descending and select until coverage threshold
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0
    selected_ids: set[int] = set()

    for token_id, count in sorted_tokens:
        cumulative += count
        selected_ids.add(token_id)
        if cumulative / total_instances >= coverage_threshold:
            break

    actual_coverage = cumulative / total_instances
    print(
        f"  Selected {len(selected_ids):,} tokens "
        f"covering {actual_coverage:.4%} of instances",
        file=sys.stderr,
    )

    # Show coverage at a few thresholds for context
    print(f"\n  Coverage table:", file=sys.stderr)
    cumulative = 0
    thresholds = [0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
    threshold_idx = 0
    for i, (_, count) in enumerate(sorted_tokens):
        cumulative += count
        frac = cumulative / total_instances
        while threshold_idx < len(thresholds) and frac >= thresholds[threshold_idx]:
            print(
                f"    {thresholds[threshold_idx]:.1%} coverage: "
                f"{i + 1:,} tokens",
                file=sys.stderr,
            )
            threshold_idx += 1
        if threshold_idx >= len(thresholds):
            break

    return selected_ids


def concatenate_novel_texts(
    text_paths: list[str],
    output_dir: str,
) -> str:
    """
    Concatenate multiple novel-language text files into one for SPM training.

    If only one file is provided, returns its path directly (no copy).

    Args:
        text_paths: List of paths to novel-language text files
        output_dir: Directory to write the concatenated file

    Returns:
        Path to the (possibly concatenated) text file
    """
    if len(text_paths) == 1:
        return text_paths[0]

    os.makedirs(output_dir, exist_ok=True)
    concat_path = os.path.join(output_dir, 'novel_combined.txt')
    total_lines = 0
    print(f"\nConcatenating {len(text_paths)} novel text files:", file=sys.stderr)
    with open(concat_path, 'w', encoding='utf-8') as out:
        for path in text_paths:
            line_count = 0
            with open(path, 'r', encoding='utf-8') as inp:
                for line in inp:
                    out.write(line)
                    line_count += 1
            total_lines += line_count
            print(f"  {path}: {line_count:,} lines", file=sys.stderr)
    print(f"  Total: {total_lines:,} lines → {concat_path}", file=sys.stderr)
    return concat_path


def train_novel_spm(
    text_path: str,
    vocab_size: int,
    output_dir: str,
    base_tokenizer: PreTrainedTokenizerFast,
    character_coverage: float = 1.0,
) -> spm.SentencePieceProcessor:
    """
    Train a SentencePiece Unigram model on novel-language text.

    Args:
        text_path: Path to novel-language text (one line per example)
        vocab_size: Target vocabulary size for the novel SPM
        output_dir: Directory to save the trained SPM model
        base_tokenizer: Base tokenizer (for special token extraction)
        character_coverage: SentencePiece character coverage (default: 1.0)

    Returns:
        Trained SentencePieceProcessor
    """
    special_tokens_config = _extract_special_tokens(
        base_tokenizer,
        inherit_additional=False,
    )

    print(
        f"\nTraining novel-language SPM (vocab_size={vocab_size})",
        file=sys.stderr,
    )
    os.makedirs(output_dir, exist_ok=True)
    sp_model = _train_sentencepiece_model(
        text_file_path=text_path,
        model_type='unigram',
        vocab_size=vocab_size,
        special_tokens_config=special_tokens_config,
        output_path=output_dir,
        character_coverage=character_coverage,
    )
    return sp_model


def extract_novel_tokens(
    sp_model: spm.SentencePieceProcessor,
) -> list[tuple[str, float]]:
    """
    Extract non-special tokens and scores from a trained SentencePiece model.

    Args:
        sp_model: Trained SentencePieceProcessor

    Returns:
        List of (token_string, log_prob_score) for normal (non-special) tokens
    """
    tokens = []
    for i in range(sp_model.get_piece_size()):
        token = sp_model.id_to_piece(i)
        score = sp_model.get_score(i)

        # Skip special tokens (score == 0 in SPM convention)
        if score == 0.0:
            continue
        if token in ('<unk>', '<s>', '</s>', '<pad>'):
            continue

        tokens.append((token, score))

    return tokens


def estimate_corpus_scores(
    tokenizer: PreTrainedTokenizerFast,
    corpus_path: str,
) -> dict[int, float]:
    """
    Estimate Unigram log-prob scores from corpus token frequencies.

    Tokenizes the corpus, counts token frequencies, and converts to
    log probabilities. This produces scores that are coherent (sum to 1
    in exp-space) and calibrated to the actual training distribution.

    Args:
        tokenizer: Tokenizer to use for segmentation
        corpus_path: Path to text corpus (one sentence per line)

    Returns:
        Dict mapping token ID to log-probability score
    """
    token_counts: Counter[int] = Counter()
    print(f"\nEstimating corpus-based scores:", file=sys.stderr)
    print(f"  Corpus: {corpus_path}", file=sys.stderr)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_counts.update(token_ids)

    total = sum(token_counts.values())
    print(f"  Total token instances: {total:,}", file=sys.stderr)
    print(f"  Unique tokens used: {len(token_counts):,}", file=sys.stderr)

    # Convert counts to log probabilities
    scores: dict[int, float] = {}
    for token_id, count in token_counts.items():
        scores[token_id] = math.log(count / total)

    # Tokens with zero count get a very low score
    vocab_size = len(tokenizer)
    zero_count = vocab_size - len(token_counts)
    if zero_count > 0:
        print(
            f"  Tokens with zero count: {zero_count:,} (assigned score -30.0)",
            file=sys.stderr,
        )

    return scores


def build_combined_tokenizer(
    base_tokenizer: PreTrainedTokenizerFast,
    base_vocab_scores: list[tuple[str, float]],
    selected_base_ids: set[int],
    novel_tokens: list[tuple[str, float]],
    target_vocab_size: int,
) -> PreTrainedTokenizerFast:
    """
    Combine base-selected tokens and novel tokens into a single Unigram tokenizer.

    Token ordering in the combined vocabulary:
    1. Special tokens (BOS, EOS, UNK, PAD) with score 0.0
    2. Base-selected tokens with original XGLM Unigram scores
    3. Novel tokens with scores from novel SPM training

    Novel tokens that duplicate a base-selected token string are skipped.

    Args:
        base_tokenizer: Original base model tokenizer
        base_vocab_scores: Full base vocabulary as (token, score) pairs
        selected_base_ids: Token IDs selected for base-language coverage
        novel_tokens: (token, score) pairs from novel SPM
        target_vocab_size: Desired total vocabulary size

    Returns:
        Combined HuggingFace Fast tokenizer
    """
    # Identify special tokens and their positions in the base tokenizer
    special_token_strings = set()
    special_entries: list[tuple[str, float]] = []

    for attr in ('bos_token', 'eos_token', 'unk_token', 'pad_token'):
        token_str = getattr(base_tokenizer, attr, None)
        if token_str and token_str not in special_token_strings:
            token_id = base_tokenizer.convert_tokens_to_ids(token_str)
            score = base_vocab_scores[token_id][1]
            special_entries.append((token_str, score))
            special_token_strings.add(token_str)

    # Collect base-selected tokens (excluding specials)
    base_entries: list[tuple[str, float]] = []
    base_token_strings = set(special_token_strings)
    for token_id in sorted(selected_base_ids):
        token_str, score = base_vocab_scores[token_id]
        if token_str in base_token_strings:
            continue
        base_entries.append((token_str, score))
        base_token_strings.add(token_str)

    # Collect novel tokens, skipping any that overlap with base selection
    novel_entries: list[tuple[str, float]] = []
    overlap_count = 0
    for token_str, score in novel_tokens:
        if token_str in base_token_strings:
            overlap_count += 1
            continue
        novel_entries.append((token_str, score))

    # Combine: specials + base + novel
    combined = special_entries + base_entries + novel_entries

    # Report
    print(f"\nCombined vocabulary composition:", file=sys.stderr)
    print(f"  Special tokens: {len(special_entries)}", file=sys.stderr)
    print(f"  Base tokens (retained): {len(base_entries)}", file=sys.stderr)
    print(f"  Novel tokens (new languages): {len(novel_entries)}", file=sys.stderr)
    print(f"  Novel-base overlap (skipped): {overlap_count}", file=sys.stderr)
    print(f"  Total: {len(combined)}", file=sys.stderr)

    if len(combined) != target_vocab_size:
        print(
            f"\n  WARNING: Combined size {len(combined)} != "
            f"target {target_vocab_size}",
            file=sys.stderr,
        )
        if len(combined) < target_vocab_size:
            print(
                f"  The novel SPM produced fewer unique tokens than expected. "
                f"This can happen when overlap is larger than anticipated.",
                file=sys.stderr,
            )
        else:
            print(
                f"  Truncating to target vocab size (dropping lowest-scored "
                f"novel tokens)",
                file=sys.stderr,
            )
            # Keep all specials and base tokens, truncate novel tokens
            max_novel = target_vocab_size - len(special_entries) - len(base_entries)
            novel_entries = novel_entries[:max_novel]
            combined = special_entries + base_entries + novel_entries

    # Determine unk_id from the combined list
    unk_token = base_tokenizer.unk_token
    unk_id = 0
    for i, (token_str, _) in enumerate(combined):
        if token_str == unk_token:
            unk_id = i
            break

    # Build HuggingFace tokenizer from combined vocab+scores
    backend_tokenizer = _create_unigram_tokenizer(combined, unk_id=unk_id)

    # Copy post-processor from base tokenizer (critical for XGLM's EOS prepending)
    if (
        hasattr(base_tokenizer, '_tokenizer')
        and hasattr(base_tokenizer._tokenizer, 'post_processor')
        and base_tokenizer._tokenizer.post_processor is not None
    ):
        backend_tokenizer.post_processor = base_tokenizer._tokenizer.post_processor
        print(
            "Copied post-processor from base tokenizer",
            file=sys.stderr,
        )

    # Wrap in PreTrainedTokenizerFast
    new_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        bos_token=base_tokenizer.bos_token,
        eos_token=base_tokenizer.eos_token,
        unk_token=base_tokenizer.unk_token,
        pad_token=base_tokenizer.pad_token,
        clean_up_tokenization_spaces=True,
    )

    return new_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Prune-then-Extend (PTEx) vocabulary construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--base-text',
        required=True,
        help=(
            'Path to base-language text file (one line per example). '
            'Used to select which base tokenizer tokens to keep by frequency.'
        ),
    )
    parser.add_argument(
        '--novel-text',
        required=True,
        nargs='+',
        help='Path(s) to novel-language text file(s) (one line per example)',
    )
    parser.add_argument(
        '--base-tokenizer',
        default='facebook/xglm-1.7B',
        help='Base model tokenizer (default: facebook/xglm-1.7B)',
    )
    parser.add_argument(
        '--target-vocab-size',
        type=int,
        default=None,
        help=(
            'Total combined vocabulary size. If neither this nor '
            '--novel-vocab-budget is provided, defaults to 32768. '
            'If both are given, this takes precedence.'
        ),
    )
    parser.add_argument(
        '--novel-vocab-budget',
        type=int,
        default=None,
        help=(
            'Number of novel-language token slots in the combined vocabulary. '
            'Mutually usable with --target-vocab-size; if both are given, '
            '--target-vocab-size takes precedence and this value is ignored.'
        ),
    )
    parser.add_argument(
        '--base-coverage',
        type=float,
        default=0.995,
        help='Fraction of base-language token instances to cover (default: 0.995)',
    )
    parser.add_argument(
        '--character-coverage',
        type=float,
        default=1.0,
        help='SentencePiece character coverage for novel languages (default: 1.0)',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for the combined tokenizer',
    )
    parser.add_argument(
        '--refine-corpus',
        default=None,
        help=(
            'Path to corpus for score re-estimation. If provided, tokenizes '
            'the corpus, counts token frequencies, and converts to log '
            'probabilities — producing coherent scores across base and novel '
            'tokens. Typically the full training corpus (all languages). '
            'Character coverage is handled by the novel-language SPM training.'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed (default: 1)',
    )
    args = parser.parse_args()

    # Default to historical target size when neither sizing argument is given
    if args.target_vocab_size is None and args.novel_vocab_budget is None:
        args.target_vocab_size = 32768

    # Validate inputs
    if not os.path.isfile(args.base_text):
        print(f"Error: Base text file not found: {args.base_text}", file=sys.stderr)
        sys.exit(1)
    for novel_path in args.novel_text:
        if not os.path.isfile(novel_path):
            print(f"Error: Novel text file not found: {novel_path}", file=sys.stderr)
            sys.exit(1)

    # Load base tokenizer and verify it's Unigram
    print(f"Loading base tokenizer: {args.base_tokenizer}", file=sys.stderr)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    algorithm = _detect_tokenizer_algorithm(base_tokenizer)
    if algorithm != 'unigram':
        print(
            f"Error: Base tokenizer uses {algorithm}, but this tool only supports "
            f"Unigram. BPE would require a different merge strategy.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"  Algorithm: {algorithm}, vocab size: {len(base_tokenizer):,}",
        file=sys.stderr,
    )

    # Step 1: Extract base vocab scores and select tokens for base-language coverage
    base_vocab_scores = extract_base_vocab_with_scores(base_tokenizer)

    token_counts = count_base_token_frequencies(
        text_path=args.base_text,
        base_tokenizer=base_tokenizer,
        cache_dir=args.output,
    )
    selected_ids = select_base_tokens_by_coverage(
        token_counts=token_counts,
        coverage_threshold=args.base_coverage,
    )

    # Compute novel language vocab budget
    # Count special tokens that will be included
    special_token_strings = set()
    for attr in ('bos_token', 'eos_token', 'unk_token', 'pad_token'):
        token_str = getattr(base_tokenizer, attr, None)
        if token_str:
            special_token_strings.add(token_str)
    num_special = len(special_token_strings)

    # Remove special token IDs from the selected set (they're added separately)
    for attr in ('bos_token_id', 'eos_token_id', 'unk_token_id', 'pad_token_id'):
        token_id = getattr(base_tokenizer, attr, None)
        if token_id is not None:
            selected_ids.discard(token_id)

    # Resolve novel_vocab_budget and target_vocab_size from the arguments.
    # --target-vocab-size takes precedence when both are given; if only
    # --novel-vocab-budget is given, the target size is derived from it.
    if args.target_vocab_size is not None:
        target_vocab_size = args.target_vocab_size
        novel_vocab_budget = target_vocab_size - len(selected_ids) - num_special
        if args.novel_vocab_budget is not None:
            implied_total = num_special + len(selected_ids) + args.novel_vocab_budget
            if implied_total != target_vocab_size:
                print(
                    f"NOTE: --novel-vocab-budget implies total size {implied_total:,}, "
                    f"but --target-vocab-size={target_vocab_size:,} takes precedence.",
                    file=sys.stderr,
                )
    else:
        novel_vocab_budget = args.novel_vocab_budget
        target_vocab_size = num_special + len(selected_ids) + novel_vocab_budget

    if novel_vocab_budget <= 0:
        print(
            f"Error: No vocabulary budget left for novel languages. "
            f"Base selected {len(selected_ids)} + {num_special} special = "
            f"{len(selected_ids) + num_special}, "
            f"but target is {target_vocab_size}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"\nVocabulary budget:",
        file=sys.stderr,
    )
    print(f"  Target total: {target_vocab_size:,}", file=sys.stderr)
    print(f"  Special tokens: {num_special}", file=sys.stderr)
    print(f"  Base tokens (retained): {len(selected_ids):,}", file=sys.stderr)
    print(f"  Novel language budget: {novel_vocab_budget:,}", file=sys.stderr)

    # Step 2: Train novel-language SPM
    # Train with extra headroom (1.5x budget) because some novel tokens will
    # overlap with base selection and get dropped. After combining, we truncate
    # or fill to reach the exact target size.
    spm_vocab_size = int(novel_vocab_budget * 1.5) + num_special
    novel_spm_dir = os.path.join(args.output, 'novel_spm')

    # Concatenate multiple novel text files if needed
    novel_text_path = concatenate_novel_texts(args.novel_text, novel_spm_dir)

    sp_model = train_novel_spm(
        text_path=novel_text_path,
        vocab_size=spm_vocab_size,
        output_dir=novel_spm_dir,
        base_tokenizer=base_tokenizer,
        character_coverage=args.character_coverage,
    )

    # Extract non-special tokens from novel SPM
    novel_tokens = extract_novel_tokens(sp_model)
    print(
        f"  Novel tokens extracted: {len(novel_tokens):,} "
        f"(excluding special tokens)",
        file=sys.stderr,
    )

    # Step 3: Combine into final tokenizer
    # Build the base token string set for overlap detection
    base_token_strings = set()
    for attr in ('bos_token', 'eos_token', 'unk_token', 'pad_token'):
        token_str = getattr(base_tokenizer, attr, None)
        if token_str:
            base_token_strings.add(token_str)
    for token_id in selected_ids:
        base_token_strings.add(base_vocab_scores[token_id][0])

    # Count how many novel tokens overlap with base selection
    novel_unique = [(t, s) for t, s in novel_tokens if t not in base_token_strings]
    overlap_count = len(novel_tokens) - len(novel_unique)
    print(f"  Novel-base overlap: {overlap_count:,} tokens", file=sys.stderr)
    print(f"  Novel unique tokens available: {len(novel_unique):,}", file=sys.stderr)

    # If we don't have enough novel tokens to fill the budget, backfill with
    # additional base tokens (next most frequent ones not yet selected)
    slots_for_novel = target_vocab_size - len(selected_ids) - num_special
    if len(novel_unique) < slots_for_novel:
        shortfall = slots_for_novel - len(novel_unique)
        print(
            f"  Shortfall: {shortfall} slots to fill with additional base tokens",
            file=sys.stderr,
        )
        # Add more base tokens by frequency order
        all_base_by_freq = sorted(
            token_counts.items(), key=lambda x: x[1], reverse=True,
        )
        added = 0
        for token_id, _ in all_base_by_freq:
            if added >= shortfall:
                break
            token_str = base_vocab_scores[token_id][0]
            if token_id not in selected_ids and token_str not in base_token_strings:
                selected_ids.add(token_id)
                base_token_strings.add(token_str)
                added += 1
        print(f"  Added {added} additional base tokens", file=sys.stderr)

    new_tokenizer = build_combined_tokenizer(
        base_tokenizer=base_tokenizer,
        base_vocab_scores=base_vocab_scores,
        selected_base_ids=selected_ids,
        novel_tokens=novel_tokens,
        target_vocab_size=target_vocab_size,
    )

    # Optional: re-estimate scores from corpus token frequencies
    if args.refine_corpus:
        if not os.path.isfile(args.refine_corpus):
            print(
                f"Error: Refine corpus not found: {args.refine_corpus}",
                file=sys.stderr,
            )
            sys.exit(1)

        corpus_scores = estimate_corpus_scores(new_tokenizer, args.refine_corpus)
        current_vocab = extract_base_vocab_with_scores(new_tokenizer)
        refined_vocab: list[tuple[str, float]] = []
        for token_id, (token_str, old_score) in enumerate(current_vocab):
            # Keep original score of 0.0 for special tokens
            if token_str in ('<unk>', '<s>', '</s>', '<pad>'):
                refined_vocab.append((token_str, old_score))
            else:
                refined_vocab.append((token_str, corpus_scores.get(token_id, -30.0)))

        unk_token = base_tokenizer.unk_token
        unk_id = 0
        for i, (token_str, _) in enumerate(refined_vocab):
            if token_str == unk_token:
                unk_id = i
                break

        backend_tokenizer = _create_unigram_tokenizer(refined_vocab, unk_id=unk_id)
        if (
            hasattr(base_tokenizer, '_tokenizer')
            and hasattr(base_tokenizer._tokenizer, 'post_processor')
            and base_tokenizer._tokenizer.post_processor is not None
        ):
            backend_tokenizer.post_processor = (
                base_tokenizer._tokenizer.post_processor
            )

        new_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend_tokenizer,
            bos_token=base_tokenizer.bos_token,
            eos_token=base_tokenizer.eos_token,
            unk_token=base_tokenizer.unk_token,
            pad_token=base_tokenizer.pad_token,
            clean_up_tokenization_spaces=True,
        )
        print(
            f"  Rebuilt tokenizer with corpus-estimated scores "
            f"(vocab size: {len(new_tokenizer):,})",
            file=sys.stderr,
        )

    # Validate and save
    actual_size = len(new_tokenizer)
    if actual_size != target_vocab_size:
        print(
            f"\nWARNING: Final vocab size {actual_size:,} != "
            f"target {target_vocab_size:,}",
            file=sys.stderr,
        )
    _validate_tokenizer(new_tokenizer, actual_size)

    os.makedirs(args.output, exist_ok=True)
    new_tokenizer.save_pretrained(args.output)
    print(f"\nTokenizer saved to {args.output}", file=sys.stderr)

    # Final summary
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"PTEx tokenizer built successfully", file=sys.stderr)
    print(f"  Output: {args.output}", file=sys.stderr)
    print(f"  Vocab size: {len(new_tokenizer):,}", file=sys.stderr)
    print(f"  Base tokenizer: {args.base_tokenizer}", file=sys.stderr)
    print(f"  Base-language coverage: {args.base_coverage}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)


if __name__ == '__main__':
    main()
