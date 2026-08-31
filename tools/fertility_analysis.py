"""Analyze compression gain of novel seed tokens relative to a base tokenizer.

Computes symmetric compression scores: novel seed tokens receive positive
scores for compression they provide, while base-only tokens receive negative
scores (displacement) when they are replaced by more efficient novel tokens.
Shared tokens (in both vocabs) are tracked but not scored, since their value
comes from embedding reuse rather than compression.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm
from transformers import AutoTokenizer

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lapt.tokenizer_utils import extract_target_seed_vocab


def load_corpus_word_counts(
    corpus_path: str,
    max_lines: int = 0,
    seed: int = 42,
) -> Counter:
    """Load corpus and return word type counts.

    Args:
        corpus_path: Path to SentencePiece training text file (one sentence per line).
        max_lines: Subsample this many lines (0 = all lines).
        seed: Random seed for subsampling.

    Returns:
        Counter mapping word types to their occurrence counts.
    """
    print(f"Loading corpus from {corpus_path}", file=sys.stderr)
    with open(corpus_path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  Read {len(lines):,} lines", file=sys.stderr)

    if max_lines > 0 and max_lines < len(lines):
        random.seed(seed)
        lines = random.sample(lines, max_lines)
        print(f"  Subsampled to {max_lines:,} lines (seed={seed})", file=sys.stderr)

    word_counts: Counter = Counter()
    for line in lines:
        for word in line.split():
            word_counts[word] += 1

    print(f"  Found {len(word_counts):,} unique word types", file=sys.stderr)
    return word_counts


def identify_novel_tokens(
    sp_model: spm.SentencePieceProcessor,
    base_vocab: set[str],
    min_token_length: int = 2,
) -> set[str]:
    """Identify seed tokens not present in the base vocabulary.

    Args:
        sp_model: Loaded SentencePiece seed model.
        base_vocab: Set of token strings from the base tokenizer.
        min_token_length: Minimum character length for novel tokens.

    Returns:
        Set of novel token strings.
    """
    seed_tokens = set()
    for i in range(sp_model.get_piece_size()):
        token = sp_model.id_to_piece(i)
        seed_tokens.add(token)

    novel = {
        token for token in seed_tokens
        if token not in base_vocab and len(token) >= min_token_length
    }
    print(f"  Seed vocab: {len(seed_tokens):,} tokens", file=sys.stderr)
    print(f"  Novel tokens (min length {min_token_length}): {len(novel):,}", file=sys.stderr)
    return novel


def compute_compression_scores(
    word_counts: Counter,
    sp_model: spm.SentencePieceProcessor,
    base_tokenizer: AutoTokenizer,
    novel_tokens: set[str],
    seed_vocab: set[str],
    base_vocab: set[str],
) -> tuple[dict[str, dict], dict[str, dict], dict]:
    """Compute symmetric compression scores for novel and base-only tokens.

    For each word type containing at least one novel token:
    - Novel seed tokens get +gain (compression credit)
    - Base-only tokens on the base side get -gain (displacement penalty)
    - Shared tokens on either side are tracked but not scored

    Args:
        word_counts: Counter of word type occurrences.
        sp_model: Loaded SentencePiece seed model.
        base_tokenizer: HuggingFace base tokenizer.
        novel_tokens: Set of novel seed token strings (seed - base).
        seed_vocab: Set of all seed token strings.
        base_vocab: Set of all base token strings.

    Returns:
        Tuple of (novel_scores, displacement_scores, base_side_stats) where:
        - novel_scores: dict mapping novel tokens to score dicts
        - displacement_scores: dict mapping base-only tokens to score dicts
        - base_side_stats: summary counts of base-side piece categories
    """
    def make_score_dict() -> dict:
        return {"compression_score": 0, "word_types": 0, "word_tokens": 0}

    novel_scores: dict[str, dict] = {token: make_score_dict() for token in novel_tokens}
    displacement_scores: dict[str, dict] = {}

    shared_tokens = seed_vocab & base_vocab
    base_only_tokens = base_vocab - seed_vocab

    # Track how base-side pieces break down across categories
    base_side_stats = {
        "base_only_piece_occurrences": 0,
        "shared_piece_occurrences": 0,
        "other_piece_occurrences": 0,
    }

    words_with_novel = 0

    for word, count in word_counts.items():
        seed_pieces = sp_model.encode(word, out_type=str)
        novel_in_word = [piece for piece in seed_pieces if piece in novel_tokens]
        if not novel_in_word:
            continue

        base_pieces = base_tokenizer.tokenize(word)
        gain = len(base_pieces) - len(seed_pieces)
        words_with_novel += 1

        # Credit novel tokens on the seed side
        for token in set(novel_in_word):
            novel_scores[token]["compression_score"] += gain * count
            novel_scores[token]["word_types"] += 1
            novel_scores[token]["word_tokens"] += count

        # Penalize base-only tokens on the base side
        for piece in base_pieces:
            if piece in base_only_tokens:
                base_side_stats["base_only_piece_occurrences"] += count
                if piece not in displacement_scores:
                    displacement_scores[piece] = make_score_dict()
                displacement_scores[piece]["compression_score"] -= gain * count
                displacement_scores[piece]["word_types"] += 1
                displacement_scores[piece]["word_tokens"] += count
            elif piece in shared_tokens:
                base_side_stats["shared_piece_occurrences"] += count
            else:
                base_side_stats["other_piece_occurrences"] += count

        if words_with_novel % 50000 == 0:
            print(
                f"  Processed {words_with_novel:,} words with novel tokens...",
                file=sys.stderr,
            )

    print(f"  Words containing novel tokens: {words_with_novel:,}", file=sys.stderr)
    return novel_scores, displacement_scores, base_side_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze compression gain of novel seed tokens vs base tokenizer",
    )
    parser.add_argument(
        "seed_model",
        help="Path to seed tokenizer .model file",
    )
    parser.add_argument(
        "corpus",
        help="Path to SentencePiece training text file (one sentence per line)",
    )
    parser.add_argument(
        "--base-tokenizer",
        default="facebook/xglm-564M",
        help="HuggingFace model name for base tokenizer (default: facebook/xglm-564M)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Subsample N lines from corpus (default: 0 = all lines)",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=2,
        help="Minimum character length for novel tokens (default: 2)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output TSV path (default: stdout)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling (default: 42)",
    )
    args = parser.parse_args()

    # Load corpus word counts
    word_counts = load_corpus_word_counts(
        args.corpus,
        max_lines=args.max_lines,
        seed=args.seed,
    )

    # Load seed SPM model
    print(f"Loading seed model from {args.seed_model}", file=sys.stderr)
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(args.seed_model)

    # Load base tokenizer
    print(f"Loading base tokenizer: {args.base_tokenizer}", file=sys.stderr)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    base_vocab = set(base_tokenizer.get_vocab().keys())
    print(f"  Base vocab size: {len(base_vocab):,}", file=sys.stderr)

    # Identify novel tokens and build vocab sets
    print("Identifying novel tokens...", file=sys.stderr)
    novel_tokens = identify_novel_tokens(
        sp_model,
        base_vocab,
        min_token_length=args.min_token_length,
    )

    seed_token_set = set()
    for i in range(sp_model.get_piece_size()):
        seed_token_set.add(sp_model.id_to_piece(i))

    # Get seed counts for the output
    print("Extracting seed vocabulary counts...", file=sys.stderr)
    seed_vocab_counts = extract_target_seed_vocab(args.seed_model)

    # Compute symmetric compression scores
    print("Computing compression scores...", file=sys.stderr)
    novel_scores, displacement_scores, base_side_stats = compute_compression_scores(
        word_counts, sp_model, base_tokenizer, novel_tokens, seed_token_set, base_vocab,
    )

    # Build output rows with category column
    rows = []
    for token in novel_tokens:
        score_info = novel_scores[token]
        seed_count = seed_vocab_counts.get(token, 0.0)
        word_tokens = score_info["word_tokens"]
        compression_score = score_info["compression_score"]
        avg_gain = compression_score / word_tokens if word_tokens > 0 else 0.0
        rows.append((
            "novel",
            token,
            seed_count,
            compression_score,
            score_info["word_types"],
            word_tokens,
            avg_gain,
        ))

    for token, score_info in displacement_scores.items():
        word_tokens = score_info["word_tokens"]
        compression_score = score_info["compression_score"]
        avg_gain = compression_score / word_tokens if word_tokens > 0 else 0.0
        rows.append((
            "base_only",
            token,
            0.0,
            compression_score,
            score_info["word_types"],
            word_tokens,
            avg_gain,
        ))

    # Sort by compression_score descending (novel positive at top, displaced negative at bottom)
    rows.sort(key=lambda r: r[3], reverse=True)

    # Write TSV output
    header = "category\ttoken\tseed_count\tcompression_score\tword_types\tword_tokens\tavg_gain"
    out_file = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        out_file.write(header + "\n")
        for category, token, seed_count, compression_score, word_types, word_tokens, avg_gain in rows:
            out_file.write(
                f"{category}\t{token}\t{seed_count:.1f}\t{compression_score}\t"
                f"{word_types}\t{word_tokens}\t{avg_gain:.4f}\n"
            )
    finally:
        if args.output:
            out_file.close()

    # Summary statistics to stderr
    novel_rows = [r for r in rows if r[0] == "novel"]
    displaced_rows = [r for r in rows if r[0] == "base_only"]
    top_n = 10

    print(f"\n{'='*60}", file=sys.stderr)
    print("NOVEL TOKENS (compression gain)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Total: {len(novel_rows):,}", file=sys.stderr)
    print(f"  With positive compression: {sum(1 for r in novel_rows if r[3] > 0):,}", file=sys.stderr)
    print(f"  Total gain: {sum(r[3] for r in novel_rows):,}", file=sys.stderr)
    print(f"\n  Top {top_n}:", file=sys.stderr)
    for _, token, _, compression_score, word_types, _, avg_gain in novel_rows[:top_n]:
        print(
            f"    {repr(token):30s}  score={compression_score:>12,}  "
            f"types={word_types:>6,}  avg_gain={avg_gain:.2f}",
            file=sys.stderr,
        )

    print(f"\n{'='*60}", file=sys.stderr)
    print("DISPLACED BASE-ONLY TOKENS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Total: {len(displaced_rows):,}", file=sys.stderr)
    print(
        f"  With negative score: {sum(1 for r in displaced_rows if r[3] < 0):,}",
        file=sys.stderr,
    )
    total_displacement = sum(r[3] for r in displaced_rows)
    print(f"  Total displacement: {total_displacement:,}", file=sys.stderr)
    # Show most displaced (most negative = bottom of sorted list)
    most_displaced = sorted(displaced_rows, key=lambda r: r[3])
    print(f"\n  Most displaced (bottom {top_n}):", file=sys.stderr)
    for _, token, _, compression_score, word_types, _, avg_gain in most_displaced[:top_n]:
        print(
            f"    {repr(token):30s}  score={compression_score:>12,}  "
            f"types={word_types:>6,}  avg_gain={avg_gain:.2f}",
            file=sys.stderr,
        )

    print(f"\n{'='*60}", file=sys.stderr)
    print("BASE-SIDE PIECE BREAKDOWN (in words with novel tokens)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    total_base_pieces = sum(base_side_stats.values())
    for category, count in sorted(base_side_stats.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_base_pieces * 100 if total_base_pieces > 0 else 0
        print(f"  {category}: {count:,} ({pct:.1f}%)", file=sys.stderr)

    if args.output:
        print(f"\nOutput written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
