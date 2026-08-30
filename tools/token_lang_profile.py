"""Compute per-token language affinity scores for embedding color-coding.

For each token in a tokenizer's vocabulary, computes P(language | token) from
corpus frequencies, then maps the distribution to an RGB color:
  - Hue: weighted blend of language colors by P(lang|token)
  - Saturation: 1 - normalized_entropy (distinctive tokens are vivid,
    tokens shared equally across languages fade to the background color)

Output is a JSON file with the score matrix and pre-computed CSS color strings,
suitable for use as marker colors in plotly-based visualization tools.

Usage:
    python tools/token_lang_profile.py \\
        --tokenizer tokenizers/old_germanic/xglm564m_focus-v32k-s5m/ \\
        --corpus eng:data/english.txt \\
        --corpus got:data/gothic.txt \\
        --corpus non:data/norse.txt \\
        --color eng:#2E86AB --color got:#A23B72 --color non:#F18F01 \\
        -o token_colors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

DEFAULT_PALETTE = [
    "#2E86AB",
    "#A23B72",
    "#F18F01",
    "#3B8132",
    "#D63230",
    "#F7B32B",
    "#5C4D7D",
    "#1B998B",
]


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert a hex color string to an (R, G, B) tuple."""
    hex_str = hex_str.lstrip("#")
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def count_corpus_tokens(
    tokenizer: AutoTokenizer,
    corpus_path: str,
) -> Counter:
    """Tokenize a plaintext corpus and return token ID counts."""
    counts: Counter[int] = Counter()
    with open(corpus_path) as f:
        for line in f:
            text = line.rstrip("\n")
            if text:
                ids = tokenizer.encode(text, add_special_tokens=False)
                counts.update(ids)
    return counts


def compute_scores(
    tokenizer: AutoTokenizer,
    corpora: dict[str, str],
) -> np.ndarray:
    """Compute P(language | token) for every token in the vocabulary.

    Args:
        tokenizer: HuggingFace tokenizer.
        corpora: Mapping from language name to plaintext corpus path.

    Returns:
        Array of shape (vocab_size, n_langs). Each row sums to 1 for tokens
        seen in at least one corpus, and is all-zeros for unseen tokens.
    """
    languages = list(corpora.keys())
    vocab_size = len(tokenizer)
    n_langs = len(languages)

    # P(token | language): token frequency normalized by corpus size
    p_token_given_lang = np.zeros((vocab_size, n_langs))

    for j, lang in enumerate(languages):
        counts = count_corpus_tokens(tokenizer, corpora[lang])
        total = sum(counts.values())
        if total == 0:
            print(f"WARNING: no tokens found in corpus for {lang}", file=sys.stderr)
            continue
        for token_id, count in counts.items():
            if token_id < vocab_size:
                p_token_given_lang[token_id, j] = count / total
        print(f"  {lang}: {total:,} tokens from {corpora[lang]}", file=sys.stderr)

    # Bayes' rule with uniform prior: P(lang|token) ∝ P(token|lang)
    row_sums = p_token_given_lang.sum(axis=1, keepdims=True)
    scores = np.divide(
        p_token_given_lang,
        row_sums,
        out=np.zeros_like(p_token_given_lang),
        where=row_sums > 0,
    )
    return scores


def scores_to_colors(
    scores: np.ndarray,
    lang_rgbs: np.ndarray,
    background: tuple[int, int, int] = (255, 255, 255),
) -> list[str]:
    """Map P(lang|token) distributions to CSS rgb() color strings.

    Args:
        scores: (vocab_size, n_langs) array of P(lang|token).
        lang_rgbs: (n_langs, 3) array of RGB values per language.
        background: RGB color for fully desaturated (max-entropy) tokens.

    Returns:
        List of "rgb(r,g,b)" strings, one per token.
    """
    n_langs = scores.shape[1]
    bg = np.array(background, dtype=np.float64)

    # Weighted blend of language colors
    blended = scores @ lang_rgbs.astype(np.float64)

    # Entropy-based saturation: low entropy → vivid, high entropy → background
    with np.errstate(divide="ignore", invalid="ignore"):
        log_scores = np.where(scores > 0, np.log(scores), 0.0)
    entropy = -np.sum(scores * log_scores, axis=1)
    max_entropy = np.log(n_langs) if n_langs > 1 else 1.0
    saturation = 1.0 - entropy / max_entropy

    # Tokens absent from all corpora: force to background
    unseen = scores.sum(axis=1) == 0
    saturation[unseen] = 0.0
    blended[unseen] = bg

    # Lerp: background → blended by saturation
    final = bg + saturation[:, None] * (blended - bg)
    final = np.clip(final, 0, 255).astype(int)

    return [f"rgb({r},{g},{b})" for r, g, b in final]


def parse_kv_spec(spec: str, sep: str = ":") -> tuple[str, str]:
    """Parse a 'key:value' string, raising on missing separator."""
    if sep not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected KEY{sep}VALUE format, got {spec!r}"
        )
    key, value = spec.split(sep, 1)
    return key, value


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-token language affinity scores and "
            "pre-computed RGB colors for embedding visualization."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer (local directory or model name).",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG:PATH",
        help=(
            "Language corpus as LANG:PATH, where PATH is a plaintext file "
            "(one example per line). Repeatable for each language."
        ),
    )
    parser.add_argument(
        "--color",
        action="append",
        metavar="LANG:#HEX",
        help=(
            "Language color as LANG:#HEX (e.g., eng:#2E86AB). "
            "Unspecified languages get auto-assigned from a default palette."
        ),
    )
    parser.add_argument(
        "--background",
        default="#FFFFFF",
        help="Background color for max-entropy (shared) tokens. Default: white.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    # Parse corpus specs
    corpora: dict[str, str] = {}
    for spec in args.corpus:
        lang, path = parse_kv_spec(spec)
        corpora[lang] = path
    languages = list(corpora.keys())

    # Parse color specs, fill defaults for unspecified
    lang_colors: dict[str, str] = {}
    if args.color:
        for spec in args.color:
            lang, color = parse_kv_spec(spec)
            lang_colors[lang] = color
    palette_idx = 0
    for lang in languages:
        if lang not in lang_colors:
            lang_colors[lang] = DEFAULT_PALETTE[palette_idx % len(DEFAULT_PALETTE)]
            palette_idx += 1

    lang_rgbs = np.array([hex_to_rgb(lang_colors[lang]) for lang in languages])
    bg_rgb = hex_to_rgb(args.background)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    print(f"  vocab_size={vocab_size}", file=sys.stderr)

    # Compute P(lang|token)
    print("Counting token frequencies...", file=sys.stderr)
    scores = compute_scores(tokenizer, corpora)

    # Compute colors
    colors = scores_to_colors(scores, lang_rgbs, background=bg_rgb)

    # Print summary
    unseen_count = int((scores.sum(axis=1) == 0).sum())
    print("\nSummary:", file=sys.stderr)
    for j, lang in enumerate(languages):
        dominant = (scores.argmax(axis=1) == j) & (scores.sum(axis=1) > 0)
        print(
            f"  {lang} ({lang_colors[lang]}): {int(dominant.sum()):,} tokens dominated",
            file=sys.stderr,
        )
    print(f"  Unseen (no corpus): {unseen_count:,} / {vocab_size:,}", file=sys.stderr)

    # Write output
    output = {
        "tokenizer": args.tokenizer,
        "languages": languages,
        "lang_colors": {lang: lang_colors[lang] for lang in languages},
        "background": args.background,
        "scores": scores.tolist(),
        "colors": colors,
    }
    with open(args.output, "w") as f:
        json.dump(output, f)
    print(f"\nWrote {args.output} ({vocab_size} tokens × {len(languages)} langs)", file=sys.stderr)


if __name__ == "__main__":
    main()
