"""Unsupervised scoring of canonical Gothic affixes (word-edge n-grams).

Identifies which word edges behave like affixes — suffixes *and* prefixes —
*without* any grammar or glossing (the portability constraint; see
`.claude/gothic/oov_robustness_augmentation.md`). The signal is a length-
normalized over-representation ratio:

    ratio(a) = observed(a) / expected(a)

where ``observed(a)`` is the number of word *types* carrying the k-character
affix ``a`` at the relevant edge, and ``expected(a)`` is what you'd see if those
k characters were drawn independently from their **position-specific** unigram
frequencies (the frequency of each character at the last position, second-to-
last, etc. for suffixes; first, second, etc. for prefixes). A real affix's
characters co-occur at the word edge far more than independent per-position rates
predict, so it scores high; an incidental edge scores ~1.

Two properties make this the right null:
  * **Length-normalized.** ``expected`` is a product of k probabilities, so it
    shrinks with k; a longer affix does not need a high raw count to stand out.
    (A length-1 affix scores exactly 1 by construction — a single edge character
    cannot be over-represented relative to its own frequency — so real signal
    only appears at k >= 2.)
  * **Morphology-blind null.** Independent per-position characters capture raw
    phonotactics but not the joint regularity of a multi-character morpheme, so
    the ratio isolates exactly the morpheme-like dependence.

Ranking by the raw ratio alone over-rewards long, rare, cohesive edges (stem
chunks and lexical roots that happen to be tight) and buries the short, highly
productive affixes whose characters are individually common. So candidates are
ranked by **frequency-weighted mutual information**,

    weighted_mi(a) = observed(a) * log(ratio(a)),

which balances over-representation against productivity. This produces no case /
number / class labels — it is the intended unsupervised notion of "predictable
edge". The number of real affixes differs by direction and language (Gothic has
far fewer productive prefixes than suffixes), so ``top_n`` is a per-direction
knob the caller must set.
"""

import argparse
import math
from collections import Counter
from dataclasses import dataclass

from gothic.oov_augmentation.vocab import content_vocab, load_training_vocab


@dataclass
class AffixScore:
    """A scored word-edge n-gram candidate.

    Attributes:
        affix: The word-edge character n-gram.
        side: ``"suffix"`` or ``"prefix"``.
        length: Number of characters in ``affix``.
        observed: Number of word types carrying ``affix`` at that edge.
        expected: Expected type count under the position-unigram null.
        ratio: ``observed / expected`` — the over-representation score.
        weighted_mi: ``observed * log(ratio)`` — the selection score. Zero when
            ratio <= 1.
    """

    affix: str
    side: str
    length: int
    observed: int
    expected: float
    ratio: float
    weighted_mi: float


def score_affixes(
    word_types: list[str],
    side: str = "suffix",
    max_length: int = 4,
    min_count: int = 5,
) -> list[AffixScore]:
    """Score word-edge n-grams by over-representation against a position null.

    Args:
        word_types: Distinct romanized content words (function words pre-removed).
        side: ``"suffix"`` (word-final) or ``"prefix"`` (word-initial).
        max_length: Longest affix to consider.
        min_count: Discard affixes seen in fewer than this many types (noise).

    Returns:
        AffixScores sorted by descending weighted MI.
    """
    scores: list[AffixScore] = []
    for length in range(1, max_length + 1):
        eligible = [word for word in word_types if len(word) >= length]
        total = len(eligible)
        if total == 0:
            continue

        # position_counts[j] counts characters at the j-th position in from the
        # relevant edge (j=0 is the edge character: last char for a suffix, first
        # char for a prefix).
        position_counts = [Counter() for _ in range(length)]
        affix_counts: Counter = Counter()
        for word in eligible:
            if side == "suffix":
                affix_counts[word[-length:]] += 1
                for j in range(length):
                    position_counts[j][word[-(j + 1)]] += 1
            else:
                affix_counts[word[:length]] += 1
                for j in range(length):
                    position_counts[j][word[j]] += 1

        for affix, observed in affix_counts.items():
            if observed < min_count:
                continue
            expected_probability = 1.0
            for j in range(length):
                character = affix[-(j + 1)] if side == "suffix" else affix[j]
                expected_probability *= position_counts[j][character] / total
            expected = expected_probability * total
            ratio = observed / expected if expected > 0 else float("inf")
            weighted_mi = observed * math.log(ratio) if ratio > 1.0 else 0.0
            scores.append(
                AffixScore(affix, side, length, observed, expected, ratio, weighted_mi)
            )

    scores.sort(key=lambda score: score.weighted_mi, reverse=True)
    return scores


def _edge_nested(shorter: str, longer: str, side: str) -> bool:
    """Whether ``shorter`` is a strict edge-substring of ``longer`` on ``side``."""
    if len(shorter) >= len(longer):
        return False
    return longer.endswith(shorter) if side == "suffix" else longer.startswith(shorter)


def find_elbow(values: list[float]) -> int:
    """Find the elbow of a descending curve by max distance below its chord.

    The dependency-free geometric core of the Kneedle algorithm (Satopää et al.,
    2011): normalize the curve to the unit square, draw the chord between the
    endpoints, and return the count up to the point of maximum distance below it.

    Note (validated on Gothic affix curves): this detects the single sharpest
    bend — a *dominance* elbow. Because the weighted-MI curve has a long, near-
    zero noise tail, call this on a **bounded window** of the top candidates, not
    the full scored list, or the tail compresses the head and the elbow slides
    right. Even bounded, the cut is aggressive (keeps only clearly-dominant
    affixes); for graft *coverage* a manual ``top_n`` is usually preferable.

    Args:
        values: Scores in descending order (e.g. weighted MI).

    Returns:
        The number of leading elements to keep (1-based count of the elbow point).
    """
    n = len(values)
    if n <= 2:
        return n
    y_first, y_last = values[0], values[-1]
    span = y_first - y_last
    if span == 0:
        return n

    best_index = 0
    best_gap = -1.0
    for index in range(n):
        x = index / (n - 1)
        y = (values[index] - y_last) / span
        gap = (1.0 - x) - y
        if gap > best_gap:
            best_gap = gap
            best_index = index
    return best_index + 1


def select_affixes(
    scores: list[AffixScore],
    top_n: int | None = 40,
    min_weighted_mi: float = 0.0,
    dedup_nested: bool = False,
    elbow_window: int = 50,
) -> list[str]:
    """Pick a canonical affix set from scored affixes.

    Args:
        scores: Output of ``score_affixes`` (already weighted-MI-sorted).
        top_n: Keep at most this many affixes. If ``None``, the count is chosen
            automatically by the weighted-MI **elbow** (``find_elbow``) over the
            top ``elbow_window`` filtered candidates. The elbow is robust but
            aggressive (dominance, not coverage) — see ``find_elbow``.
        min_weighted_mi: Require at least this weighted-MI score.
        dedup_nested: If True, drop an affix that is edge-nested with an
            already-kept (higher-weighted-MI) affix — keeping only the stronger of
            each nested pair. This removes spurious fragments (e.g. prefix ``fr``
            under ``fra``, or ``ga``'s ``gal``/``gas`` extensions). Use on the
            **prefix** side only: on the suffix side the short variants (``-nds``,
            ``-ns``) have independent coverage and must be kept.
        elbow_window: When ``top_n`` is None, restrict elbow detection to this
            many leading candidates (the curve's noise tail otherwise displaces
            the elbow).

    Returns:
        Affix strings, most affix-like first.
    """
    filtered: list[AffixScore] = []
    for score in scores:
        if score.weighted_mi < min_weighted_mi:
            continue
        if dedup_nested and any(
            _edge_nested(score.affix, other.affix, score.side)
            or _edge_nested(other.affix, score.affix, score.side)
            for other in filtered
        ):
            continue
        filtered.append(score)

    if top_n is None:
        window = [score.weighted_mi for score in filtered[:elbow_window]]
        top_n = find_elbow(window)

    return [score.affix for score in filtered[:top_n]]


def sort_by_length(affixes: list[str]) -> list[str]:
    """Return affixes ordered longest-first, for longest-match stripping."""
    return sorted(affixes, key=len, reverse=True)


def strip_affix(
    word: str,
    affixes_by_length: list[str],
    side: str = "suffix",
    min_residual: int = 3,
) -> tuple[str, str]:
    """Strip the longest matching affix from one edge, guarding stem length.

    The longest matching affix is stripped, but only if doing so leaves a stem of
    at least ``min_residual`` characters — so ``saijands`` loses ``-ands`` (stem
    ``saij``) rather than ``-nds``, while a short word keeps its affix rather than
    being ground down to nothing.

    Args:
        word: A romanized word form.
        affixes_by_length: Affixes for this side, ordered longest-first.
        side: ``"suffix"`` or ``"prefix"``.
        min_residual: Minimum stem length that must remain after stripping.

    Returns:
        ``(stem, affix)`` — ``stem`` is always the remainder. If no affix can be
        stripped within the guard, returns ``(word, "")``.
    """
    for affix in affixes_by_length:
        if len(word) - len(affix) < min_residual:
            continue
        if side == "suffix" and word.endswith(affix):
            return word[: len(word) - len(affix)], affix
        if side == "prefix" and word.startswith(affix):
            return word[len(affix):], affix
    return word, ""


def strip_both(
    word: str,
    prefixes_by_length: list[str],
    suffixes_by_length: list[str],
    min_residual: int = 3,
) -> tuple[str, str, str]:
    """Strip the longest matching prefix and suffix, guarding the stem.

    Prefix is stripped first, then the suffix from the remainder; the guard is
    applied at each stage, so the residual stem is at least ``min_residual``.

    Args:
        word: A romanized word form.
        prefixes_by_length: Prefixes ordered longest-first.
        suffixes_by_length: Suffixes ordered longest-first.
        min_residual: Minimum stem length to leave at each strip.

    Returns:
        ``(prefix, stem, suffix)`` with empty strings where nothing was stripped.
    """
    after_prefix, prefix = strip_affix(word, prefixes_by_length, "prefix", min_residual)
    stem, suffix = strip_affix(after_prefix, suffixes_by_length, "suffix", min_residual)
    return prefix, stem, suffix


def attested_junction_chars(
    content_words: list[str],
    affixes_by_length: list[str],
    side: str = "suffix",
    min_residual: int = 3,
) -> dict[str, set[str]]:
    """Map each affix to the stem characters attested across its junction.

    Built by running the *same* longest-match stripping over the real vocabulary,
    so the junction character is defined identically to how grafting works. For a
    suffix this is the stem-*final* character preceding it; for a prefix, the
    stem-*initial* character following it. Used to reject implausible junctions
    (e.g. grafting ``-nds`` onto a ``t``-final stem -> ``…tnds``).

    Args:
        content_words: Real romanized word forms (function-word-filtered).
        affixes_by_length: Affixes for this side, ordered longest-first.
        side: ``"suffix"`` or ``"prefix"``.
        min_residual: Minimum stem length used when stripping.

    Returns:
        A dict from affix to the set of attested adjacent stem characters. Affixes
        that never strip within the guard are absent.
    """
    junction: dict[str, set[str]] = {}
    for word in content_words:
        stem, affix = strip_affix(word, affixes_by_length, side, min_residual)
        if affix:
            adjacent = stem[-1] if side == "suffix" else stem[0]
            junction.setdefault(affix, set()).add(adjacent)
    return junction


def build_pseudo_stems(
    content_words: list[str],
    prefixes_by_length: list[str],
    suffixes_by_length: list[str],
    min_residual: int = 3,
) -> list[str]:
    """Reduce content words to pseudo-stems by stripping both edges.

    Args:
        content_words: Function-word-filtered romanized word forms.
        prefixes_by_length: Prefixes ordered longest-first.
        suffixes_by_length: Suffixes ordered longest-first.
        min_residual: Minimum stem length to leave when stripping.

    Returns:
        One pseudo-stem per input word (the word itself if nothing was stripped).
    """
    return [
        strip_both(word, prefixes_by_length, suffixes_by_length, min_residual)[1]
        for word in content_words
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--function-word-top-k", type=int, default=50)
    parser.add_argument("--min-word-length", type=int, default=3)
    args = parser.parse_args()

    vocab = content_vocab(
        load_training_vocab(),
        min_length=args.min_word_length,
        top_k=args.function_word_top_k,
    )
    print(f"content types: {len(vocab)}")
    for side in ("suffix", "prefix"):
        scores = score_affixes(
            list(vocab),
            side=side,
            max_length=args.max_length,
            min_count=args.min_count,
        )
        print(f"\n=== {side}es (top {args.top_n}) ===")
        print(f"{'affix':>8}  {'len':>3}  {'obs':>5}  {'ratio':>8}  {'wMI':>8}")
        for score in scores[: args.top_n]:
            print(
                f"{score.affix:>8}  {score.length:>3}  {score.observed:>5}  "
                f"{score.ratio:>8.1f}  {score.weighted_mi:>8.1f}"
            )


if __name__ == "__main__":
    main()
