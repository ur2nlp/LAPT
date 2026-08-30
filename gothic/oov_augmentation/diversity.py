"""Continuous plausibility + diversity metrics for tuning non-word sampling.

Exact-collision metrics (real-word acceptance, whole-word self-collision) saturate
here: the output space dwarfs the ~8k-word lexicon, so collisions essentially
never happen and the signal is flat (see
`.claude/gothic/oov_robustness_augmentation.md`). These two metrics replace the
discrete events with continuous, sublexical ones that keep a gradient:

  * **Plausibility (anti-gibberish)** — per-character log-probability of a
    generated word under an *independent* reference char n-gram model trained on
    whole *real* words (not the stem generator, which would be circular). Gibberish
    scores systematically higher NLL. Reported as the fraction of generated words
    whose per-char NLL falls within the real-word range (<= a high quantile of a
    held-out real-word NLL distribution).
  * **Diversity (anti-collapse)** — distinct character trigram ratio across the
    sample. Mode collapse repeats *substrings* (`andands`/`andandjan`) even when
    whole words differ, so this keeps signal where distinct-*word* ratio saturates.

The two bound temperature from opposite sides: plausibility caps it from above
(high T -> gibberish), diversity floors it from below (low T -> collapse).
"""

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from gothic.oov_augmentation.stems import build_stem_model, generate_nonword
from gothic.oov_augmentation.vocab import content_vocab, load_training_vocab

START = "^"
END = "$"


class CharNGramModel:
    """A char n-gram model with stupid-backoff scoring, over whole real words.

    Attributes:
        order: The n-gram order (context length is ``order - 1``).
        backoff: Multiplicative penalty applied each time the context is shortened.
        add_k: Add-k smoothing constant for the unigram fallback.
    """

    def __init__(self, order: int = 4, backoff: float = 0.4, add_k: float = 0.1) -> None:
        self.order = order
        self.backoff = backoff
        self.add_k = add_k
        self._counts: dict[str, Counter] = defaultdict(Counter)
        self._alphabet: set[str] = set()

    def fit(self, words: list[str]) -> "CharNGramModel":
        """Accumulate context counts at every order from 0 to ``order - 1``."""
        for word in words:
            padded = START * (self.order - 1) + word + END
            for position in range(self.order - 1, len(padded)):
                character = padded[position]
                self._alphabet.add(character)
                for context_length in range(self.order):
                    context = padded[position - context_length : position]
                    self._counts[context][character] += 1
        return self

    def char_logprob(self, context: str, character: str) -> float:
        """Log-probability of ``character`` given ``context`` via stupid backoff."""
        context = context[-(self.order - 1):] if self.order > 1 else ""
        weight = 1.0
        while True:
            counts = self._counts.get(context)
            if counts and counts[character] > 0:
                return math.log(weight * counts[character] / sum(counts.values()))
            if context == "":
                vocabulary = len(self._alphabet) + 1
                base = self._counts.get("", Counter())
                total = sum(base.values()) + self.add_k * vocabulary
                return math.log(weight * (base.get(character, 0) + self.add_k) / total)
            context = context[1:]
            weight *= self.backoff

    def per_char_nll(self, word: str) -> float:
        """Mean per-character negative log-likelihood of ``word`` (incl. END)."""
        padded = START * (self.order - 1) + word + END
        total = 0.0
        count = 0
        for position in range(self.order - 1, len(padded)):
            context = padded[position - (self.order - 1) : position]
            total += self.char_logprob(context, padded[position])
            count += 1
        return -total / count if count else 0.0


@dataclass
class DiversityPoint:
    """Metrics measured at one (temperature, top_p) setting."""

    temperature: float
    top_p: float
    acceptance: float
    plausible_fraction: float
    distinct_trigram: float
    failure_rate: float


def build_reference_model(
    order: int = 4,
    holdout_fraction: float = 0.1,
    seed: int = 1,
) -> tuple[CharNGramModel, list[str]]:
    """Train the whole-word reference model, holding out real words for baseline.

    Args:
        order: N-gram order.
        holdout_fraction: Fraction of real word types reserved to characterize the
            real-word NLL distribution (not trained on).
        seed: RNG seed for the split.

    Returns:
        ``(model, heldout_words)``.
    """
    words = list(load_training_vocab())
    rng = random.Random(seed)
    rng.shuffle(words)
    cut = int(len(words) * holdout_fraction)
    heldout = words[:cut]
    train = words[cut:]
    return CharNGramModel(order=order).fit(train), heldout


def quantile(values: list[float], q: float) -> float:
    """Return the ``q``-quantile of ``values`` (nearest-rank)."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[index]


def distinct_ngram_ratio(words: list[str], n: int = 3) -> float:
    """Ratio of unique to total character n-grams across ``words``."""
    seen: set[str] = set()
    total = 0
    for word in words:
        grams = [word[i : i + n] for i in range(len(word) - n + 1)] or [word]
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else 0.0


def evaluate_setting(
    stem_model,
    reference: CharNGramModel,
    plausibility_threshold: float,
    source_words: list[str],
    n_samples: int,
    rng: random.Random,
) -> DiversityPoint:
    """Measure acceptance, plausibility and diversity at the model's settings.

    Args:
        stem_model: A built stem model; its ``temperature`` / ``top_p`` are read
            as-is (set them before calling).
        reference: The whole-word reference char model.
        plausibility_threshold: Per-char NLL cutoff (a high quantile of held-out
            real-word NLL); a generated word is "plausible" at or below it.
        source_words: Real words to draw replacement targets from.
        n_samples: Number of draws.
        rng: Seeded RNG.

    Returns:
        A DiversityPoint at the current settings.
    """
    known = stem_model.model.known_words
    generated: list[str] = []
    accepted = 0
    for _ in range(n_samples):
        candidate = generate_nonword(stem_model, rng.choice(source_words), reject_known=False)
        if candidate is None:
            continue
        generated.append(candidate)
        if candidate not in known:
            accepted += 1

    produced = len(generated)
    plausible = sum(
        1 for word in generated if reference.per_char_nll(word) <= plausibility_threshold
    )
    return DiversityPoint(
        temperature=stem_model.model.temperature,
        top_p=stem_model.model.top_p,
        acceptance=accepted / produced if produced else 0.0,
        plausible_fraction=plausible / produced if produced else 0.0,
        distinct_trigram=distinct_ngram_ratio(generated, n=3),
        failure_rate=1.0 - produced / n_samples if n_samples else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument(
        "--plausibility-quantile",
        type=float,
        default=0.9,
        help="Real-word NLL quantile used as the plausibility cutoff.",
    )
    parser.add_argument(
        "--min-plausible",
        type=float,
        default=0.8,
        help="Minimum plausible fraction a setting must reach to be selectable.",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    reference, heldout = build_reference_model(order=args.order, seed=args.seed)
    real_nll = [reference.per_char_nll(word) for word in heldout]
    threshold = quantile(real_nll, args.plausibility_quantile)
    real_median = quantile(real_nll, 0.5)
    print(
        f"reference order {args.order}: real NLL median={real_median:.3f}, "
        f"plausibility cutoff (q{args.plausibility_quantile})={threshold:.3f}"
    )

    stem_model = build_stem_model(seed=args.seed)
    source_words = list(content_vocab(load_training_vocab()))

    grid: list[DiversityPoint] = []
    for temperature in args.temperatures:
        stem_model.model.temperature = temperature
        stem_model.model.top_p = args.top_p
        rng = random.Random(args.seed)
        grid.append(
            evaluate_setting(
                stem_model, reference, threshold, source_words, args.n_samples, rng
            )
        )

    selectable = [point for point in grid if point.plausible_fraction >= args.min_plausible]
    pool = selectable or grid
    best = max(pool, key=lambda point: point.distinct_trigram)

    print(f"\n{'temp':>6}  {'accept':>7}  {'plausible':>9}  {'distinct3':>9}  {'fail':>6}")
    for point in grid:
        marker = "  <-- best" if point is best else ""
        print(
            f"{point.temperature:>6.2f}  {point.acceptance:>7.3f}  "
            f"{point.plausible_fraction:>9.3f}  {point.distinct_trigram:>9.3f}  "
            f"{point.failure_rate:>6.3f}{marker}"
        )
    print(
        f"\nchosen: temperature={best.temperature} "
        f"(max distinct-3 with plausible >= {args.min_plausible})"
    )


if __name__ == "__main__":
    main()
