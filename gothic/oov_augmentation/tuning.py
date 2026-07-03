"""Auto-tune non-word sampling parameters by the real-word collision rate.

The right operating point for temperature / top-p is objective and corpus-
grounded (see `.claude/gothic/oov_robustness_augmentation.md`): the **acceptance
rate** — the fraction of generated candidates that are *not* real Gothic words —
is a proxy for distance from the data manifold.

  * acceptance ~ 1.0: never collides with a real word => generating gibberish far
    from Gothic phonotactics (the "keys-on-spelling-weirdness" failure).
  * acceptance ~ 0.0: almost always a real word => memorizing the lexicon.
  * acceptance ~ 0.25-0.5 (resample 2-4x): non-words sit right at the manifold
    edge — maximally confusable with real words, which is the "well-formed but
    unknown" signal we want.

Tuning reuses a single fitted stem model and only varies ``temperature`` /
``top_p`` (the trigram counts are independent of both), so the grid search is
cheap. Collision is measured by generating with the known-word filter *off* and
checking membership against the real-word vocabulary.
"""

import argparse
import random
from dataclasses import dataclass

from gothic.oov_augmentation.stems import StemModel, build_stem_model, generate_nonword
from gothic.oov_augmentation.vocab import content_vocab, load_training_vocab


@dataclass
class GridPoint:
    """Acceptance measured at one (temperature, top_p) setting.

    Attributes:
        temperature: Sampling temperature.
        top_p: Nucleus cutoff.
        acceptance: Fraction of produced candidates that are not real words.
        failure_rate: Fraction of draws that produced no candidate (e.g. hit the
            length cap without ending) — a diagnostic that the cap is too tight
            for this temperature.
    """

    temperature: float
    top_p: float
    acceptance: float
    failure_rate: float


@dataclass
class TuneResult:
    """The chosen setting plus the full measured grid."""

    best: GridPoint
    grid: list[GridPoint]


def acceptance_rate(
    stem_model: StemModel,
    source_words: list[str],
    n_samples: int,
    rng: random.Random,
) -> GridPoint:
    """Measure the acceptance (non-word) rate at the model's current settings.

    Args:
        stem_model: A built stem model; its ``temperature`` / ``top_p`` are read
            as-is (set them before calling).
        source_words: Real words to draw from as replacement targets, so affix
            grafting matches deployment.
        n_samples: Number of draws.
        rng: Seeded RNG for source-word selection.

    Returns:
        A GridPoint recording acceptance and failure rate at the current settings.
    """
    known = stem_model.model.known_words
    produced = 0
    accepted = 0
    for _ in range(n_samples):
        source = rng.choice(source_words)
        candidate = generate_nonword(stem_model, source, reject_known=False)
        if candidate is None:
            continue
        produced += 1
        if candidate not in known:
            accepted += 1

    acceptance = accepted / produced if produced else 0.0
    failure_rate = 1.0 - produced / n_samples if n_samples else 0.0
    return GridPoint(
        temperature=stem_model.model.temperature,
        top_p=stem_model.model.top_p,
        acceptance=acceptance,
        failure_rate=failure_rate,
    )


def tune_sampling(
    stem_model: StemModel,
    source_words: list[str],
    temperatures: list[float],
    top_ps: list[float],
    target_acceptance: float = 0.33,
    n_samples: int = 500,
    seed: int = 1,
) -> TuneResult:
    """Grid-search (temperature, top_p) for acceptance closest to the target.

    Args:
        stem_model: A built stem model (mutated in place during the search).
        source_words: Real words to draw replacement targets from.
        temperatures: Temperatures to try.
        top_ps: Nucleus cutoffs to try.
        target_acceptance: Desired acceptance (0.33 ~ resample 3x). Ties broken
            toward lower temperature, i.e. more phonotactically typical output.
        n_samples: Draws per grid point.
        seed: RNG seed (reused per grid point so points are comparable).

    Returns:
        A TuneResult with the best GridPoint and the full grid.
    """
    grid: list[GridPoint] = []
    for temperature in temperatures:
        for top_p in top_ps:
            stem_model.model.temperature = temperature
            stem_model.model.top_p = top_p
            rng = random.Random(seed)
            grid.append(acceptance_rate(stem_model, source_words, n_samples, rng))

    best = min(
        grid,
        key=lambda point: (abs(point.acceptance - target_acceptance), point.temperature),
    )
    return TuneResult(best=best, grid=grid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6, 0.7, 0.8],
    )
    parser.add_argument("--top-ps", type=float, nargs="+", default=[0.8, 0.9, 1.0])
    parser.add_argument("--target-acceptance", type=float, default=0.33)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    stem_model = build_stem_model(seed=args.seed)
    source_words = list(content_vocab(load_training_vocab()))

    result = tune_sampling(
        stem_model,
        source_words,
        temperatures=args.temperatures,
        top_ps=args.top_ps,
        target_acceptance=args.target_acceptance,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    print(f"{'temp':>6}  {'top_p':>6}  {'accept':>7}  {'fail':>6}")
    for point in result.grid:
        marker = "  <-- best" if point is result.best else ""
        print(
            f"{point.temperature:>6.2f}  {point.top_p:>6.2f}  "
            f"{point.acceptance:>7.3f}  {point.failure_rate:>6.3f}{marker}"
        )
    print(
        f"\nchosen: temperature={result.best.temperature}, "
        f"top_p={result.best.top_p} "
        f"(acceptance={result.best.acceptance:.3f}, "
        f"~{1 / result.best.acceptance:.1f}x resample)"
    )


if __name__ == "__main__":
    main()
