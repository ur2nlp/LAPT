"""
Scatterplot of generation chrF against minimum translation holdout bpc.

For each run we have two independent measurements of translation quality:

  * chrF of the model's greedy generations on the translation holdout, parsed
    from the ``# chrF:`` header line of ``tools/chrf_eval.py`` reports living in
    ``generation/translation/<run>.txt``; and
  * the best (minimum) ``eval_got-translation_holdout_bpc`` reached anywhere in
    training, read from ``outputs/trainer_states/<run>.json``.

This script joins the two by run name, scatters chrF (y) against min bpc (x),
and reports Pearson and Spearman correlations. The point is to check how well
the cheap in-training proxy (bpc) tracks the more meaningful generation metric
(chrF) across completed runs.

Usage:
    python tools/chrf_vs_bpc.py
    python tools/chrf_vs_bpc.py \
        --chrf-dir generation/translation \
        --states-dir outputs/trainer_states \
        --bpc-key eval_got-translation_holdout_bpc \
        --output plots/chrf_vs_bpc.png
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    labs,
    theme_bw,
)

# chrF value on the report header line, e.g. "# chrF: chrF2++ = 34.27".
CHRF_HEADER_PATTERN = re.compile(r"^#\s*chrF:.*=\s*([0-9.]+)\s*$")


def experiment_number(run_name: str) -> int | None:
    """Parse a run's experiment index from its name (the trailing integer).

    E.g. ``v74L-i40`` -> 40, ``qwen-i1`` -> 1. Returns None if the name has no
    digits (such a run is dropped from the history plot, which is ordered by it).
    """
    matches = re.findall(r'\d+', run_name)
    if not matches:
        return None
    return int(matches[-1])


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson product-moment correlation coefficient."""
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation: Pearson on the rank-transformed values.

    Uses average ranks so ties are handled the same way scipy would.
    """
    x_ranks = pd.Series(x).rank().to_numpy()
    y_ranks = pd.Series(y).rank().to_numpy()
    return pearson_corr(x_ranks, y_ranks)


def parse_chrf(report_path: Path) -> float:
    """Extract the corpus chrF score from a chrf_eval.py report header.

    Args:
        report_path: Path to a ``generation/translation/<run>.txt`` report.

    Returns:
        The corpus chrF score.

    Raises:
        ValueError: If no ``# chrF:`` header line is found.
    """
    with report_path.open(encoding='utf-8') as handle:
        for line in handle:
            match = CHRF_HEADER_PATTERN.match(line)
            if match:
                return float(match.group(1))
            # The header block ends at the rule; stop before per-example lines.
            if line.startswith('#=') or not line.startswith('#'):
                break
    raise ValueError(f"No '# chrF:' header line found in {report_path}")


def min_bpc(state_path: Path, bpc_key: str) -> float | None:
    """Return the minimum value of ``bpc_key`` across a trainer state's log.

    Args:
        state_path: Path to a trainer_state JSON file.
        bpc_key: The log-history key to minimize (an eval bpc metric).

    Returns:
        The minimum logged value, or None if the key never appears (the run
        did not evaluate on this holdout).
    """
    state = json.loads(state_path.read_text(encoding='utf-8'))
    values = [
        entry[bpc_key]
        for entry in state.get('log_history', [])
        if bpc_key in entry
    ]
    if not values:
        return None
    return min(values)


def collect_pairs(
    chrf_dir: Path,
    states_dir: Path,
    bpc_key: str,
) -> tuple[list[str], list[float], list[float]]:
    """Join chrF reports and trainer states by run name.

    Args:
        chrf_dir: Directory of ``<run>.txt`` chrF reports.
        states_dir: Directory of ``<run>.json`` trainer states.
        bpc_key: Trainer-state eval key to minimize.

    Returns:
        Parallel lists of (run_name, min_bpc, chrf), one entry per run that has
        both a parseable chrF report and a matching state with the bpc key.
    """
    names = []
    bpcs = []
    chrfs = []
    for report_path in sorted(chrf_dir.glob('*.txt')):
        run_name = report_path.stem
        state_path = states_dir / f"{run_name}.json"
        if not state_path.exists():
            print(f"skip {run_name}: no trainer state at {state_path}", file=sys.stderr)
            continue
        bpc = min_bpc(state_path, bpc_key)
        if bpc is None:
            print(f"skip {run_name}: key {bpc_key!r} not in trainer state", file=sys.stderr)
            continue
        chrf = parse_chrf(report_path)
        names.append(run_name)
        bpcs.append(bpc)
        chrfs.append(chrf)
    return names, bpcs, chrfs


def make_plot(
    frame: pd.DataFrame,
    bpc_key: str,
    pearson_r: float,
    spearman_r: float,
    output_path: Path,
    annotate: bool,
) -> None:
    """Render the chrF-vs-bpc scatterplot with plotnine and save it.

    Args:
        frame: DataFrame with 'run', 'label', 'min_bpc', and 'chrf' columns.
        bpc_key: The bpc metric name, for the x-axis label.
        pearson_r: Pearson correlation, shown in the subtitle.
        spearman_r: Spearman correlation, shown in the subtitle.
        output_path: Where to save the figure.
        annotate: Whether to label each point with its run suffix.
    """
    plot = (
        ggplot(frame, aes(x='min_bpc', y='chrf'))
        + geom_point(color='steelblue', size=3)
        + labs(
            x=f"min {bpc_key}",
            y="generation chrF",
            title=f"chrF vs. min translation bpc  (n={len(frame)})",
            subtitle=f"Pearson r = {pearson_r:.3f}   Spearman rho = {spearman_r:.3f}",
        )
        + theme_bw()
    )
    if annotate:
        plot = plot + geom_text(
            aes(label='label'),
            nudge_y=0.15, size=7, color='dimgray',
        )

    plot.save(output_path, width=8, height=6, dpi=150, verbose=False)
    print(f"Wrote plot to {output_path}", file=sys.stderr)


def make_history_plot(
    frame: pd.DataFrame,
    output_path: Path,
    window: int,
) -> None:
    """Plot chrF against experiment number to expose era-level trends.

    The point of this view (vs. the bpc scatter) is to see whether groups of
    consecutive experiments sit above or below the overall chrF noise floor —
    a run of points above the mean line is a genuinely better era, not scatter.

    Args:
        frame: DataFrame with 'exp_num' and 'chrf' columns (rows with a null
            exp_num are dropped).
        output_path: Where to save the figure.
        window: Centered rolling-mean window (in experiments) for the trend line.
    """
    ordered = (
        frame.dropna(subset=['exp_num'])
        .sort_values('exp_num')
        .reset_index(drop=True)
    )
    mean_chrf = ordered['chrf'].mean()
    ordered = ordered.assign(
        rolling=ordered['chrf'].rolling(window=window, center=True, min_periods=1).mean()
    )

    # Label every fifth experiment (by number) so the x-axis stays readable.
    labeled = ordered[ordered['exp_num'] % 5 == 0]

    plot = (
        ggplot(ordered, aes(x='exp_num', y='chrf'))
        + geom_hline(yintercept=mean_chrf, linetype='dashed', color='gray')
        + geom_line(color='lightsteelblue')
        + geom_point(color='steelblue', size=2.5)
        + geom_line(aes(y='rolling'), color='firebrick', size=1)
        + geom_text(
            aes(label='label'), data=labeled,
            nudge_y=0.28, size=7, color='dimgray',
        )
        + labs(
            x="experiment number",
            y="generation chrF",
            title=f"chrF over experiment history  (n={len(ordered)})",
            subtitle=(
                f"gray dashed = mean {mean_chrf:.2f};  "
                f"red = rolling mean (window={window})"
            ),
        )
        + theme_bw()
    )
    plot.save(output_path, width=9, height=5, dpi=150, verbose=False)
    print(f"Wrote plot to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Scatter generation chrF against min translation holdout bpc"
    )
    parser.add_argument(
        '--chrf-dir', type=Path, default=Path('generation/translation'),
        help='Directory of <run>.txt chrF reports (default: generation/translation)'
    )
    parser.add_argument(
        '--states-dir', type=Path, default=Path('outputs/trainer_states'),
        help='Directory of <run>.json trainer states (default: outputs/trainer_states)'
    )
    parser.add_argument(
        '--bpc-key', type=str, default='eval_got-translation_holdout_bpc',
        help='Trainer-state log key to minimize (default: eval_got-translation_holdout_bpc)'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('plots/chrf_vs_bpc.png'),
        help='Path to save the chrF-vs-bpc scatter (default: plots/chrf_vs_bpc.png)'
    )
    parser.add_argument(
        '--history-output', type=Path, default=Path('plots/chrf_history.png'),
        help='Path to save the chrF-over-experiment-number plot '
             '(default: plots/chrf_history.png)'
    )
    parser.add_argument(
        '--history-window', type=int, default=5,
        help='Centered rolling-mean window for the history trend line (default: 5)'
    )
    parser.add_argument(
        '--no-annotate', action='store_true',
        help='Do not label each scatter point with its run suffix'
    )
    args = parser.parse_args()

    names, bpcs, chrfs = collect_pairs(args.chrf_dir, args.states_dir, args.bpc_key)
    if len(names) < 2:
        print(f"Need at least 2 paired runs; got {len(names)}.", file=sys.stderr)
        sys.exit(1)

    frame = pd.DataFrame({
        'run': names,
        'label': [name.split('-')[-1] if '-' in name else name for name in names],
        'exp_num': [experiment_number(name) for name in names],
        'min_bpc': bpcs,
        'chrf': chrfs,
    })

    bpc_array = frame['min_bpc'].to_numpy()
    chrf_array = frame['chrf'].to_numpy()
    pearson_r = pearson_corr(bpc_array, chrf_array)
    spearman_r = spearman_corr(bpc_array, chrf_array)

    print(f"Paired runs: {len(frame)}")
    print(f"Pearson  r   = {pearson_r:.4f}")
    print(f"Spearman rho = {spearman_r:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_plot(
        frame=frame,
        bpc_key=args.bpc_key,
        pearson_r=pearson_r,
        spearman_r=spearman_r,
        output_path=args.output,
        annotate=not args.no_annotate,
    )

    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    make_history_plot(
        frame=frame,
        output_path=args.history_output,
        window=args.history_window,
    )


if __name__ == '__main__':
    main()
