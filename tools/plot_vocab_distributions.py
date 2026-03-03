"""
Visualize seed vocabulary interpolation as distribution bar plots.

Shows how base (pre-trained) and seed (corpus-derived) vocabulary distributions
are merged via lambda-weighted interpolation in the FOCUS tokenizer pipeline.

Usage:
    # All 4 plots for a single lambda
    python tools/plot_vocab_distributions.py \\
        path/to/base_vocab_counts.txt \\
        path/to/seed_tokenizer/spm.model \\
        --lambda 0.99 --cutoff auto --log-scale

    # Compare plot 4 across multiple lambdas
    python tools/plot_vocab_distributions.py \\
        path/to/base_vocab_counts.txt \\
        path/to/seed_tokenizer/spm.model \\
        --lambda 0.01 0.1 0.3 0.5 0.7 0.9 0.99 \\
        --plots 4 --cutoff auto --log-scale

    # Just plots 3 and 4 for two lambdas
    python tools/plot_vocab_distributions.py \\
        path/to/base_vocab_counts.txt \\
        path/to/seed_tokenizer/spm.model \\
        --lambda 0.3 0.99 --plots 3 4 --log-scale
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tokenizer_utils import (
    apply_character_weighting,
    extract_target_seed_vocab,
    normalize_vocab_mass,
)

COLOR_BASE = "#2E86AB"
COLOR_TARGET = "#E8702A"
COLOR_BOUNDARY = "#888888"
COLOR_CUTOFF = "#333333"


def load_base_vocab(path: str) -> dict[str, int]:
    """Load tab-separated token<TAB>count file."""
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                token, count_str = parts
                vocab[token] = int(count_str)
    return vocab


def load_seed_vocab_and_size(
    spm_model_path: str,
    target_mass: int,
) -> tuple[dict[str, float], int]:
    """Load seed vocab and return (vocab_dict, spm_vocab_size).

    Args:
        spm_model_path: Path to .model file from SentencePiece training.
        target_mass: Target total count for normalization.

    Returns:
        Tuple of (token->count dict, total piece count in the SPM model).
    """
    import sentencepiece as spm

    vocab = extract_target_seed_vocab(
        spm_model_path=spm_model_path,
        target_mass=target_mass,
        filter_special_tokens=True,
    )

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spm_model_path)
    seed_vocab_size = sp_model.get_piece_size()

    return vocab, seed_vocab_size


def build_token_ordering(
    base_vocab: dict[str, int | float],
    target_vocab: dict[str, float],
) -> tuple[list[str], int]:
    """Build the unified token ordering shared by plots 1-3.

    Seed tokens come first (sorted by target count descending), then base-only
    tokens (sorted by base count descending).

    Args:
        base_vocab: Token->count from base tokenizer corpus analysis.
        target_vocab: Token->count from seed tokenizer (normalized floats).

    Returns:
        Tuple of (ordered token list, number of seed tokens).
    """
    seed_tokens = set(target_vocab.keys())
    base_only = set(base_vocab.keys()) - seed_tokens

    seed_sorted = sorted(seed_tokens, key=lambda t: target_vocab[t], reverse=True)
    base_only_sorted = sorted(base_only, key=lambda t: base_vocab[t], reverse=True)

    return seed_sorted + base_only_sorted, len(seed_sorted)


def build_arrays_for_lambda(
    ordered_tokens: list[str],
    base_vocab: dict[str, int | float],
    target_vocab: dict[str, float],
    lambda_weight: float,
) -> dict:
    """Build numpy arrays for a single lambda value.

    Args:
        ordered_tokens: Unified token ordering from build_token_ordering().
        base_vocab: Token->count from base tokenizer corpus analysis.
        target_vocab: Token->count from seed tokenizer (normalized floats).
        lambda_weight: Interpolation weight (0=target, 1=base).

    Returns:
        Dict with arrays for plotting.
    """
    total_length = len(ordered_tokens)

    target_counts = np.zeros(total_length)
    base_counts = np.zeros(total_length)

    for i, token in enumerate(ordered_tokens):
        target_counts[i] = target_vocab.get(token, 0.0)
        base_counts[i] = base_vocab.get(token, 0.0)

    weighted_base = base_counts * lambda_weight
    weighted_target = target_counts * (1.0 - lambda_weight)
    combined = weighted_base + weighted_target

    rerank_order = np.argsort(-combined)

    return {
        "target_counts": target_counts,
        "base_counts": base_counts,
        "weighted_base": weighted_base,
        "weighted_target": weighted_target,
        "combined": combined,
        "rerank_order": rerank_order,
    }


def bin_array(arr: np.ndarray, num_bins: int) -> np.ndarray:
    """Downsample an array into equal-width bins using mean aggregation.

    Args:
        arr: 1D array of values (one per token rank).
        num_bins: Number of output bins.

    Returns:
        1D array of length num_bins with mean values per bin.
    """
    n = len(arr)
    if n <= num_bins:
        return arr
    chunks = np.array_split(arr, num_bins)
    return np.array([chunk.mean() for chunk in chunks])


def bin_position(raw_pos: int, total_tokens: int, num_bins: int) -> float:
    """Map a token-rank position to the corresponding bin-space position.

    Args:
        raw_pos: Position in the original token-rank array.
        total_tokens: Total number of tokens before binning.
        num_bins: Number of bins after downsampling.

    Returns:
        Fractional bin index.
    """
    return raw_pos * (num_bins / total_tokens)


def proportional_log_stacks(
    base_arr: np.ndarray,
    target_arr: np.ndarray,
    base_proportion: np.ndarray | None = None,
    target_proportion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log-height bars split by linear proportion.

    Total bar height is log1p(base_arr + target_arr). Color split comes from
    the proportion arrays if given, otherwise from the height arrays themselves.

    Args:
        base_arr: Base contribution to bar height (typically lambda-weighted).
        target_arr: Target contribution to bar height (typically lambda-weighted).
        base_proportion: Base values for color split (e.g., raw pre-lambda counts).
        target_proportion: Target values for color split (e.g., raw pre-lambda counts).
    """
    combined = base_arr + target_arr
    log_height = np.log1p(combined)

    prop_base = base_proportion if base_proportion is not None else base_arr
    prop_target = target_proportion if target_proportion is not None else target_arr
    prop_total = prop_base + prop_target
    with np.errstate(invalid="ignore"):
        base_frac = np.where(prop_total > 0, prop_base / prop_total, 0.0)

    log_base = log_height * base_frac
    log_target = log_height * (1.0 - base_frac)
    return log_base, log_target


# --- Individual plot renderers ---


def draw_plot1(
    ax: plt.Axes,
    target_counts: np.ndarray,
    boundary_pos: float,
    bar_kwargs: dict,
    ylabel: str = "Count",
) -> None:
    """Plot 1: Seed tokenizer distribution."""
    x = np.arange(len(target_counts))
    ax.bar(x, target_counts, color=COLOR_TARGET, **bar_kwargs)
    ax.axvline(x=boundary_pos - 0.5, color=COLOR_BOUNDARY, linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title("Seed Tokenizer Distribution")


def draw_plot2(
    ax: plt.Axes,
    base_counts: np.ndarray,
    boundary_pos: float,
    bar_kwargs: dict,
    ylabel: str = "Count",
) -> None:
    """Plot 2: Base tokenizer distribution (aligned to seed rank)."""
    x = np.arange(len(base_counts))
    ax.bar(x, base_counts, color=COLOR_BASE, **bar_kwargs)
    ax.axvline(x=boundary_pos - 0.5, color=COLOR_BOUNDARY, linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title("Base Tokenizer Distribution (aligned to seed rank)")


def draw_plot3(
    ax: plt.Axes,
    w_base: np.ndarray,
    w_target: np.ndarray,
    lambda_weight: float,
    boundary_pos: float,
    bar_kwargs: dict,
    score_mode: str = "count",
    raw_base: np.ndarray | None = None,
    raw_target: np.ndarray | None = None,
) -> None:
    """Plot 3: Interpolated, stacked, original rank order (log-proportional)."""
    x = np.arange(len(w_base))
    log_base, log_target = proportional_log_stacks(w_base, w_target, raw_base, raw_target)
    unit = "char score" if score_mode == "charlength" else "count"

    prop_label = "raw" if raw_base is not None else "weighted"
    ax.bar(x, log_base, color=COLOR_BASE, label=f"Base × {lambda_weight}", **bar_kwargs)
    ax.bar(
        x, log_target, bottom=log_base,
        color=COLOR_TARGET, label=f"Target × {1 - lambda_weight:.4g}", **bar_kwargs,
    )
    ax.axvline(x=boundary_pos - 0.5, color=COLOR_BOUNDARY, linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylabel(f"log(1 + {unit})")
    ax.set_title(f"Interpolated (λ={lambda_weight}, original rank order)")
    ax.legend(loc="upper right", framealpha=0.8, fontsize=9)
    ax.annotate(
        f"Log-scaled heights; color shows {prop_label} per-rank {unit} proportion, not relative mass",
        xy=(0.02, 0.95), xycoords="axes fraction",
        ha="left", va="top", fontsize=7, fontstyle="italic", color="#666666",
    )


def draw_plot4(
    ax: plt.Axes,
    r_base: np.ndarray,
    r_target: np.ndarray,
    lambda_weight: float,
    cutoff_bin: float | None,
    cutoff_rank: int | None,
    bar_kwargs: dict,
    score_mode: str = "count",
    raw_base: np.ndarray | None = None,
    raw_target: np.ndarray | None = None,
) -> None:
    """Plot 4: Interpolated, re-ranked by combined mass (log-proportional)."""
    x = np.arange(len(r_base))
    log_base, log_target = proportional_log_stacks(r_base, r_target, raw_base, raw_target)
    unit = "char score" if score_mode == "charlength" else "count"

    prop_label = "raw" if raw_base is not None else "weighted"
    ax.bar(x, log_base, color=COLOR_BASE, label=f"Base × {lambda_weight}", **bar_kwargs)
    ax.bar(
        x, log_target, bottom=log_base,
        color=COLOR_TARGET, label=f"Target × {1 - lambda_weight:.4g}", **bar_kwargs,
    )

    if cutoff_bin is not None:
        ax.axvline(
            x=cutoff_bin - 0.5, color=COLOR_CUTOFF, linestyle="--", linewidth=1.5,
            label=f"Cutoff (rank {cutoff_rank:,})",
        )

    ax.set_xlabel(f"Token Rank (re-sorted by combined {unit})")
    ax.set_ylabel(f"log(1 + {unit})")
    ax.set_title(f"Interpolated (λ={lambda_weight}, re-ranked by combined {unit})")
    ax.legend(loc="upper right", framealpha=0.8, fontsize=9)
    ax.annotate(
        f"Log-scaled heights; color shows {prop_label} per-rank {unit} proportion, not relative mass",
        xy=(0.02, 0.95), xycoords="axes fraction",
        ha="left", va="top", fontsize=7, fontstyle="italic", color="#666666",
    )


# --- Figure assembly ---


def create_figure(
    ordered_tokens: list[str],
    n_seed_tokens: int,
    base_vocab: dict[str, int | float],
    target_vocab: dict[str, float],
    lambda_weights: list[float],
    plots: list[int],
    num_bins: int = 256,
    log_scale: bool = False,
    cutoff_position: int | None = None,
    title: str | None = None,
    width: float = 16,
    height_per_row: float = 3.0,
    score_mode: str = "count",
    raw_proportions: bool = False,
) -> plt.Figure:
    """Create figure with selected plots for one or more lambda values.

    When multiple lambdas are given, each lambda gets its own row of the
    selected plots. Plots 1 and 2 are lambda-independent and appear only once
    at the top.

    Args:
        ordered_tokens: Unified token ordering from build_token_ordering().
        n_seed_tokens: Number of seed tokens (boundary position).
        base_vocab: Token->count from base tokenizer.
        target_vocab: Token->count from seed tokenizer.
        lambda_weights: One or more lambda values to plot.
        plots: Which plot numbers to include (subset of [1, 2, 3, 4]).
        num_bins: Bins for downsampling (0 = no binning).
        log_scale: Log y-axis for plots 1-2.
        cutoff_position: Raw rank for cutoff line in plot 4.
        title: Custom overall title.
        width: Figure width in inches.
        height_per_row: Height per subplot row in inches.
        score_mode: Scoring method used ("count" or "charlength"), for title.
        raw_proportions: If True, color split in plots 3-4 uses pre-lambda
            counts while bar height still reflects lambda-weighted counts.

    Returns:
        Matplotlib Figure.
    """
    total_tokens = len(ordered_tokens)
    do_bin = num_bins > 0 and total_tokens > num_bins

    if do_bin:
        n_display = num_bins
        boundary_pos = bin_position(n_seed_tokens, total_tokens, num_bins)
    else:
        n_display = total_tokens
        boundary_pos = n_seed_tokens

    bar_kwargs = dict(width=1.0, linewidth=0, rasterized=True)

    # Determine subplot layout:
    # Plots 1 & 2 are lambda-independent, shown once at the top
    # Plots 3 & 4 are lambda-dependent, one row per lambda
    static_plots = [p for p in plots if p in (1, 2)]
    lambda_plots = [p for p in plots if p in (3, 4)]

    n_static = len(static_plots)
    n_lambda_rows = len(lambda_weights) * len(lambda_plots) if lambda_plots else 0
    n_rows = n_static + n_lambda_rows

    if n_rows == 0:
        print("ERROR: No plots selected.", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(n_rows, 1, figsize=(width, height_per_row * n_rows), squeeze=False)
    axes = axes.flatten()

    row_idx = 0

    # Pre-bin static arrays (only needed if plots 1 or 2 are selected)
    if static_plots:
        # Use lambda_weights[0] just to get the raw arrays; plots 1-2 don't use lambda
        static_data = build_arrays_for_lambda(
            ordered_tokens, base_vocab, target_vocab, lambda_weights[0],
        )
        if do_bin:
            binned_target = bin_array(static_data["target_counts"], num_bins)
            binned_base = bin_array(static_data["base_counts"], num_bins)
        else:
            binned_target = static_data["target_counts"]
            binned_base = static_data["base_counts"]

        ylabel_12 = "Char Score" if score_mode == "charlength" else "Count"
        static_axes = []
        for p in static_plots:
            ax = axes[row_idx]
            if p == 1:
                draw_plot1(ax, binned_target, boundary_pos, bar_kwargs, ylabel=ylabel_12)
            elif p == 2:
                draw_plot2(ax, binned_base, boundary_pos, bar_kwargs, ylabel=ylabel_12)
            if log_scale:
                ax.set_yscale("log")
            ax.set_xlim(-0.5, n_display - 0.5)
            static_axes.append(ax)
            row_idx += 1

        # Share y-axis limits between plots 1 and 2 so the spread
        # difference between the two distributions is directly visible
        if len(static_axes) == 2:
            ylims = [ax.get_ylim() for ax in static_axes]
            shared_ylim = (min(y[0] for y in ylims), max(y[1] for y in ylims))
            for ax in static_axes:
                ax.set_ylim(shared_ylim)

    # Lambda-dependent plots
    for lw in lambda_weights:
        data = build_arrays_for_lambda(ordered_tokens, base_vocab, target_vocab, lw)

        if do_bin:
            w_base = bin_array(data["weighted_base"], num_bins)
            w_target = bin_array(data["weighted_target"], num_bins)
        else:
            w_base = data["weighted_base"]
            w_target = data["weighted_target"]

        # Pre-lambda counts for color proportions (when --raw-proportions)
        prop_base_3 = None
        prop_target_3 = None
        prop_base_4 = None
        prop_target_4 = None
        if raw_proportions:
            if do_bin:
                prop_base_3 = bin_array(data["base_counts"], num_bins)
                prop_target_3 = bin_array(data["target_counts"], num_bins)
            else:
                prop_base_3 = data["base_counts"]
                prop_target_3 = data["target_counts"]

        rerank = data["rerank_order"]
        r_base_raw = data["weighted_base"][rerank]
        r_target_raw = data["weighted_target"][rerank]
        if do_bin:
            r_base = bin_array(r_base_raw, num_bins)
            r_target = bin_array(r_target_raw, num_bins)
            n_display_4 = num_bins
        else:
            r_base = r_base_raw
            r_target = r_target_raw
            n_display_4 = total_tokens

        if raw_proportions:
            r_prop_base_raw = data["base_counts"][rerank]
            r_prop_target_raw = data["target_counts"][rerank]
            if do_bin:
                prop_base_4 = bin_array(r_prop_base_raw, num_bins)
                prop_target_4 = bin_array(r_prop_target_raw, num_bins)
            else:
                prop_base_4 = r_prop_base_raw
                prop_target_4 = r_prop_target_raw

        # Resolve cutoff into bin-space
        if cutoff_position is not None:
            cut_bin = bin_position(cutoff_position, total_tokens, num_bins) if do_bin else cutoff_position
        else:
            cut_bin = None

        lambda_axes = []
        for p in lambda_plots:
            ax = axes[row_idx]
            if p == 3:
                draw_plot3(
                    ax, w_base, w_target, lw, boundary_pos, bar_kwargs, score_mode,
                    raw_base=prop_base_3, raw_target=prop_target_3,
                )
                ax.set_xlim(-0.5, n_display - 0.5)
            elif p == 4:
                draw_plot4(
                    ax, r_base, r_target, lw, cut_bin, cutoff_position, bar_kwargs, score_mode,
                    raw_base=prop_base_4, raw_target=prop_target_4,
                )
                ax.set_xlim(-0.5, n_display_4 - 0.5)
            lambda_axes.append(ax)
            row_idx += 1

        # Share y-axis limits between plots 3 and 4 for this lambda
        if len(lambda_axes) >= 2:
            ylims = [ax.get_ylim() for ax in lambda_axes]
            shared_ylim = (min(y[0] for y in ylims), max(y[1] for y in ylims))
            for ax in lambda_axes:
                ax.set_ylim(shared_ylim)

    # Shared formatting
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    # Suppress x-tick labels on all but the last row
    for ax in axes[:-1]:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    # Build title
    if title:
        fig_title = title
    else:
        mode_label = f", {score_mode}" if score_mode != "count" else ""
        if len(lambda_weights) == 1:
            fig_title = f"Seed Vocabulary Interpolation (λ={lambda_weights[0]}{mode_label})"
        else:
            lw_str = ", ".join(str(lw) for lw in lambda_weights)
            fig_title = f"Seed Vocabulary Interpolation (λ = {lw_str}{mode_label})"

    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def parse_cutoff(
    cutoff_arg: str | None,
    seed_vocab_size: int,
) -> int | None:
    """Resolve --cutoff argument to an integer position or None.

    Args:
        cutoff_arg: CLI argument value: None, "off", "auto", or an integer string.
        seed_vocab_size: Vocab size from the seed SPM model (used for "auto").

    Returns:
        Integer cutoff position, or None if disabled.
    """
    if cutoff_arg is None or cutoff_arg == "off":
        return None
    if cutoff_arg == "auto":
        return seed_vocab_size
    try:
        return int(cutoff_arg)
    except ValueError:
        print(
            f"ERROR: --cutoff must be 'off', 'auto', or an integer, got: {cutoff_arg}",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize seed vocabulary interpolation as distribution bar plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "base_vocab_counts",
        help="Path to base_vocab_counts.txt (tab-separated token<TAB>count)",
    )
    parser.add_argument(
        "seed_model",
        help="Path to seed tokenizer .model file (SentencePiece)",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_weights",
        type=float,
        nargs="+",
        required=True,
        help="Interpolation weight(s) (0=pure target, 1=pure base). Multiple values accepted.",
    )
    parser.add_argument(
        "--plots",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="Which plots to include (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--seed-mass-multiplier",
        type=float,
        default=1.0,
        help="Scale factor for target mass (default: 1.0)",
    )
    parser.add_argument(
        "--round-mode",
        choices=["round", "ceil", "floor"],
        default="round",
        help="Rounding method for merge (default: round)",
    )
    parser.add_argument(
        "--cutoff",
        default=None,
        help=(
            "Vertical cutoff line in plot 4. "
            "'off' = no line (default), 'auto' = seed tokenizer vocab size, "
            "or an integer position."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default="vocab_distributions.png",
        help="Output file path (default: vocab_distributions.png)",
    )
    parser.add_argument("--width", type=float, default=16, help="Figure width in inches")
    parser.add_argument(
        "--row-height",
        type=float,
        default=3.0,
        help="Height per subplot row in inches (default: 3.0)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for y-axis on plots 1-2",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=256,
        help="Number of bins to downsample into (0 = no binning, default: 256)",
    )
    parser.add_argument("--title", help="Custom overall title")
    parser.add_argument(
        "--score-mode",
        choices=["count", "charlength"],
        default="count",
        help="Scoring method: count (default) or charlength (weight by token length)",
    )
    parser.add_argument(
        "--raw-proportions",
        action="store_true",
        help=(
            "In plots 3-4, use pre-lambda counts for color proportions "
            "while bar height remains lambda-weighted"
        ),
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.base_vocab_counts).exists():
        print(f"ERROR: Base vocab file not found: {args.base_vocab_counts}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.seed_model).exists():
        print(f"ERROR: Seed model not found: {args.seed_model}", file=sys.stderr)
        sys.exit(1)

    # Load base vocab
    print(f"Loading base vocab from {args.base_vocab_counts}", file=sys.stderr)
    base_vocab = load_base_vocab(args.base_vocab_counts)
    total_base_tokens = sum(base_vocab.values())
    print(f"  {len(base_vocab):,} tokens, {total_base_tokens:,} total count", file=sys.stderr)

    # Compute target mass
    target_mass = int(total_base_tokens * args.seed_mass_multiplier)

    # Load seed vocab
    print(f"Loading seed tokenizer from {args.seed_model}", file=sys.stderr)
    target_vocab, seed_vocab_size = load_seed_vocab_and_size(args.seed_model, target_mass)
    print(f"  {len(target_vocab):,} tokens, SPM vocab size: {seed_vocab_size:,}", file=sys.stderr)

    # Scale base counts if mass multiplier != 1.0
    if args.seed_mass_multiplier != 1.0:
        print(f"  Scaling base counts by {args.seed_mass_multiplier}", file=sys.stderr)
        base_vocab = {
            token: count * args.seed_mass_multiplier
            for token, count in base_vocab.items()
        }

    # Apply character-length weighting if requested
    if args.score_mode == "charlength":
        print("Applying character-length weighting", file=sys.stderr)
        base_vocab = apply_character_weighting(base_vocab)
        target_vocab = apply_character_weighting(target_vocab)
        base_char_mass = int(sum(base_vocab.values()))
        target_vocab = normalize_vocab_mass(target_vocab, base_char_mass)
        print(f"  Base character mass: {base_char_mass:,}", file=sys.stderr)

    # Compute overlap statistics
    base_tokens = set(base_vocab.keys())
    seed_tokens = set(target_vocab.keys())
    shared = base_tokens & seed_tokens
    base_only = base_tokens - seed_tokens
    seed_only = seed_tokens - base_tokens
    print(
        f"  Shared: {len(shared):,}, Base-only: {len(base_only):,}, "
        f"Seed-only: {len(seed_only):,}",
        file=sys.stderr,
    )

    # Build token ordering (shared across all lambdas)
    print("Building token ordering...", file=sys.stderr)
    ordered_tokens, n_seed_tokens = build_token_ordering(base_vocab, target_vocab)
    print(
        f"  {len(ordered_tokens):,} total slots "
        f"({n_seed_tokens:,} seed + {len(ordered_tokens) - n_seed_tokens:,} base-only)",
        file=sys.stderr,
    )

    # Resolve cutoff
    cutoff_position = parse_cutoff(args.cutoff, seed_vocab_size)
    if cutoff_position is not None:
        print(f"  Cutoff line at rank {cutoff_position:,}", file=sys.stderr)

    # Create figure
    lambda_str = ", ".join(str(lw) for lw in args.lambda_weights)
    plot_str = ", ".join(str(p) for p in args.plots)
    print(f"Creating figure (λ={lambda_str}, plots={plot_str})...", file=sys.stderr)

    fig = create_figure(
        ordered_tokens=ordered_tokens,
        n_seed_tokens=n_seed_tokens,
        base_vocab=base_vocab,
        target_vocab=target_vocab,
        lambda_weights=args.lambda_weights,
        plots=args.plots,
        num_bins=args.bins,
        log_scale=args.log_scale,
        cutoff_position=cutoff_position,
        title=args.title,
        width=args.width,
        height_per_row=args.row_height,
        score_mode=args.score_mode,
        raw_proportions=args.raw_proportions,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
