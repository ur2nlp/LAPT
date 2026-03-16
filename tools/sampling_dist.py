"""
Show sampling probabilities and exhaustion rates for a multinomial dataset.

Mirrors the probability logic in src/dataset_utils.py:_compute_sampling_probs.
Sources may have their probability pinned via --sp (sampling_prob) or --uf
(upsampling_factor); remaining sources are weighted by alpha.

Usage:
    # Pure alpha-based
    python tools/samples_to_exhaust.py --alpha 0.5 got:5000 non:100000 eng:5000000

    # With pinned sources
    python tools/samples_to_exhaust.py --alpha 0.5 \\
        got:5000 got-eng:3000 non:100000 eng:5000000 \\
        --sp got:0.06 --sp got-eng:0.02

    # With absolute draw counts at a given total
    python tools/samples_to_exhaust.py --alpha 0.5 --total 9000000 \\
        got:5000 non:100000 eng:5000000 --sp got:0.06
"""

import argparse
import sys


def parse_name_value(s: str, value_type: type, flag_name: str) -> tuple:
    """Parse a 'name:value' string into (name, value)."""
    parts = s.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"{flag_name} argument must be in 'name:value' format, got: {s!r}"
        )
    name, raw_value = parts
    try:
        return name, value_type(raw_value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{flag_name}: could not parse {raw_value!r} as {value_type.__name__}"
        )


def compute_sampling_probs(
    names: list[str],
    sizes: list[int],
    alpha: float | None,
    pinned_probs: dict[str, float],
) -> list[float]:
    """
    Compute per-source sampling probabilities, mirroring _compute_sampling_probs.

    Args:
        names: Source names.
        sizes: Source sizes (number of examples).
        alpha: Temperature for unpinned sources. May be None if all sources are pinned.
        pinned_probs: Mapping from source name to pinned sampling probability.

    Returns:
        List of sampling probabilities, one per source, summing to 1.0.

    Raises:
        ValueError: If pinned probabilities are invalid or alpha is missing when needed.
    """
    num_sources = len(names)
    pinned_total = sum(pinned_probs.values())

    if len(pinned_probs) == num_sources:
        if abs(pinned_total - 1.0) > 1e-9:
            raise ValueError(
                f"All sources are pinned but probabilities sum to {pinned_total:.6f}, not 1.0"
            )
        return [pinned_probs[name] for name in names]

    if pinned_total >= 1.0:
        raise ValueError(
            f"Sum of pinned probabilities is {pinned_total:.4f}, "
            "must be less than 1.0 to leave budget for remaining sources"
        )

    if alpha is None:
        raise ValueError("--alpha is required when any sources are unpinned")

    remaining_budget = 1.0 - pinned_total
    unpinned_names = [name for name in names if name not in pinned_probs]
    unpinned_sizes = [sizes[names.index(name)] for name in unpinned_names]

    weights = [size ** alpha for size in unpinned_sizes]
    total_weight = sum(weights)
    unpinned_probs = {
        name: (weight / total_weight) * remaining_budget
        for name, weight in zip(unpinned_names, weights)
    }

    return [pinned_probs.get(name, unpinned_probs.get(name)) for name in names]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sources",
        nargs="+",
        metavar="name:size",
        help="Dataset sources as name:size pairs (e.g. got:5000 eng:5000000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Temperature for alpha-based sources. Required unless all sources are pinned.",
    )
    parser.add_argument(
        "--sp",
        dest="sampling_probs",
        metavar="name:prob",
        action="append",
        default=[],
        help=(
            "Pin a source's sampling probability directly, e.g. --sp got:0.06. "
            "Repeatable. Corresponds to the 'sampling_prob' field in the dataset config."
        ),
    )
    parser.add_argument(
        "--uf",
        dest="upsampling_factors",
        metavar="name:factor",
        action="append",
        default=[],
        help=(
            "Pin a source via upsampling factor relative to its natural rate, "
            "e.g. --uf got:3.0. Repeatable. Corresponds to the 'upsampling_factor' field."
        ),
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        metavar="N",
        help="Total samples (optional). Prints absolute draw counts per source.",
    )
    args = parser.parse_args()

    # Parse sources
    names = []
    sizes = []
    for s in args.sources:
        try:
            name, size = parse_name_value(s, int, "source")
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        names.append(name)
        sizes.append(size)

    if len(set(names)) != len(names):
        parser.error("Duplicate source names are not allowed")

    total_size = sum(sizes)

    # Build pinned_probs, converting upsampling_factor to probability first
    pinned_probs: dict[str, float] = {}

    for raw in args.upsampling_factors:
        try:
            name, factor = parse_name_value(raw, float, "--uf")
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if name not in names:
            parser.error(f"--uf: source {name!r} not found in sources")
        natural_prob = sizes[names.index(name)] / total_size
        prob = natural_prob * factor
        if prob <= 0 or prob >= 1.0:
            parser.error(
                f"--uf: factor {factor} gives probability {prob:.4f} for {name!r}, "
                "must be in (0, 1)"
            )
        pinned_probs[name] = prob

    # sampling_prob takes precedence over upsampling_factor (mirrors source code behavior)
    for raw in args.sampling_probs:
        try:
            name, prob = parse_name_value(raw, float, "--sp")
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if name not in names:
            parser.error(f"--sp: source {name!r} not found in sources")
        if prob <= 0 or prob >= 1.0:
            parser.error(f"--sp: probability must be in (0, 1), got {prob} for {name!r}")
        pinned_probs[name] = prob

    try:
        probs = compute_sampling_probs(names, sizes, args.alpha, pinned_probs)
    except ValueError as exc:
        parser.error(str(exc))

    # Print table
    name_w = max(len("Source"), max(len(n) for n in names))
    size_w = max(len("Size"), max(len(f"{s:,}") for s in sizes))

    col_sep = "  "
    columns = [
        f"{'Source':<{name_w}}",
        f"{'Size':>{size_w}}",
        f"{'Nat%':>8}",
        f"{'Final%':>8}",
        f"{'UF':>7}",
        f"{'To Exhaust':>12}",
    ]
    if args.total is not None:
        columns.append(f"{'Draws':>10}")

    header = col_sep.join(columns)
    print(header)
    print("-" * len(header))

    for name, size, prob in zip(names, sizes, probs):
        natural_prob = size / total_size
        upsampling_factor = prob / natural_prob if natural_prob > 0 else float("inf")
        samples_to_exhaust = int(size / prob) if prob > 0 else float("inf")
        pin_marker = "*" if name in pinned_probs else " "

        row_cols = [
            f"{name + pin_marker:<{name_w}}",
            f"{size:{size_w},}",
            f"{natural_prob:>7.2%}",
            f"{prob:>7.2%}",
            f"{upsampling_factor:>6.2f}x",
            f"{samples_to_exhaust:>12,}",
        ]
        if args.total is not None:
            draws = int(prob * args.total)
            row_cols.append(f"{draws:>10,}")

        print(col_sep.join(row_cols))

    print()
    if args.alpha is not None:
        print(f"alpha={args.alpha}", end="")
        if pinned_probs:
            print(f"  |  pinned (*): {', '.join(pinned_probs.keys())}", end="")
        print()
    elif pinned_probs:
        print(f"pinned (*): {', '.join(pinned_probs.keys())}")
    if args.total is not None:
        print(f"total={args.total:,}")


if __name__ == "__main__":
    main()
