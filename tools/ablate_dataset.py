"""Derive an ablated copy of a multinomial dataset config with source(s) removed.

Given a dataset config such as ``configs/dataset/gothic_instruct_1b.yaml`` and one
or more source ids, this writes a new config with those sources dropped and the
remaining explicit ``sampling_prob`` values renormalized so that the *pinned*
probability budget is unchanged:

    new_prob_i = orig_prob_i * pinned_budget / (pinned_budget - ablated_probs)

When every source is pinned, ``pinned_budget`` is 1.0 and this reduces to
``orig_prob_i / (1 - ablated_prob)``. When some sources are unpinned (i.e. their
share is set by alpha-based temperature weighting), holding the pinned budget
fixed means the unpinned sources keep exactly the share they had before, and the
ablated mass is redistributed only among the other pinned sources.

Sources pinned via ``upsampling_factor`` rather than ``sampling_prob`` are left
untouched and reported as a caveat: their induced probability is
``size_i * factor_i / total_size``, and ``total_size`` changes when a source is
removed, so it cannot be recomputed from the config alone (it needs the loaded
corpus sizes). Ablating a fully unpinned source needs no rewriting at all --
alpha reallocates its mass automatically.

Comments and formatting of the input file are preserved where possible: the
output is produced by editing the original lines, falling back to a plain YAML
dump (losing comments) if the edited text does not re-parse to the expected
config.

Usage:
    # Drop one source, write configs/dataset/gothic_instruct_1b_no-got-lexicon.yaml
    python tools/ablate_dataset.py configs/dataset/gothic_instruct_1b.yaml \\
        --ablate got-lexicon

    # Drop several, choose the output path explicitly
    python tools/ablate_dataset.py configs/dataset/gothic_instruct_1b.yaml \\
        --ablate got-lexicon --ablate eng-squad \\
        -o configs/dataset/gothic_instruct_1b_ablate2.yaml

    # Preview only
    python tools/ablate_dataset.py configs/dataset/gothic_instruct_1b.yaml \\
        --ablate got-lexicon --dry-run
"""

import argparse
import sys
from pathlib import Path

import yaml


def get_source_id(source: dict, index: int) -> str:
    """Return a source's id, falling back to a positional label."""
    for key in ("id", "name", "path"):
        value = source.get(key)
        if value is not None:
            return str(value)
    return f"source_{index}"


def classify_pinning(source: dict) -> str:
    """Return how a source's probability is pinned: 'prob', 'factor', or 'alpha'."""
    if source.get("sampling_prob") is not None:
        return "prob"
    if source.get("upsampling_factor") is not None:
        return "factor"
    return "alpha"


def compute_target_budget(
    sources: list[dict],
    ablated_indices: set[int],
    mode: str,
) -> float:
    """
    Compute the probability mass the surviving pinned sources should sum to.

    Under ``pinned``, the mass held by ``sampling_prob``-pinned sources is kept
    constant, so unpinned sources keep exactly their previous share and only the
    pinned mass is redistributed.

    Under ``all``, a lone unpinned source is treated as de-facto pinned -- its
    probability is exactly the residual the pinned sources leave over, so it is a
    degree of freedom in name only -- and everything is scaled up uniformly by
    ``1 / (1 - ablated_share)``. Rewriting the pinned probabilities is enough:
    the residual grows by the same factor on its own. This mode requires at most
    one unpinned source and no ``upsampling_factor`` sources, since otherwise no
    de-facto probability is recoverable from the config alone.

    Args:
        sources: All source dicts from the input config, in file order.
        ablated_indices: Indices of the sources being removed.
        mode: Either 'pinned' or 'all'.

    Returns:
        The target probability mass for the surviving pinned sources.

    Raises:
        ValueError: If mode is 'all' but the de-facto shares cannot be determined.
    """
    survivors_by_pinning = {
        kind: [
            index
            for index, source in enumerate(sources)
            if index not in ablated_indices and classify_pinning(source) == kind
        ]
        for kind in ("factor", "alpha")
    }
    pinned_budget = sum(
        source["sampling_prob"]
        for source in sources
        if classify_pinning(source) == "prob"
    )

    if mode == "pinned":
        # if nothing is left to absorb the residual budget, the pinned
        # probabilities must sum to 1.0 rather than to their previous total
        absorbs_residual = bool(survivors_by_pinning["factor"] or survivors_by_pinning["alpha"])
        return pinned_budget if absorbs_residual else 1.0

    factor_ids = [
        get_source_id(source, index)
        for index, source in enumerate(sources)
        if classify_pinning(source) == "factor"
    ]
    if factor_ids:
        raise ValueError(
            f"--renormalize all cannot be used with upsampling_factor sources "
            f"({', '.join(factor_ids)}): their de-facto probability depends on the "
            "loaded corpus sizes. Use --renormalize pinned."
        )

    alpha_indices = [
        index for index, source in enumerate(sources) if classify_pinning(source) == "alpha"
    ]
    # a single unpinned source is de-facto pinned: its probability is exactly the
    # residual the pinned sources leave over. With two or more, the residual is
    # split between them by alpha weighting over the loaded corpus sizes, so no
    # individual de-facto probability is recoverable from the config alone.
    if len(alpha_indices) > 1:
        alpha_ids = [get_source_id(sources[index], index) for index in alpha_indices]
        raise ValueError(
            f"--renormalize all needs at most one unpinned source, but found "
            f"{len(alpha_indices)} ({', '.join(alpha_ids)}): their individual "
            "de-facto probabilities are set by alpha weighting over the loaded "
            "corpus sizes and cannot be recovered from the config. Use "
            "--renormalize pinned."
        )
    ablated_alpha = [index for index in alpha_indices if index in ablated_indices]

    # the unpinned group's de-facto share is whatever the pinned sources leave over
    ablated_pinned = sum(
        sources[index]["sampling_prob"]
        for index in ablated_indices
        if classify_pinning(sources[index]) == "prob"
    )
    ablated_share = ablated_pinned
    if ablated_alpha:
        ablated_share += 1.0 - pinned_budget
    if ablated_share >= 1.0:
        raise ValueError("The ablated sources account for the entire sampling distribution")

    return (pinned_budget - ablated_pinned) / (1.0 - ablated_share)


def reallocate_probs(
    sources: list[dict],
    ablated_indices: set[int],
    decimals: int,
    mode: str,
) -> dict[int, float]:
    """
    Compute renormalized sampling probabilities for the surviving pinned sources.

    Args:
        sources: All source dicts from the input config, in file order.
        ablated_indices: Indices of the sources being removed.
        decimals: Number of decimal places to round the new probabilities to.
        mode: Renormalization mode, either 'pinned' or 'all'. See
            :func:`compute_target_budget`.

    Returns:
        Mapping from source index (in the original list) to its new probability.
        Empty if no reallocation is needed.

    Raises:
        ValueError: If the surviving pinned sources hold no probability mass, or
            the target budget cannot be determined.
    """
    prob_indices = [
        index
        for index, source in enumerate(sources)
        if classify_pinning(source) == "prob"
    ]
    surviving = [index for index in prob_indices if index not in ablated_indices]
    ablated = [index for index in prob_indices if index in ablated_indices]

    pinned_budget = sum(sources[index]["sampling_prob"] for index in prob_indices)
    target_budget = compute_target_budget(sources, ablated_indices, mode)

    if not surviving:
        if prob_indices:
            raise ValueError(
                "Every sampling_prob-pinned source was ablated; nothing left to "
                "reallocate the freed probability mass to"
            )
        return {}

    if not ablated and target_budget == pinned_budget:
        return {}

    surviving_total = sum(sources[index]["sampling_prob"] for index in surviving)
    scale = target_budget / surviving_total

    exact_probs = {index: sources[index]["sampling_prob"] * scale for index in surviving}

    # naive rounding would break the exact-sum checks in _compute_sampling_probs,
    # so quantize with largest-remainder allocation: every value lands on the
    # rounding grid and the total still matches the target budget exactly
    step = 10 ** -decimals
    units = {index: int(prob / step) for index, prob in exact_probs.items()}
    remaining_units = round(target_budget / step) - sum(units.values())
    by_remainder = sorted(
        surviving,
        key=lambda index: exact_probs[index] / step - units[index],
        reverse=True,
    )
    for position in range(remaining_units):
        units[by_remainder[position % len(by_remainder)]] += 1

    return {index: round(count * step, decimals) for index, count in units.items()}


def build_ablated_config(
    config: dict,
    ablated_indices: set[int],
    new_probs: dict[int, float],
) -> dict:
    """Return a copy of the config with sources dropped and probabilities updated."""
    ablated_config = dict(config)
    new_sources = []
    for index, source in enumerate(config["sources"]):
        if index in ablated_indices:
            continue
        new_source = dict(source)
        if index in new_probs:
            new_source["sampling_prob"] = new_probs[index]
        new_sources.append(new_source)
    ablated_config["sources"] = new_sources
    return ablated_config


def find_source_line_spans(lines: list[str]) -> list[tuple[int, int]]:
    """
    Locate the line span of each item in the top-level ``sources`` list.

    Args:
        lines: Lines of the original YAML file, without trailing newlines.

    Returns:
        List of (start, end) line indices, end exclusive, one per source, in file
        order. Empty if the block-style list could not be identified.
    """
    sources_line = None
    for index, line in enumerate(lines):
        if line.startswith("sources:") and not line[len("sources:"):].strip():
            sources_line = index
            break
    if sources_line is None:
        return []

    starts = []
    item_indent = None
    end = len(lines)
    for index in range(sources_line + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            end = index
            break
        if item_indent is None and line.lstrip().startswith("- "):
            item_indent = indent
        if indent == item_indent and line.lstrip().startswith("- "):
            starts.append(index)

    if not starts:
        return []

    spans = []
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else end
        spans.append((start, stop))
    return spans


def rewrite_preserving_comments(
    text: str,
    ablated_indices: set[int],
    new_probs: dict[int, float],
) -> str | None:
    """
    Produce the ablated YAML by editing the original lines, keeping comments.

    Args:
        text: Original file contents.
        ablated_indices: Indices of sources to drop.
        new_probs: Mapping from source index to its new sampling probability.

    Returns:
        The rewritten text, or None if the file's layout could not be handled.
    """
    lines = text.splitlines()
    spans = find_source_line_spans(lines)
    if not spans:
        return None

    replacements = {}
    for index, (start, stop) in enumerate(spans):
        if index not in new_probs:
            continue
        prob_lines = [
            line_index
            for line_index in range(start, stop)
            if lines[line_index].lstrip("- ").startswith("sampling_prob:")
        ]
        if len(prob_lines) != 1:
            return None
        line_index = prob_lines[0]
        original = lines[line_index]
        indent = original[: len(original) - len(original.lstrip())]
        prefix = "- " if original.lstrip().startswith("- ") else ""
        replacements[line_index] = f"{indent}{prefix}sampling_prob: {new_probs[index]}"

    dropped_lines = set()
    for index in ablated_indices:
        start, stop = spans[index]
        dropped_lines.update(range(start, stop))

    output_lines = []
    for line_index, line in enumerate(lines):
        if line_index in dropped_lines:
            continue
        output_lines.append(replacements.get(line_index, line))
    return "\n".join(output_lines) + "\n"


def render_config(
    text: str,
    expected: dict,
    ablated_indices: set[int],
    new_probs: dict[int, float],
) -> tuple[str, bool]:
    """
    Render the ablated config, preferring the comment-preserving path.

    Returns:
        The rendered YAML text and whether comments were preserved.
    """
    rewritten = rewrite_preserving_comments(text, ablated_indices, new_probs)
    if rewritten is not None:
        try:
            if yaml.safe_load(rewritten) == expected:
                return rewritten, True
        except yaml.YAMLError:
            pass
    return yaml.safe_dump(expected, sort_keys=False, default_flow_style=False), False


def default_output_path(input_path: Path, ablated_ids: list[str]) -> Path:
    """Derive an output path by appending a '_no-<id>...' suffix to the stem."""
    suffix = "_no-" + "-".join(ablated_ids)
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the input multinomial dataset config",
    )
    parser.add_argument(
        "--ablate",
        metavar="SOURCE_ID",
        action="append",
        default=[],
        required=True,
        help="Source id to remove; repeat the flag to remove several",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output config path (default: <input stem>_no-<ids>.yaml alongside the input)",
    )
    parser.add_argument(
        "--renormalize",
        choices=("all", "pinned", "none"),
        default="all",
        help=(
            "How to reallocate the ablated probability mass. 'all' (default) treats "
            "unpinned sources as de-facto pinned and scales every source up "
            "uniformly; 'pinned' holds the pinned budget constant so unpinned "
            "sources keep their previous share; 'none' just drops the sources"
        ),
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for the renormalized probabilities (default: 3, i.e. 0.1%%)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resulting config to stdout instead of writing it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    args = parser.parse_args()

    text = args.config.read_text()
    config = yaml.safe_load(text)

    if not isinstance(config, dict) or "sources" not in config:
        print(f"{args.config}: not a dataset config with a 'sources' list", file=sys.stderr)
        return 1
    if config.get("type") != "multinomial":
        print(
            f"{args.config}: type is {config.get('type')!r}, expected 'multinomial'",
            file=sys.stderr,
        )
        return 1

    sources = config["sources"]
    ids = [get_source_id(source, index) for index, source in enumerate(sources)]

    ablated_indices = set()
    for source_id in args.ablate:
        matches = [index for index, name in enumerate(ids) if name == source_id]
        if not matches:
            print(
                f"Source id {source_id!r} not found; available: {', '.join(ids)}",
                file=sys.stderr,
            )
            return 1
        ablated_indices.update(matches)

    if len(ablated_indices) == len(sources):
        print("Refusing to ablate every source", file=sys.stderr)
        return 1

    if args.renormalize == "none":
        new_probs = {}
    else:
        try:
            new_probs = reallocate_probs(
                sources,
                ablated_indices,
                args.decimals,
                args.renormalize,
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1

    ablated_config = build_ablated_config(config, ablated_indices, new_probs)
    rendered, comments_kept = render_config(text, ablated_config, ablated_indices, new_probs)

    print("Ablated sources:")
    for index in sorted(ablated_indices):
        pinning = classify_pinning(sources[index])
        detail = {
            "prob": f"sampling_prob={sources[index].get('sampling_prob')}",
            "factor": f"upsampling_factor={sources[index].get('upsampling_factor')}",
            "alpha": "unpinned (alpha-weighted)",
        }[pinning]
        print(f"  - {ids[index]}: {detail}")

    if new_probs:
        print("\nReallocated sampling_prob values:")
        for index in sorted(new_probs):
            print(f"  - {ids[index]}: {sources[index]['sampling_prob']} -> {new_probs[index]}")
    else:
        print("\nNo sampling_prob values needed reallocation.")

    factor_ids = [
        ids[index]
        for index, source in enumerate(sources)
        if index not in ablated_indices and classify_pinning(source) == "factor"
    ]
    alpha_ids = [
        ids[index]
        for index, source in enumerate(sources)
        if index not in ablated_indices and classify_pinning(source) == "alpha"
    ]
    if factor_ids:
        print(
            "\nWARNING: left upsampling_factor sources untouched: "
            f"{', '.join(factor_ids)}. Their induced probability is "
            "size * factor / total_size, and total_size shrinks when a source is "
            "removed, so their effective share will drift upward. Recomputing it "
            "requires the loaded corpus sizes (see tools/sampling_dist.py).",
            file=sys.stderr,
        )
    if alpha_ids and args.renormalize == "pinned":
        print(
            f"\nNote: unpinned (alpha-weighted) sources {', '.join(alpha_ids)} keep "
            "their previous share, since the pinned budget was held constant."
        )
    elif alpha_ids and not factor_ids:
        surviving_pinned = sum(
            new_probs.get(index, source.get("sampling_prob") or 0.0)
            for index, source in enumerate(sources)
            if index not in ablated_indices
        )
        old_residual = 1.0 - sum(
            source.get("sampling_prob") or 0.0 for source in sources
        )
        print(
            f"\nNote: unpinned source {', '.join(alpha_ids)} was treated as de-facto "
            f"pinned; its share moves {round(old_residual, args.decimals)} -> "
            f"{round(1.0 - surviving_pinned, args.decimals)} implicitly, since it "
            "absorbs whatever the pinned sources leave over."
        )
    if not comments_kept:
        print(
            "\nWARNING: could not edit the original text safely; the output was "
            "re-dumped from parsed YAML and comments/formatting were lost.",
            file=sys.stderr,
        )

    if args.dry_run:
        print("\n--- ablated config ---")
        print(rendered, end="")
        return 0

    output_path = args.output or default_output_path(args.config, args.ablate)
    if output_path.exists() and not args.force:
        print(f"\n{output_path} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    output_path.write_text(rendered)
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
