#!/usr/bin/env python3
"""
Plot frequency histograms of Gothic surface forms and Gothic/English surface
pairs in verified word-spotting alignments.

Used to diagnose Claude's tendency to repeatedly select a small set of common
words (e.g. atta/father, dagam/days) when batched into many small requests.
The histograms inform a threshold above which we resample alignments for those
sentences.

Usage:
    python -m gothic.word_spotting.alignment_frequencies \\
        data/gothic_word_spotting/train_verified_b.jsonl \\
        data/gothic_word_spotting/test_verified_b.jsonl \\
        --output-dir outputs/word_spotting_freq
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_text,
    geom_col,
    ggplot,
    labs,
    scale_x_discrete,
    scale_y_log10,
    theme,
    theme_bw,
)


def load_alignments(paths: list[Path]) -> list[dict]:
    alignments = []
    for path in paths:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                for alignment in record.get("alignments", []):
                    alignments.append(alignment)
    return alignments


def normalize_gothic(form: str) -> str:
    return form.strip().lower()


def normalize_english(form: str) -> str:
    return form.strip().lower()


def plot_repetition_distribution(
    counts: Counter,
    title: str,
    output_path: Path,
) -> None:
    # distribution of repetition counts: how many distinct items appear k times
    count_of_counts = Counter(counts.values())
    df = pd.DataFrame(
        {
            "occurrences": list(count_of_counts.keys()),
            "n_items": list(count_of_counts.values()),
        }
    )
    plot = (
        ggplot(df, aes(x="occurrences", y="n_items"))
        + geom_col(fill="steelblue", color="black")
        + scale_y_log10()
        + labs(
            title=f"{title}: repetition-count distribution",
            x="Number of occurrences (k)",
            y="Number of distinct items occurring k times (log)",
        )
        + theme_bw()
    )
    plot.save(str(output_path), width=8, height=5, dpi=120, verbose=False)


def plot_top_n(
    counts: Counter,
    title: str,
    output_path: Path,
    top_n: int = 30,
) -> None:
    top_items = counts.most_common(top_n)
    df = pd.DataFrame(
        {
            "item": [str(item) for item, _ in top_items],
            "count": [n for _, n in top_items],
        }
    )
    # preserve descending order under coord_flip
    ordered = list(reversed(df["item"].tolist()))
    plot = (
        ggplot(df, aes(x="item", y="count"))
        + geom_col(fill="steelblue", color="black")
        + scale_x_discrete(limits=ordered)
        + coord_flip()
        + labs(
            title=f"{title}: top {top_n} most frequent",
            x="",
            y="Occurrences",
        )
        + theme_bw()
        + theme(axis_text_y=element_text(size=7))
    )
    plot.save(str(output_path), width=8, height=8, dpi=120, verbose=False)


def summarize(counts: Counter, name: str, thresholds: list[int]) -> None:
    total_items = sum(counts.values())
    distinct = len(counts)
    print(f"\n== {name} ==")
    print(f"  total alignments contributing: {total_items}")
    print(f"  distinct items:                {distinct}")
    print(f"  mean occurrences/item:         {total_items / distinct:.2f}")
    print(f"  max occurrences (singleton):   {max(counts.values())}")
    for t in thresholds:
        n_items = sum(1 for c in counts.values() if c >= t)
        n_alignments = sum(c for c in counts.values() if c >= t)
        print(
            f"  items with count >= {t:>3}: {n_items:>5} "
            f"({n_alignments} alignments, "
            f"{100 * n_alignments / total_items:.1f}% of total)"
        )
    print(f"\n  Top 20:")
    for item, c in counts.most_common(20):
        print(f"    {c:>5}  {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more verified word-spotting JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write histogram PNGs.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[2, 3, 5, 10, 20, 50],
        help="Count thresholds for summary table.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    alignments = load_alignments(args.inputs)
    print(f"Loaded {len(alignments)} alignments from {len(args.inputs)} file(s).")

    gothic_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    for a in alignments:
        gothic = normalize_gothic(a["gothic_word_roman"])
        english = normalize_english(a["target_word"])
        gothic_counts[gothic] += 1
        pair_counts[(gothic, english)] += 1

    summarize(gothic_counts, "Gothic surface form", args.thresholds)
    summarize(pair_counts, "Gothic / English surface pair", args.thresholds)

    plot_repetition_distribution(
        gothic_counts,
        "Gothic surface form",
        args.output_dir / "gothic_surface_repetition.png",
    )
    plot_top_n(
        gothic_counts,
        "Gothic surface form",
        args.output_dir / "gothic_surface_top.png",
    )
    plot_repetition_distribution(
        pair_counts,
        "Gothic / English surface pair",
        args.output_dir / "gothic_english_pair_repetition.png",
    )
    plot_top_n(
        pair_counts,
        "Gothic / English surface pair",
        args.output_dir / "gothic_english_pair_top.png",
    )
    print(f"\nPlots written to {args.output_dir}/")


if __name__ == "__main__":
    main()
