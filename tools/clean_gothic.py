#!/usr/bin/env python3
"""
Clean Gothic Bible text by removing metadata and extracting pure Gothic text.

Input: data/gotica/gotica.txt
Output: data/gotica/gotica_clean.txt

Format: One verse per line, no metadata.

By default, deduplicates verses that appear in multiple codices by randomly
selecting one variant per verse reference. Use --keep_all_variants to preserve
all codex variants (one line per variant).
"""

import argparse
import random
import re
from collections import defaultdict


def clean_gothic(
    input_path: str,
    output_path: str,
    seed: int = 1,
    keep_all_variants: bool = False
):
    """
    Extract Gothic text from verses, removing all metadata.

    Args:
        input_path: Path to raw Gothic Bible file
        output_path: Path for cleaned output
        seed: Random seed for reproducible deduplication (only used when keep_all_variants=False)
        keep_all_variants: If True, keep all codex variants; if False, randomly select one per verse
    """
    random.seed(seed)

    # First pass: collect all verses grouped by reference
    verse_groups = defaultdict(list)
    verse_order = []

    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.rstrip()

            # Skip comment lines
            if line.startswith('#'):
                continue

            # Skip empty lines
            if not line.strip():
                continue

            # Skip calendar entries (Cal prefix)
            if line.startswith('Cal '):
                continue

            # Parse verse reference and text
            # Format: "Mt 5:15 [CA] <Gothic text here>"
            # Extract reference (without codex marker) and text
            match = re.match(r'([A-Za-z0-9]+\s+[0-9]+:[0-9]+)\s+\[[^\]]+\]\s*(.+)$', line)
            if match:
                verse_ref = match.group(1)
                gothic_text = match.group(2).strip()

                # Normalize lacunae marks to consistent ellipsis (...)
                # Matches sequences of 2+ dots with optional spaces between them
                gothic_text = re.sub(r'\.(\s*\.)+', '...', gothic_text)

                if gothic_text:
                    # Track first occurrence for ordering
                    if verse_ref not in verse_groups:
                        verse_order.append(verse_ref)

                    verse_groups[verse_ref].append(gothic_text)

    # Second pass: write verses (all variants or deduplicated)
    total_verses = len(verse_order)
    duplicates_found = sum(1 for variants in verse_groups.values() if len(variants) > 1)
    total_variants = sum(len(variants) for variants in verse_groups.values())

    print(f"Found {total_verses} unique verse references")
    print(f"Found {duplicates_found} verses with multiple codex variants")
    print(f"Total variants across all codices: {total_variants}")

    if keep_all_variants:
        print("Mode: Keeping all codex variants")
    else:
        print(f"Mode: Deduplicating (using random seed: {seed})")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        lines_written = 0
        for verse_ref in verse_order:
            variants = verse_groups[verse_ref]

            if keep_all_variants:
                # Write all variants
                for variant_text in variants:
                    f_out.write(variant_text + '\n')
                    lines_written += 1
            else:
                # Randomly select one variant
                chosen_text = random.choice(variants)
                f_out.write(chosen_text + '\n')
                lines_written += 1

        print(f"Wrote {lines_written} lines to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean Gothic Bible text with optional deduplication of codex variants'
    )
    parser.add_argument(
        '--input_file',
        default='data/gotica/gotica.txt',
        help='Path to raw Gothic Bible file (e.g., data/gotica/gotica.txt)'
    )
    parser.add_argument(
        '--output_file',
        default='data/gotica/gotica_clean.txt',
        help='Path for cleaned output (e.g., data/gotica/gotica_clean.txt)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed for reproducible deduplication (default: 1, only used when not keeping all variants)'
    )
    parser.add_argument(
        '--keep_all_variants',
        action='store_true',
        help='Keep all codex variants instead of randomly selecting one per verse (default: False)'
    )

    args = parser.parse_args()

    print(f"Cleaning Gothic Bible: {args.input_file} → {args.output_file}")
    clean_gothic(
        args.input_file,
        args.output_file,
        args.seed,
        args.keep_all_variants
    )
    print("Done!")
