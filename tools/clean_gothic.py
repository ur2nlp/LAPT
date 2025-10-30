#!/usr/bin/env python3
"""
Clean Gothic Bible text by removing metadata and extracting pure Gothic text.

Input: data/gotica/gotica.txt
Output: data/gotica/gotica_clean.txt

Format: One verse per line, no metadata. Deduplicates verses that appear in
multiple codices by randomly selecting one variant per verse reference.
"""

import argparse
import random
import re
from collections import defaultdict


def clean_gothic(input_path: str, output_path: str, seed: int = 1):
    """
    Extract Gothic text from verses, removing all metadata and deduplicating.

    Args:
        input_path: Path to raw Gothic Bible file
        output_path: Path for cleaned output
        seed: Random seed for reproducible deduplication
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

    # Second pass: randomly select one variant per verse and write
    total_verses = len(verse_order)
    duplicates_found = sum(1 for variants in verse_groups.values() if len(variants) > 1)

    print(f"Found {total_verses} unique verse references")
    print(f"Found {duplicates_found} verses with multiple codex variants")
    print(f"Using random seed: {seed}")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for verse_ref in verse_order:
            variants = verse_groups[verse_ref]
            # Randomly select one variant
            chosen_text = random.choice(variants)
            f_out.write(chosen_text + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean Gothic Bible text and deduplicate verses with multiple codex variants'
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
        help='Random seed for reproducible deduplication (default: 1)'
    )

    args = parser.parse_args()

    print(f"Cleaning Gothic Bible: {args.input_file} → {args.output_file}")
    clean_gothic(args.input_file, args.output_file, args.seed)
    print("Done!")
