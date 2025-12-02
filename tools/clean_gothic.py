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
    keep_all_variants: bool = False,
    output_script: str = 'roman'
):
    """
    Extract Gothic text from verses, removing all metadata.

    Args:
        input_path: Path to raw Gothic Bible file
        output_path: Path for cleaned output
        seed: Random seed for reproducible deduplication (only used when keep_all_variants=False)
        keep_all_variants: If True, keep all codex variants; if False, randomly select one per verse
        output_script: Output script - 'roman' (default), 'script' (Gothic Unicode), or 'both'
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

                # Remove meta-brackets (keeping their contents)
                gothic_text = gothic_text.replace('<', '').replace('>', '')
                gothic_text = gothic_text.replace('[', '').replace(']', '')

                if gothic_text:
                    # Track first occurrence for ordering
                    if verse_ref not in verse_groups:
                        verse_order.append(verse_ref)

                    verse_groups[verse_ref].append(gothic_text)

    # Second pass: collect verses (all variants or deduplicated)
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

    # Collect lines to write
    lines = []
    for verse_ref in verse_order:
        variants = verse_groups[verse_ref]

        if keep_all_variants:
            # Collect all variants
            for variant_text in variants:
                lines.append(variant_text)
        else:
            # Randomly select one variant
            chosen_text = random.choice(variants)
            lines.append(chosen_text)

    # Write output based on script choice
    if output_script in ['roman', 'both']:
        # Write romanized version (default)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in lines:
                f_out.write(line + '\n')
        print(f"Wrote {len(lines)} lines to {output_path}")

    if output_script in ['script', 'both']:
        # Write Gothic Unicode script version
        from transliterate import transliterate_latin_to_gothic

        script_path = output_path.replace('.txt', '_script.txt')
        with open(script_path, 'w', encoding='utf-8') as f_out:
            for line in lines:
                gothic_line = transliterate_latin_to_gothic(line)
                f_out.write(gothic_line + '\n')
        print(f"Wrote {len(lines)} lines (Gothic script) to {script_path}")


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
    parser.add_argument(
        '--output_script',
        choices=['roman', 'script', 'both'],
        default='roman',
        help='Output script: "roman" (default, with þ), "script" (Gothic Unicode), or "both" (write two files)'
    )

    args = parser.parse_args()

    print(f"Cleaning Gothic Bible: {args.input_file} → {args.output_file}")
    clean_gothic(
        args.input_file,
        args.output_file,
        args.seed,
        args.keep_all_variants,
        args.output_script
    )
    print("Done!")
