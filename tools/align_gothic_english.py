#!/usr/bin/env python3
"""
Align Gothic Bible with English Bible (World English Bible) to create parallel data.

Reads the Gothic Bible and WEB Bible, matches verses by book/chapter/verse references,
and outputs parallel text for language model training.
"""

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List


# Mapping from Gothic abbreviations to WEB book names
GOTHIC_TO_WEB_BOOKS = {
    # New Testament
    'Mt': 'Matthew',
    'Mk': 'Mark',
    'Lk': 'Luke',
    'Jo': 'John',
    'Rm': 'Romans',
    'Co1': '1 Corinthians',
    'Co2': '2 Corinthians',
    'Ga': 'Galatians',
    'Ef': 'Ephesians',
    'Fp': 'Philippians',
    'Cl': 'Colossians',
    'Th1': '1 Thessalonians',
    'Th2': '2 Thessalonians',
    'Ti1': '1 Timothy',
    'Ti2': '2 Timothy',
    'Tt': 'Titus',
    'Phm': 'Philemon',
    'Hb': 'Hebrews',
    'Jc': 'James',
    'Pe1': '1 Peter',
    'Pe2': '2 Peter',
    'Jo1': '1 John',
    'Jo2': '2 John',
    'Jo3': '3 John',
    'Jd': 'Jude',
    'Ap': 'Revelation',

    # Old Testament (if present in Gothic)
    'Neh': 'Nehemiah',
}


def parse_gothic_bible(gothic_file: str, seed: int = 1) -> Dict[Tuple[str, int, int], str]:
    """
    Parse Gothic Bible and extract verses.

    When a verse appears in multiple codices, randomly selects one variant.

    Args:
        gothic_file: Path to Gothic Bible text file
        seed: Random seed for reproducible variant selection

    Returns:
        Dictionary mapping (book, chapter, verse) to Gothic text
    """
    # First pass: collect all variants for each verse
    verse_variants: Dict[Tuple[str, int, int], List[str]] = {}

    with open(gothic_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Match verse pattern: Mt 5:15 [CA] <text>
            match = re.match(r'^([A-Za-z0-9]+)\s+(\d+):(\d+)\s+\[[A-Z]+\]\s+(.+)$', line)
            if match:
                book_abbrev = match.group(1)
                chapter = int(match.group(2))
                verse = int(match.group(3))
                text = match.group(4).strip()

                # Normalize lacunae markers to "..."
                # Replace multiple dots (with or without spaces) with single "..."
                text = re.sub(r'\.(\s*\.)+', '...', text)

                if text and text != '...':  # Only include if there's actual text beyond lacunae
                    if book_abbrev in GOTHIC_TO_WEB_BOOKS:
                        book = GOTHIC_TO_WEB_BOOKS[book_abbrev]
                        key = (book, chapter, verse)

                        if key not in verse_variants:
                            verse_variants[key] = []
                        verse_variants[key].append(text)

    # Second pass: randomly select one variant per verse
    random.seed(seed)
    verses = {}
    duplicate_count = 0

    for key, variants in verse_variants.items():
        if len(variants) > 1:
            duplicate_count += 1

        # Randomly select one variant
        selected = random.choice(variants)
        verses[key] = selected

    print(f"  Found {duplicate_count} verses with multiple codex variants", file=sys.stderr)

    return verses


def parse_web_bible(web_file: str) -> Dict[Tuple[str, int, int], str]:
    """
    Parse World English Bible and extract verses.

    Args:
        web_file: Path to WEB Bible text file

    Returns:
        Dictionary mapping (book, chapter, verse) to English text
    """
    verses = {}
    current_book = None

    with open(web_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()

            # Check for book header: Book 40 Matthew
            book_match = re.match(r'^Book\s+\d+\s+(.+)$', line)
            if book_match:
                current_book = book_match.group(1)
                continue

            # Check for verse: 001:001 <text>
            if current_book:
                verse_match = re.match(r'^(\d{3}):(\d{3})\s+(.+)$', line)
                if verse_match:
                    chapter = int(verse_match.group(1))
                    verse = int(verse_match.group(2))
                    text = verse_match.group(3).strip()

                    # Remove inline footnotes in curly braces
                    text = re.sub(r'\{[^}]+\}', '', text)
                    text = text.strip()

                    if text:
                        verses[(current_book, chapter, verse)] = text

                # Handle continuation lines (indented, no verse number)
                elif line.startswith('        ') and verses:
                    # Append to last verse
                    last_key = list(verses.keys())[-1]
                    if last_key[0] == current_book:
                        continuation = line.strip()
                        # Remove inline footnotes
                        continuation = re.sub(r'\{[^}]+\}', '', continuation)
                        if continuation:
                            verses[last_key] += ' ' + continuation

    return verses


def create_parallel_data(
    gothic_verses: Dict[Tuple[str, int, int], str],
    english_verses: Dict[Tuple[str, int, int], str],
    output_file: str,
    bidirectional: str = 'on',
    instruction_format: bool = True,
    seed: int = 1
):
    """
    Create parallel data from aligned Gothic and English verses.

    Args:
        gothic_verses: Dictionary of Gothic verses
        english_verses: Dictionary of English verses
        output_file: Path to output file
        bidirectional: 'on' for both directions, 'off' for English→Gothic only, 'random' for random direction per pair
        instruction_format: If True, use instruction tuning format; if False, simple concatenation
        seed: Random seed for reproducible random direction selection
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aligned_count = 0
    gothic_only_count = 0
    examples_written = 0

    # Set random seed for reproducible random direction selection
    if bidirectional == 'random':
        random.seed(seed)

    with open(output_file, 'w', encoding='utf-8') as f:
        for key in sorted(gothic_verses.keys()):
            book, chapter, verse = key
            gothic_text = gothic_verses[key]

            if key in english_verses:
                english_text = english_verses[key]

                # Determine which direction(s) to write
                if bidirectional == 'random':
                    # Randomly choose one direction
                    write_eng_to_got = random.choice([True, False])
                    write_got_to_eng = not write_eng_to_got
                elif bidirectional == 'on':
                    # Write both directions
                    write_eng_to_got = True
                    write_got_to_eng = True
                else:  # bidirectional == 'off'
                    # Only English → Gothic
                    write_eng_to_got = True
                    write_got_to_eng = False

                if instruction_format:
                    # Instruction tuning format
                    if write_eng_to_got:
                        eng_to_got = f"Translate to Gothic: {english_text} Translation: {gothic_text}"
                        eng_to_got = ' '.join(eng_to_got.split())  # Collapse to single line
                        f.write(eng_to_got + '\n')
                        examples_written += 1

                    if write_got_to_eng:
                        got_to_eng = f"Translate to English: {gothic_text} Translation: {english_text}"
                        got_to_eng = ' '.join(got_to_eng.split())  # Collapse to single line
                        f.write(got_to_eng + '\n')
                        examples_written += 1
                else:
                    # Simple concatenation format for continued pretraining
                    if write_eng_to_got:
                        eng_to_got = f"{english_text} {gothic_text}"
                        eng_to_got = ' '.join(eng_to_got.split())  # Collapse to single line
                        f.write(eng_to_got + '\n')
                        examples_written += 1

                    if write_got_to_eng:
                        got_to_eng = f"{gothic_text} {english_text}"
                        got_to_eng = ' '.join(got_to_eng.split())  # Collapse to single line
                        f.write(got_to_eng + '\n')
                        examples_written += 1

                aligned_count += 1
            else:
                gothic_only_count += 1
                print(f"Warning: No English verse for {book} {chapter}:{verse}", file=sys.stderr)

    print(f"\nAlignment Statistics:", file=sys.stderr)
    print(f"  Aligned verse pairs: {aligned_count}", file=sys.stderr)
    print(f"  Gothic-only verses (no match): {gothic_only_count}", file=sys.stderr)
    print(f"  Total Gothic verses: {len(gothic_verses)}", file=sys.stderr)
    print(f"  Bidirectional: {bidirectional}", file=sys.stderr)
    print(f"  Instruction format: {instruction_format}", file=sys.stderr)
    print(f"\nWrote {examples_written} training examples to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Align Gothic and English Bible verses to create parallel data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create bidirectional parallel data with instruction format (default)
  python align_gothic_english.py

  # Create only English→Gothic examples
  python align_gothic_english.py --bidirectional off

  # Create parallel data with random direction per pair (good for continued pretraining)
  python align_gothic_english.py --bidirectional random --no-instruction-format

  # Create parallel data without instruction format (for continued pretraining)
  python align_gothic_english.py --no-instruction-format

Output format (bidirectional=on, instruction format):
  Translate to Gothic: {english text} Translation: {gothic text}
  Translate to English: {gothic text} Translation: {english text}

Output format (bidirectional=on, no instruction format):
  {english text} {gothic text}
  {gothic text} {english text}

Output format (bidirectional=random, no instruction format):
  {english text} {gothic text}  # or {gothic text} {english text} - randomly chosen per pair
        """
    )

    parser.add_argument(
        '--gothic',
        type=str,
        default='data/gotica/gotica.txt',
        help='Path to Gothic Bible text file (default: data/gotica/gotica.txt)'
    )

    parser.add_argument(
        '--english',
        type=str,
        default='data/web_bible/web_bible.txt',
        help='Path to English Bible text file (default: data/web_bible/web_bible.txt)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/parallel/gothic_english.txt',
        help='Path to output parallel data file (default: data/parallel/gothic_english.txt)'
    )

    parser.add_argument(
        '--bidirectional',
        type=str,
        choices=['on', 'off', 'random'],
        default='on',
        help='Direction mode: "on" for both directions, "off" for English→Gothic only, "random" for random direction per pair (default: on)'
    )

    parser.add_argument(
        '--no-instruction-format',
        action='store_true',
        help='Use simple concatenation instead of instruction format (for continued pretraining)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed for selecting codex variants (default: 1)'
    )

    args = parser.parse_args()

    print("Parsing Gothic Bible...", file=sys.stderr)
    gothic_verses = parse_gothic_bible(args.gothic, seed=args.seed)
    print(f"  Found {len(gothic_verses)} Gothic verses", file=sys.stderr)

    print("\nParsing English Bible...", file=sys.stderr)
    english_verses = parse_web_bible(args.english)
    print(f"  Found {len(english_verses)} English verses", file=sys.stderr)

    print("\nCreating parallel data...", file=sys.stderr)
    create_parallel_data(
        gothic_verses,
        english_verses,
        args.output,
        bidirectional=args.bidirectional,
        instruction_format=not args.no_instruction_format,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
