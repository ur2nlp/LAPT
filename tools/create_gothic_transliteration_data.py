#!/usr/bin/env python3
"""
Create parallel data for Gothic script transliteration.

Takes Gothic text (in Latin romanization) and creates bidirectional training examples
between Latin and Gothic Unicode scripts. Each input sentence generates two training
examples: Latin→Gothic and Gothic→Latin.

Input: Cleaned Gothic text file (one sentence per line, Latin romanization)
Output: Parallel data file with script variants

Example output (simple concatenation, default):
    saihwan þata 𐍃𐌰𐌹𐍈𐌰𐌽 𐌸𐌰𐍄𐌰
    𐍃𐌰𐌹𐍈𐌰𐌽 𐌸𐌰𐍄𐌰 saihwan þata

Example output (instruction format):
    Transliterate to Gothic script: saihwan þata Transliteration: 𐍃𐌰𐌹𐍈𐌰𐌽 𐌸𐌰𐍄𐌰
    Transliterate to Latin script: 𐍃𐌰𐌹𐍈𐌰𐌽 𐌸𐌰𐍄𐌰 Transliteration: saihwan þata
"""

import argparse
import sys
from pathlib import Path
from transliterate import transliterate_latin_to_gothic


def create_transliteration_data(
    input_file: str,
    output_file: str,
    instruction_format: bool = False
):
    """
    Create bidirectional parallel data from Gothic text.

    For each input sentence in Latin script, generates:
    1. Latin → Gothic script example
    2. Gothic script → Latin example

    Args:
        input_file: Path to input file (one sentence per line, Latin romanization)
        output_file: Path to output parallel data file
        instruction_format: If True, use instruction tuning format; if False, simple concatenation (default)
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentences_read = 0
    examples_written = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            latin_text = line.strip()

            # Skip empty lines
            if not latin_text:
                continue

            sentences_read += 1

            # Generate Gothic script version
            gothic_text = transliterate_latin_to_gothic(latin_text)

            # Create both directions
            if instruction_format:
                # Instruction tuning format
                latin_to_gothic = f"Transliterate to Gothic script: {latin_text} Transliteration: {gothic_text}"
                gothic_to_latin = f"Transliterate to Latin script: {gothic_text} Transliteration: {latin_text}"
            else:
                # Simple concatenation format
                latin_to_gothic = f"{latin_text} {gothic_text}"
                gothic_to_latin = f"{gothic_text} {latin_text}"

            # Collapse whitespace and write both examples
            f_out.write(' '.join(latin_to_gothic.split()) + '\n')
            f_out.write(' '.join(gothic_to_latin.split()) + '\n')
            examples_written += 2

    print(f"\nProcessing Statistics:", file=sys.stderr)
    print(f"  Input sentences: {sentences_read}", file=sys.stderr)
    print(f"  Training examples generated: {examples_written}", file=sys.stderr)
    print(f"  Instruction format: {instruction_format}", file=sys.stderr)
    print(f"\nWrote {examples_written} training examples to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Create bidirectional parallel data for Gothic script transliteration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create simple concatenation format (default, for continued pretraining)
  python create_gothic_transliteration_data.py data/gotica/gotica_clean.txt data/parallel/gothic_transliteration.txt

  # Create instruction format data
  python create_gothic_transliteration_data.py data/gotica/gotica_clean.txt data/parallel/gothic_transliteration.txt --instruction_format

Output format (default):
  {latin text} {gothic script text}
  {gothic script text} {latin text}

Output format (instruction format):
  Transliterate to Gothic script: {latin text} Transliteration: {gothic script text}
  Transliterate to Latin script: {gothic script text} Transliteration: {latin text}
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input Gothic text file (one sentence per line, Latin romanization)'
    )

    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output parallel data file'
    )

    parser.add_argument(
        '--instruction_format',
        action='store_true',
        help='Use instruction tuning format instead of simple concatenation'
    )

    args = parser.parse_args()

    print(f"Creating Gothic transliteration data: {args.input_file} → {args.output_file}", file=sys.stderr)
    create_transliteration_data(
        args.input_file,
        args.output_file,
        instruction_format=args.instruction_format
    )
    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    main()
