"""
Clean Icelandic Sagas text files.

Removes chapter headings, HTML artifacts, and splits into sentences.
Outputs one sentence per line for language model training.
"""

import argparse
import glob
import os
import re


def clean_sagas(input_dir: str, output_file: str, pattern: str = "*.on.txt"):
    """
    Clean saga files and concatenate into single output file.

    Args:
        input_dir: Directory containing saga text files
        output_file: Path for cleaned output
        pattern: Glob pattern for matching saga files
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    saga_files = sorted(glob.glob(os.path.join(input_dir, pattern)))

    if not saga_files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {input_dir}")

    print(f"Found {len(saga_files)} saga files")

    total_sentences = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for saga_path in saga_files:
            saga_name = os.path.basename(saga_path)
            print(f"Processing {saga_name}...")

            sentences = process_saga(saga_path)
            for sentence in sentences:
                f_out.write(sentence + '\n')

            total_sentences += len(sentences)
            print(f"  {len(sentences)} sentences extracted")

    print(f"\nTotal: {total_sentences} sentences written to {output_file}")


def process_saga(file_path: str) -> list:
    """
    Process a single saga file.

    Args:
        file_path: Path to saga file

    Returns:
        List of cleaned sentences
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Remove HTML artifacts
        line = re.sub(r'&lt;[^&]*&gt;', '', line)
        line = re.sub(r'&amp;', '&', line)
        line = line.strip()

        if not line:
            continue

        # Skip chapter headings (e.g., "1. kafli - Description" or "1. kafli")
        if re.match(r'^\d+\.\s+kafli', line):
            continue

        # Split into sentences by major punctuation
        # Split on: . ! ? ; (optionally followed by quotes) but keep the punctuation
        parts = re.split(r"([.!?;][\"']?)", line)

        # Reconstruct sentences with their punctuation
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i].strip()
            punct = parts[i + 1] if i + 1 < len(parts) else ''
            if sentence:
                sentences.append(sentence + punct)

        # Handle last sentence if no punctuation
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

    return sentences


def main():
    parser = argparse.ArgumentParser(
        description='Clean Icelandic Sagas text files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/sagas',
        help='Directory containing saga files (default: data/sagas)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/sagas/sagas_clean.txt',
        help='Output file for cleaned text (default: data/sagas/sagas_clean.txt)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.on.txt',
        help='Glob pattern for saga files (default: *.on.txt)'
    )

    args = parser.parse_args()

    clean_sagas(args.input_dir, args.output_file, args.pattern)


if __name__ == '__main__':
    main()
