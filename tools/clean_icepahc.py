#!/usr/bin/env python3
"""
Clean IcePaHC texts by removing metadata annotations and splitting into sentences.

Input: data/icecorpus-texts/<century>/*/*.txt
Output: data/icecorpus-texts/icepahc_<centuries>_clean.txt

Format: One sentence per line, split by punctuation (. ! ? ;).
Processes only specified centuries (default: 12th, 13th)
"""

import argparse
import re
from pathlib import Path


def clean_line(line: str) -> str:
    """
    Clean a line by removing metadata and whitespace.

    Args:
        line: Line to clean

    Returns:
        Cleaned line, or empty string if line should be skipped
    """
    # Strip leading/trailing whitespace
    line = line.strip()

    # Skip blank lines
    if not line:
        return ''

    # Skip chapter/section markers like <K 1>
    if re.match(r'^<[A-Z]+ \d+>$', line):
        return ''

    # Skip lines that are only metadata like (COM:...)
    if re.match(r'^\([A-Z]+:[^)]+\)$', line):
        return ''

    # Skip lines that are just section numbers like 8.
    if re.match(r'^\d+\.$', line):
        return ''

    # Strip metadata prefixes like (COM:translation) but keep the rest
    line = re.sub(r'^\([A-Z]+:[^)]+\)\s*', '', line)

    return line


def process_icepahc(base_dir: str, centuries: list, output_path: str):
    """
    Process IcePaHC texts from specified centuries.

    Args:
        base_dir: Base directory containing century folders
        centuries: List of century names (e.g., ['12th', '13th'])
        output_path: Output file path
    """
    text_accumulator = []
    files_processed = 0

    for century in centuries:
        century_path = Path(base_dir) / century

        if not century_path.exists():
            print(f"Warning: Century directory not found: {century_path}")
            continue

        print(f"\nProcessing {century} century...")

        # Iterate through author/work directories
        for work_dir in sorted(century_path.iterdir()):
            if not work_dir.is_dir():
                continue

            # Look for the main .txt file (not .raw.txt, not numbered)
            txt_files = list(work_dir.glob('*.txt'))

            # Filter to main files: not .raw.txt and not numbered (e.g., file01.txt)
            main_files = [
                f for f in txt_files
                if not f.name.endswith('.raw.txt') and not re.search(r'\d+\.txt$', f.name)
            ]

            if not main_files and len(txt_files) == 1:
                # If only one .txt file, use it
                main_files = txt_files

            for txt_file in main_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # Clean and accumulate text
                    for line in lines:
                        cleaned = clean_line(line)
                        if cleaned:
                            text_accumulator.append(cleaned)

                    files_processed += 1
                    print(f"  {txt_file.name}")

                except Exception as e:
                    print(f"  Error processing {txt_file}: {e}")
                    continue

    # Join all text
    full_text = ' '.join(text_accumulator)

    # Clean up extra whitespace
    full_text = re.sub(r'\s+', ' ', full_text)

    # Split into sentences by major punctuation
    # Split on: . ! ? ; (optionally followed by quotes) but keep the punctuation
    sentences = re.split(r"([.!?;][\"']?)", full_text)

    # Reconstruct sentences with their punctuation
    output_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        if sentence:
            output_sentences.append(sentence + punct)

    # Handle last sentence if no punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        output_sentences.append(sentences[-1].strip())

    # Filter out sentences that are just section numbers like "8."
    output_sentences = [
        s for s in output_sentences
        if not re.match(r'^\d+\.$', s.strip())
    ]

    # Write output
    print(f"\nWriting output: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for sentence in output_sentences:
            if sentence.strip():
                f_out.write(sentence.strip() + '\n')

    print(f"\nProcessed {files_processed} files")
    print(f"Extracted {len(output_sentences)} sentences")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean IcePaHC texts by removing metadata annotations'
    )
    parser.add_argument(
        '--base-dir',
        default='data/icecorpus-texts',
        help='Base directory containing century folders (default: data/icecorpus-texts)'
    )
    parser.add_argument(
        '--centuries',
        nargs='+',
        default=['12th', '13th'],
        help='Centuries to process (default: 12th 13th)'
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: data/icecorpus-texts/icepahc_<centuries>_clean.txt)'
    )

    args = parser.parse_args()

    # Compute default output path if not provided
    if args.output is None:
        output_file = f"{args.base_dir}/icepahc_{'_'.join(args.centuries)}_clean.txt"
    else:
        output_file = args.output

    print(f"Cleaning IcePaHC: centuries {args.centuries}")
    print(f"Output: {output_file}\n")

    process_icepahc(args.base_dir, args.centuries, output_file)
    print("Done!")
