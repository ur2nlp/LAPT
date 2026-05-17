#!/usr/bin/env python3
"""
Clean Beowulf text by removing English prose, line numbers, and splitting into sentences.

Input: data/beowulf/beowulf.txt
Output: data/beowulf/beowulf_clean.txt

Format: One sentence per line (split by punctuation: . ! ? ;)
"""

import re
import sys


def clean_beowulf(input_path: str, output_path: str):
    """
    Extract Old English text from Beowulf and split into sentences.

    Args:
        input_path: Path to raw Beowulf file
        output_path: Path for cleaned output
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find where Old English starts (line with "Hwæt!")
    start_idx = None
    for i, line in enumerate(lines):
        if 'Hwæt!' in line or 'Hwæt,' in line:
            start_idx = i
            break

    if start_idx is None:
        print("Error: Could not find start of Old English text (Hwæt!)")
        sys.exit(1)

    print(f"Found Old English starting at line {start_idx + 1}")

    # Process verse lines
    text_accumulator = []

    for line in lines[start_idx:]:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Stop at glossary or end matter
        # Look for specific glossary markers rather than just uppercase
        if 'GLOSSARY' in line or 'ABBREVIATIONS' in line or line.startswith('LIST OF NAMES'):
            break

        # Remove line numbers (digits at start of line)
        line = re.sub(r'^\s*\d+\s+', '', line)

        # Skip if line is now empty
        if not line:
            continue

        # Accumulate text (verses are split across lines)
        text_accumulator.append(line)

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

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for sentence in output_sentences:
            if sentence.strip():
                f_out.write(sentence.strip() + '\n')

    print(f"Extracted {len(output_sentences)} sentences")


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Cleaning Beowulf: {input_file} → {output_file}")
    clean_beowulf(input_file, output_file)
    print("Done!")
