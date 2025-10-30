#!/usr/bin/env python3
"""
Clean Gothic Bible text by removing metadata and extracting pure Gothic text.

Input: data/gotica/gotica.txt
Output: data/gotica/gotica_clean.txt

Format: One verse per line, no metadata.
"""

import re
import sys


def clean_gothic(input_path: str, output_path: str):
    """
    Extract Gothic text from verses, removing all metadata.

    Args:
        input_path: Path to raw Gothic Bible file
        output_path: Path for cleaned output
    """
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
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

                # Extract text after the closing bracket ]
                # Format: "Mt 5:15 [CA] <Gothic text here>"
                match = re.search(r'\]\s*(.+)$', line)
                if match:
                    gothic_text = match.group(1).strip()
                    # Remove leading dots/ellipses that indicate lacunae
                    gothic_text = re.sub(r'^[\.\s]+', '', gothic_text)
                    if gothic_text:
                        f_out.write(gothic_text + '\n')


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Cleaning Gothic Bible: {input_file} → {output_file}")
    clean_gothic(input_file, output_file)
    print("Done!")
