#!/usr/bin/env python3
"""
Show tokenization of input text using a SentencePiece tokenizer.

Usage:
    echo "Some text" | python tools/show_tokenization.py path/to/spm.model
    head -5 file.txt | python tools/show_tokenization.py tokenizers/got/xglm564m_v16k_s100k_seeded/spm.model
    cat text.txt | python tools/show_tokenization.py path/to/spm.model
"""

import argparse
import sys

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser(
        description='Show tokenization of input text using SentencePiece',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'model_path',
        help='Path to SentencePiece model file (e.g., spm.model)'
    )

    args = parser.parse_args()

    # Read from stdin
    if sys.stdin.isatty():
        print("Error: No input provided. Pipe text into this script.", file=sys.stderr)
        print("Example: echo 'Hello world' | python tools/show_tokenization.py path/to/spm.model", file=sys.stderr)
        sys.exit(1)

    # Load model
    sp = spm.SentencePieceProcessor()
    sp.Load(args.model_path)

    # Process each line
    for line in sys.stdin:
        line = line.rstrip('\n')
        if line:
            pieces = sp.EncodeAsPieces(line)
            print(' '.join(pieces))
        else:
            print()


if __name__ == '__main__':
    main()
