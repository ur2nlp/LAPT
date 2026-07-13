#!/usr/bin/env python3
"""
Unified Gothic data preparation script with train/dev/test splitting.

Creates augmented Gothic training data with three types:
1. Monolingual: Gothic text in Roman and/or Gothic script
2. Transliteration: Parallel Roman ↔ Gothic script examples
3. Translation: Parallel Gothic ↔ English examples

Splits at the verse level to prevent near-duplicate leakage across splits.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from gothic.orthography import clean_gothic_artifacts, transliterate_latin_to_gothic

# Mapping from Gothic abbreviations to WEB book names (for translation alignment)
GOTHIC_TO_WEB_BOOKS = {
    'Mt': 'Matthew', 'Mk': 'Mark', 'Lk': 'Luke', 'Jo': 'John',
    'Rm': 'Romans', 'Co1': '1 Corinthians', 'Co2': '2 Corinthians',
    'Ga': 'Galatians', 'Ef': 'Ephesians', 'Fp': 'Philippians',
    'Cl': 'Colossians', 'Th1': '1 Thessalonians', 'Th2': '2 Thessalonians',
    'Ti1': '1 Timothy', 'Ti2': '2 Timothy', 'Tt': 'Titus', 'Phm': 'Philemon',
    'Hb': 'Hebrews', 'Jc': 'James', 'Pe1': '1 Peter', 'Pe2': '2 Peter',
    'Jo1': '1 John', 'Jo2': '2 John', 'Jo3': '3 John', 'Jd': 'Jude',
    'Ap': 'Revelation', 'Neh': 'Nehemiah',
}


def parse_gothic_bible(gothic_file: str) -> Dict[Tuple[str, int, int], List[str]]:
    """
    Parse Gothic Bible and extract all codex variants for each verse.

    Returns:
        Dictionary mapping (book, chapter, verse) to list of codex variant texts
    """
    verse_variants: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)

    with open(gothic_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#') or line.startswith('Cal '):
                continue

            # Match verse pattern: Mt 5:15 [CA] <text>
            match = re.match(r'^([A-Za-z0-9]+)\s+(\d+):(\d+)\s+\[[A-Z]+\]\s+(.+)$', line)
            if match:
                book_abbrev = match.group(1)
                chapter = int(match.group(2))
                verse = int(match.group(3))
                text = match.group(4).strip()

                # Normalize lacunae markers
                text = re.sub(r'\.(\s*\.)+', '...', text)

                # Remove meta-brackets
                text = text.replace('<', '').replace('>', '')
                text = text.replace('[', '').replace(']', '')

                # Strip editorial artifacts (restoration parens, numeral glosses,
                # enclitic tildes, rare diacritics) from the Gothic surface. Runs
                # before transliteration so nothing leaks untransliterated into
                # the Gothic-script outputs. See gothic.orthography.
                text = clean_gothic_artifacts(text)

                if text and text != '...':
                    # Use standardized book name if available
                    if book_abbrev in GOTHIC_TO_WEB_BOOKS:
                        book = GOTHIC_TO_WEB_BOOKS[book_abbrev]
                    else:
                        book = book_abbrev

                    verse_id = (book, chapter, verse)
                    verse_variants[verse_id].append(text)

    return verse_variants


def parse_english_bible(web_file: str) -> Dict[Tuple[str, int, int], str]:
    """
    Parse World English Bible and extract verses.

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

            if current_book:
                # Check for verse: 001:001 <text>
                verse_match = re.match(r'^(\d{3}):(\d{3})\s+(.+)$', line)
                if verse_match:
                    chapter = int(verse_match.group(1))
                    verse = int(verse_match.group(2))
                    text = verse_match.group(3).strip()

                    # Remove inline footnotes
                    text = re.sub(r'\{[^}]+\}', '', text).strip()

                    if text:
                        verses[(current_book, chapter, verse)] = text

                # Handle continuation lines (indented)
                elif line.startswith('        ') and verses:
                    last_key = list(verses.keys())[-1]
                    if last_key[0] == current_book:
                        continuation = re.sub(r'\{[^}]+\}', '', line.strip())
                        if continuation:
                            verses[last_key] += ' ' + continuation

    # Final cleanup pass: remove any remaining footnotes and collapse whitespace
    for verse_id in verses:
        text = verses[verse_id]
        # Remove footnotes (in case any were missed or span multiple parts)
        text = re.sub(r'\{[^}]*\}', '', text)
        # Collapse multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        # Clean up any leading/trailing whitespace
        verses[verse_id] = text.strip()

    return verses


def sample_one_variant_per_verse(
    verse_variants: Dict[Tuple[str, int, int], List[str]],
    seed: int
) -> Dict[Tuple[str, int, int], List[str]]:
    """
    Sample one codex variant per verse for deduplication.

    Returns:
        Dictionary with single-element lists per verse
    """
    random.seed(seed)
    sampled = {}

    for verse_id, variants in verse_variants.items():
        chosen = random.choice(variants)
        sampled[verse_id] = [chosen]

    return sampled


def construct_filename(
    data_type: str,
    sample_one_codex: bool,
    split_name: str = None,
    monolingual_script: str = None,
    transliteration_direction: str = None,
    translation_script: str = None,
    translation_direction: str = None,
    extension: str = 'txt'
) -> str:
    """
    Construct output filename encoding all multiplier settings.

    Format: {data_type}_{codex}_{type-specific-multipliers}[_{split}].{ext}

    Args:
        data_type: One of 'monolingual', 'transliteration', 'translation'
        sample_one_codex: Whether sampling one codex variant
        split_name: Optional split name ('train', 'dev', 'test')
        (other args): Type-specific multiplier settings
        extension: File extension ('txt' or 'jsonl')

    Returns:
        Filename string
    """
    parts = [data_type]

    # Codex handling (applies to all types)
    codex_part = "one-codex" if sample_one_codex else "all-codices"
    parts.append(codex_part)

    # Type-specific multipliers
    if data_type == 'monolingual':
        if monolingual_script == 'roman':
            parts.append('roman')
        elif monolingual_script == 'gothic':
            parts.append('gothic')
        else:  # both
            parts.append('both-scripts')

    elif data_type == 'transliteration':
        if transliteration_direction == 'roman_to_gothic':
            parts.append('roman-to-gothic')
        elif transliteration_direction == 'gothic_to_roman':
            parts.append('gothic-to-roman')
        else:  # both
            parts.append('both-directions')

    elif data_type == 'translation':
        # Script
        if translation_script == 'roman':
            parts.append('roman')
        elif translation_script == 'gothic':
            parts.append('gothic')
        else:  # both
            parts.append('both-scripts')

        # Direction
        if translation_direction == 'eng_to_gothic':
            parts.append('eng-to-gothic')
        elif translation_direction == 'gothic_to_eng':
            parts.append('gothic-to-eng')
        else:  # both
            parts.append('both-directions')

    # Add split name if provided
    if split_name:
        parts.append(split_name)

    return '_'.join(parts) + '.' + extension


def split_verse_ids(
    verse_ids: List[Tuple[str, int, int]],
    seed: int,
    splits: List[str],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float
) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Split verse IDs into train/dev/test sets.

    Returns:
        Dictionary mapping split name to list of verse IDs
    """
    # Normalize ratios
    active_ratios = []
    if 'train' in splits:
        active_ratios.append(('train', train_ratio))
    if 'dev' in splits:
        active_ratios.append(('dev', dev_ratio))
    if 'test' in splits:
        active_ratios.append(('test', test_ratio))

    total = sum(r for _, r in active_ratios)
    normalized_ratios = [(name, r / total) for name, r in active_ratios]

    # Shuffle deterministically
    sorted_ids = sorted(verse_ids)
    random.seed(seed)
    random.shuffle(sorted_ids)

    # Split
    result = {}
    start_idx = 0
    for i, (split_name, ratio) in enumerate(normalized_ratios):
        if i == len(normalized_ratios) - 1:
            # Last split gets remainder
            result[split_name] = sorted_ids[start_idx:]
        else:
            count = int(len(sorted_ids) * ratio)
            result[split_name] = sorted_ids[start_idx:start_idx + count]
            start_idx += count

    return result


def generate_monolingual(
    verse_ids: List[Tuple[str, int, int]],
    gothic_verses: Dict[Tuple[str, int, int], List[str]],
    output_path: str,
    script: str
):
    """Generate monolingual Gothic data."""
    lines_written = 0
    verse_instances = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for verse_id in sorted(verse_ids):
            if verse_id not in gothic_verses:
                continue

            for variant in gothic_verses[verse_id]:
                verse_instances += 1

                # Write Roman script version
                if script in ['roman', 'both']:
                    f.write(variant + '\n')
                    lines_written += 1

                # Write Gothic script version
                if script in ['gothic', 'both']:
                    gothic_text = transliterate_latin_to_gothic(variant)
                    f.write(gothic_text + '\n')
                    lines_written += 1

    return lines_written, verse_instances


# Instruction phrasings per task, for prompt-side diversification of the
# instruction-tuning data. {input} is the source text; for the English -> Gothic
# direction {target} is the Gothic-script label ("Gothic" / "Romanized Gothic").
# The original fixed phrasing is kept first in each list so it stays represented.
# Diversifying the *prompt* (phrasing + input quoting) makes the model robust to
# how a task is asked; the *response* is deliberately left a single canonical
# bare form so input quoting does not get echoed into the output. See
# `.claude/gothic/scaling_instruction_tuning.md`.
INSTRUCTION_TEMPLATES = {
    "to_gothic_script": [
        "Transliterate to Gothic script: {input}",
        "Convert the following to Gothic script: {input}",
        "Rewrite this in the Gothic alphabet: {input}",
        "Write the following in Gothic script: {input}",
    ],
    "to_latin_script": [
        "Transliterate to Latin script: {input}",
        "Convert the following to Latin script: {input}",
        "Rewrite this in the Latin alphabet: {input}",
        "Write the following in Latin script: {input}",
    ],
    "to_target": [
        "Translate to {target}: {input}",
        "Translate the following into {target}: {input}",
        "Give the {target} translation of: {input}",
        "Render the following in {target}: {input}",
    ],
    "to_english": [
        "Translate to English: {input}",
        "Translate the following into English: {input}",
        "Give the English translation of: {input}",
        "Render the following in English: {input}",
    ],
}


def build_instruction_prompt(
    task_key: str,
    input_text: str,
    rng: random.Random,
    target_label: str = None,
) -> str:
    """Build a varied instruction prompt ending in ' Response:'.

    Diversifies two axes independently with the seeded RNG: the instruction
    phrasing (one of several templates) and whether the input is wrapped in
    double quotes. The caller always pairs this with a single canonical bare
    response, which decouples input quoting from output quoting so the model
    learns to ignore input quotes rather than echo them.

    Quote wrapping is skipped when the input already contains a double quote
    (common in Bible verses) to avoid confusing nested quotes.

    Args:
        task_key: One of the keys in INSTRUCTION_TEMPLATES.
        input_text: The source text to embed in the prompt.
        rng: Seeded RNG for the template choice and the quote coin flip.
        target_label: The {target} substitution for the 'to_target' task
            ("Gothic" or "Romanized Gothic"); ignored for other tasks.

    Returns:
        The prompt string ending with ' Response:'.
    """
    template = rng.choice(INSTRUCTION_TEMPLATES[task_key])
    presented_input = input_text
    if '"' not in input_text and rng.random() < 0.5:
        presented_input = f'"{input_text}"'
    instruction = template.format(input=presented_input, target=target_label)
    return f"{instruction} Response:"


def generate_transliteration(
    verse_ids: List[Tuple[str, int, int]],
    gothic_verses: Dict[Tuple[str, int, int], List[str]],
    output_path: str,
    direction: str,
    instruction_format: bool,
    delimiter: str = None,
    output_format: str = 'plaintext',
    rng: random.Random = None
):
    """Generate transliteration parallel data."""
    lines_written = 0
    verse_instances = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for verse_id in sorted(verse_ids):
            if verse_id not in gothic_verses:
                continue

            for variant in gothic_verses[verse_id]:
                verse_instances += 1
                roman_text = variant
                gothic_text = transliterate_latin_to_gothic(variant)

                # Roman → Gothic
                if direction in ['roman_to_gothic', 'both']:
                    if instruction_format and output_format == 'jsonl':
                        prompt = build_instruction_prompt("to_gothic_script", roman_text, rng)
                        response = f" {gothic_text}"
                        f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + '\n')
                    elif instruction_format:
                        prompt = build_instruction_prompt("to_gothic_script", roman_text, rng)
                        example = f"{prompt} {gothic_text}"
                        f.write(' '.join(example.split()) + '\n')
                    else:
                        separator = delimiter if delimiter else ' '
                        example = f"{roman_text}{separator}{gothic_text}"
                        if delimiter:
                            parts = example.split(delimiter)
                            parts = [' '.join(part.split()) for part in parts]
                            collapsed = delimiter.join(parts)
                        else:
                            collapsed = ' '.join(example.split())
                        f.write(collapsed + '\n')
                    lines_written += 1

                # Gothic → Roman
                if direction in ['gothic_to_roman', 'both']:
                    if instruction_format and output_format == 'jsonl':
                        prompt = build_instruction_prompt("to_latin_script", gothic_text, rng)
                        response = f" {roman_text}"
                        f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + '\n')
                    elif instruction_format:
                        prompt = build_instruction_prompt("to_latin_script", gothic_text, rng)
                        example = f"{prompt} {roman_text}"
                        f.write(' '.join(example.split()) + '\n')
                    else:
                        separator = delimiter if delimiter else ' '
                        example = f"{gothic_text}{separator}{roman_text}"
                        if delimiter:
                            parts = example.split(delimiter)
                            parts = [' '.join(part.split()) for part in parts]
                            collapsed = delimiter.join(parts)
                        else:
                            collapsed = ' '.join(example.split())
                        f.write(collapsed + '\n')
                    lines_written += 1

    return lines_written, verse_instances


def generate_translation(
    verse_ids: List[Tuple[str, int, int]],
    gothic_verses: Dict[Tuple[str, int, int], List[str]],
    english_verses: Dict[Tuple[str, int, int], str],
    output_path: str,
    script: str,
    direction: str,
    instruction_format: bool,
    delimiter: str = None,
    output_format: str = 'plaintext',
    rng: random.Random = None
):
    """Generate translation parallel data."""
    lines_written = 0
    verse_instances = 0
    skipped = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for verse_id in sorted(verse_ids):
            if verse_id not in gothic_verses:
                continue

            # Skip if no English translation
            if verse_id not in english_verses:
                skipped += 1
                continue

            english_text = english_verses[verse_id]

            for variant in gothic_verses[verse_id]:
                verse_instances += 1
                # Generate examples for each requested script
                scripts_to_generate = []
                if script in ['roman', 'both']:
                    scripts_to_generate.append(('roman', variant))
                if script in ['gothic', 'both']:
                    gothic_script_text = transliterate_latin_to_gothic(variant)
                    scripts_to_generate.append(('gothic', gothic_script_text))

                for script_name, gothic_text in scripts_to_generate:
                    # English → Gothic
                    if direction in ['eng_to_gothic', 'both']:
                        # Use script-specific prompt for English → Gothic
                        target_lang = "Romanized Gothic" if script_name == 'roman' else "Gothic"
                        if instruction_format and output_format == 'jsonl':
                            prompt = build_instruction_prompt("to_target", english_text, rng, target_label=target_lang)
                            response = f" {gothic_text}"
                            f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + '\n')
                        elif instruction_format:
                            prompt = build_instruction_prompt("to_target", english_text, rng, target_label=target_lang)
                            example = f"{prompt} {gothic_text}"
                            f.write(' '.join(example.split()) + '\n')
                        else:
                            separator = delimiter if delimiter else ' '
                            example = f"{english_text}{separator}{gothic_text}"
                            if delimiter:
                                parts = example.split(delimiter)
                                parts = [' '.join(part.split()) for part in parts]
                                collapsed = delimiter.join(parts)
                            else:
                                collapsed = ' '.join(example.split())
                            f.write(collapsed + '\n')
                        lines_written += 1

                    # Gothic → English
                    if direction in ['gothic_to_eng', 'both']:
                        if instruction_format and output_format == 'jsonl':
                            prompt = build_instruction_prompt("to_english", gothic_text, rng)
                            response = f" {english_text}"
                            f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + '\n')
                        elif instruction_format:
                            prompt = build_instruction_prompt("to_english", gothic_text, rng)
                            example = f"{prompt} {english_text}"
                            f.write(' '.join(example.split()) + '\n')
                        else:
                            separator = delimiter if delimiter else ' '
                            example = f"{gothic_text}{separator}{english_text}"
                            if delimiter:
                                parts = example.split(delimiter)
                                parts = [' '.join(part.split()) for part in parts]
                                collapsed = delimiter.join(parts)
                            else:
                                collapsed = ' '.join(example.split())
                            f.write(collapsed + '\n')
                        lines_written += 1

    return lines_written, verse_instances, skipped


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Gothic training data with train/dev/test splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate train + test splits (90/10) with all data types
  python -m gothic.data.prepare_gothic_data --splits train test --train-ratio 0.9 --test-ratio 0.1
  # Output: monolingual_all-codices_both-scripts_train.txt, etc.

  # Generate only training data (no split suffix in filenames)
  python -m gothic.data.prepare_gothic_data --splits train
  # Output: monolingual_all-codices_both-scripts.txt, etc.

  # Generate translation data with delimiter for LLM processing
  python -m gothic.data.prepare_gothic_data --data-types translation --translation-direction eng_to_gothic --delimiter ' | '
  # Output: English text | Gothic text (one per line)

  # Generate only Roman script monolingual with one codex per verse
  python -m gothic.data.prepare_gothic_data --data-types monolingual --monolingual-script roman --sample-one-codex
  # Output: monolingual_one-codex_roman.txt

  # Generate transliteration data with instruction format (plaintext)
  python -m gothic.data.prepare_gothic_data --data-types transliteration --instruction-format
  # Output: transliteration_all-codices_both-directions.txt
  # Format: Transliterate to Gothic script: {roman} Response: {gothic}

  # Generate translation data with JSONL format for instruction tuning with loss masking
  python -m gothic.data.prepare_gothic_data --data-types translation --instruction-format --output-format jsonl
  # Output: translation_all-codices_both-scripts_both-directions.jsonl
  # Format: {"prompt": "Translate to Gothic: {english} Response:", "response": " {gothic_script}"}
  #     or: {"prompt": "Translate to Romanized Gothic: {english} Response:", "response": " {roman}"}
        """
    )

    # Input files
    parser.add_argument(
        '--gothic',
        default='data/gotica/gotica.txt',
        help='Path to Gothic Bible file (default: data/gotica/gotica.txt)'
    )
    parser.add_argument(
        '--english',
        default='data/web_bible/web_bible.txt',
        help='Path to English Bible file (default: data/web_bible/web_bible.txt)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/gothic_prepared',
        help='Output directory for prepared data (default: data/gothic_prepared)'
    )

    # Splitting configuration
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=['train', 'dev', 'test'],
        default=['train', 'dev', 'test'],
        help='Which splits to generate (default: train dev test)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed for reproducible splitting (default: 1)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Proportion for training split (default: 0.8)'
    )
    parser.add_argument(
        '--dev-ratio',
        type=float,
        default=0.1,
        help='Proportion for dev split (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Proportion for test split (default: 0.1)'
    )

    # Data type selection
    parser.add_argument(
        '--data-types',
        nargs='+',
        choices=['monolingual', 'transliteration', 'translation'],
        default=['monolingual', 'transliteration', 'translation'],
        help='Which data types to generate (default: all three)'
    )

    # Global options
    parser.add_argument(
        '--sample-one-codex',
        action='store_true',
        help='Sample one codex variant per verse (default: keep all variants)'
    )
    parser.add_argument(
        '--instruction-format',
        action='store_true',
        help='Use instruction tuning format (default: simple concatenation)'
    )
    parser.add_argument(
        '--output-format',
        choices=['plaintext', 'jsonl'],
        default='plaintext',
        help='Output format: "plaintext" (one example per line) or "jsonl" (separate prompt/response fields for loss masking). JSONL only applies when --instruction-format is set. (default: plaintext)'
    )
    parser.add_argument(
        '--delimiter',
        type=str,
        default=None,
        help='Delimiter between parallel sentences (e.g., " | " or " ||| "). Only used with simple concatenation format (not instruction format). (default: single space)'
    )

    # Monolingual options
    parser.add_argument(
        '--monolingual-script',
        choices=['roman', 'gothic', 'both'],
        default='both',
        help='Script(s) for monolingual data (default: both)'
    )

    # Transliteration options
    parser.add_argument(
        '--transliteration-direction',
        choices=['roman_to_gothic', 'gothic_to_roman', 'both'],
        default='both',
        help='Direction(s) for transliteration (default: both)'
    )

    # Translation options
    parser.add_argument(
        '--translation-script',
        choices=['roman', 'gothic', 'both'],
        default='both',
        help='Script(s) for Gothic in translation (default: both)'
    )
    parser.add_argument(
        '--translation-direction',
        choices=['eng_to_gothic', 'gothic_to_eng', 'both'],
        default='both',
        help='Direction(s) for translation (default: both)'
    )

    args = parser.parse_args()

    # Warn if jsonl format specified without instruction format
    if args.output_format == 'jsonl' and not args.instruction_format:
        print(
            "Warning: --output-format jsonl has no effect without --instruction-format. "
            "Using plaintext output.",
            file=sys.stderr
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse Gothic Bible
    print("Parsing Gothic Bible...", file=sys.stderr)
    gothic_verses = parse_gothic_bible(args.gothic)
    print(f"  Found {len(gothic_verses)} unique verses", file=sys.stderr)
    total_variants = sum(len(variants) for variants in gothic_verses.values())
    duplicate_count = sum(1 for variants in gothic_verses.values() if len(variants) > 1)
    print(f"  Total variants: {total_variants}", file=sys.stderr)
    print(f"  Verses with multiple codices: {duplicate_count}", file=sys.stderr)

    # Handle codex deduplication
    if args.sample_one_codex:
        print("\nSampling one variant per verse...", file=sys.stderr)
        gothic_verses = sample_one_variant_per_verse(gothic_verses, args.seed)
        print(f"  Keeping {len(gothic_verses)} verses (1 variant each)", file=sys.stderr)
    else:
        print(f"\nKeeping all codex variants", file=sys.stderr)

    # Parse English Bible (if needed)
    english_verses = {}
    if 'translation' in args.data_types:
        print("\nParsing English Bible...", file=sys.stderr)
        english_verses = parse_english_bible(args.english)
        print(f"  Found {len(english_verses)} English verses", file=sys.stderr)

    # Split verse IDs
    print(f"\nSplitting verses (seed={args.seed})...", file=sys.stderr)
    verse_ids = list(gothic_verses.keys())
    split_dict = split_verse_ids(
        verse_ids, args.seed, args.splits,
        args.train_ratio, args.dev_ratio, args.test_ratio
    )

    for split_name in args.splits:
        count = len(split_dict[split_name])
        print(f"  {split_name}: {count} verses", file=sys.stderr)

    # Generate data for each split and data type
    print("\nGenerating data...", file=sys.stderr)
    stats = {}

    # Seeded RNG for prompt-side diversification (phrasing + input quoting) of
    # the instruction-tuning formats. Independent of the split/codex RNG stream.
    prompt_rng = random.Random(args.seed)

    for split_name in args.splits:
        verse_ids_for_split = split_dict[split_name]

        for data_type in args.data_types:
            # Construct filename encoding all settings
            # Omit split suffix if only generating train
            split_suffix = split_name if len(args.splits) > 1 else None

            # Determine file extension based on output format
            # JSONL only applies to instruction format for transliteration/translation
            use_jsonl = (
                args.output_format == 'jsonl'
                and args.instruction_format
                and data_type in ['transliteration', 'translation']
            )
            extension = 'jsonl' if use_jsonl else 'txt'

            filename = construct_filename(
                data_type=data_type,
                sample_one_codex=args.sample_one_codex,
                split_name=split_suffix,
                monolingual_script=args.monolingual_script,
                transliteration_direction=args.transliteration_direction,
                translation_script=args.translation_script,
                translation_direction=args.translation_direction,
                extension=extension
            )
            output_path = output_dir / filename

            # Generate data based on type
            if data_type == 'monolingual':
                lines, verse_instances = generate_monolingual(
                    verse_ids_for_split, gothic_verses,
                    str(output_path), args.monolingual_script
                )
                stats[str(output_path)] = {
                    'unique_verses': len(verse_ids_for_split),
                    'verse_instances': verse_instances,
                    'lines': lines
                }

            elif data_type == 'transliteration':
                lines, verse_instances = generate_transliteration(
                    verse_ids_for_split, gothic_verses,
                    str(output_path), args.transliteration_direction,
                    args.instruction_format, args.delimiter,
                    args.output_format, prompt_rng
                )
                stats[str(output_path)] = {
                    'unique_verses': len(verse_ids_for_split),
                    'verse_instances': verse_instances,
                    'lines': lines
                }

            elif data_type == 'translation':
                lines, verse_instances, skipped = generate_translation(
                    verse_ids_for_split, gothic_verses, english_verses,
                    str(output_path), args.translation_script,
                    args.translation_direction, args.instruction_format,
                    args.delimiter, args.output_format, prompt_rng
                )
                stats[str(output_path)] = {
                    'unique_verses': len(verse_ids_for_split),
                    'verse_instances': verse_instances,
                    'lines': lines,
                    'skipped': skipped
                }

    # Print statistics
    print("\n" + "="*70, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("="*70, file=sys.stderr)

    for path, info in stats.items():
        filename = Path(path).name
        unique = info['unique_verses']
        instances = info['verse_instances']
        lines = info['lines']

        if 'skipped' in info:
            skipped = info['skipped']
            print(f"{filename}:", file=sys.stderr)
            print(f"  {unique} unique verses → {instances} verse instances → {lines} lines", file=sys.stderr)
            if skipped > 0:
                print(f"  ({skipped} verses skipped, no English match)", file=sys.stderr)
        else:
            print(f"{filename}:", file=sys.stderr)
            print(f"  {unique} unique verses → {instances} verse instances → {lines} lines", file=sys.stderr)

    print("\nDone!", file=sys.stderr)


if __name__ == '__main__':
    main()
