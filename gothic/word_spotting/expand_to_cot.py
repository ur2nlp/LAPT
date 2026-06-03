#!/usr/bin/env python3
"""
Expand canonical word-spotting alignments into chain-of-thought translation data.

Where ``expand_to_instruction.py`` projects each *alignment* into a short
vocabulary query, this module works at the *verse* level: it consumes all of a
verse's trainable alignments to build a single chain-of-thought (CoT)
translation example. The response first glosses a handful of the sentence's key
words, then states the full translation, e.g.

    "siponjos" means "disciples", "iddja" means "went", and "skip" means
    "boat", so the sentence means: "He went from there with his disciples by
    boat."

The aim is to bridge the two skills the model already half-learns separately:
word-level alignment (word spotting) and full-sentence translation. Chaining
the per-word anchors before the full output is meant to make decoding more
input-grounded, attacking the conditioning failure described in
`.claude/gothic/scaling_instruction_tuning.md`. It adds no new information
beyond the existing parallel verses and their alignments — it re-presents them
in a form that forces composition.

CoT translation does *not* by itself supply a knowledge the model lacks; at
inference on out-of-distribution input the intermediate gloss is generated and
may be as wrong as anything else. It should be *mixed* with plain translation
examples (from ``prepare_gothic_data.py``) rather than replacing them, so the
CoT format carries information instead of becoming an unconditional output prior.

Rigidity mitigations (so the format is not memorised as a rote template):
    - The number of glossed words is sampled per example (1..all trainable).
    - Gloss items are rendered with several phrasings, some with quotes and some
      without, since the model is observably sensitive to quote presence.
    - Two join styles ("a, b, and c" vs. "a. b. c.") and several prompt and
      conclusion templates, all chosen with the seeded RNG.

Input format (canonical {train,test}_alignments.jsonl from assign_alignment_ids):
    {"sentence_id": "...", "english_sentence": "...",
     "gothic_sentence_roman": "...", "gothic_sentence_gothic": "...",
     "alignments": [{"alignment_id": "...", "status": "...",
                      "target_word": "...", "gothic_word_roman": "...",
                      "gothic_word_gothic": "..."}, ...]}

Output format (instruction_jsonl, one line per example):
    {"prompt": "...\\nResponse:", "response": " ..."}

Usage:
    python -m gothic.word_spotting.expand_to_cot \
        --input data/gothic_word_spotting/train_alignments.jsonl \
        --output data/gothic_word_spotting/train_cot_translation.jsonl
"""

import argparse
import json
import random
import string
import sys
from pathlib import Path


# Alignment statuses that are trustworthy enough to train on.
TRAINABLE_STATUSES = ("verified_correct", "kept_edited")

# Punctuation stripped from sentence tokens when locating a word's position.
PUNCTUATION = string.punctuation + "·"

SCRIPT_FIELDS = {
    "roman": ("gothic_sentence_roman", "gothic_word_roman"),
    "gothic": ("gothic_sentence_gothic", "gothic_word_gothic"),
}

DIRECTIONS = ("got2eng", "eng2got")

# Prompt templates per direction. {source_sentence} is filled with the
# source-language sentence; the language label is fixed by the direction.
PROMPT_TEMPLATES = {
    "got2eng": [
        (
            "Translate the key words, then translate the whole Gothic sentence "
            "into English.\n"
            "Gothic: {source_sentence}\n"
            "Response:"
        ),
        (
            "Identify the meaning of the important words, then give the full "
            "English translation of this Gothic sentence.\n"
            "Gothic: {source_sentence}\n"
            "Response:"
        ),
        (
            "Work through this Gothic sentence word by word, then give its "
            "English meaning.\n"
            "Gothic: {source_sentence}\n"
            "Response:"
        ),
    ],
    "eng2got": [
        (
            "Translate the key words, then translate the whole sentence into "
            "Gothic.\n"
            "English: {source_sentence}\n"
            "Response:"
        ),
        (
            "Identify the Gothic for the important words, then give the full "
            "Gothic translation of this sentence.\n"
            "English: {source_sentence}\n"
            "Response:"
        ),
        (
            "Work through this sentence word by word, then give its Gothic "
            "translation.\n"
            "English: {source_sentence}\n"
            "Response:"
        ),
    ],
}

# Gloss-item phrasings. {a} is the source-language word, {b} its translation.
# Quote presence is varied deliberately.
GLOSS_ITEM_STYLES = [
    '"{a}" means "{b}"',
    '"{a}" means {b}',
    '{a} means "{b}"',
    '{a} = {b}',
]

# Conclusion templates per direction. {full} is the full target translation.
CONCLUSION_TEMPLATES = {
    "got2eng": [
        'so the sentence means: "{full}"',
        'so in English this is: "{full}"',
        'therefore the full translation is "{full}"',
        "so the whole sentence means {full}",
    ],
    "eng2got": [
        'so the sentence in Gothic is: "{full}"',
        'so in Gothic: "{full}"',
        'therefore the full Gothic translation is "{full}"',
        "so the whole sentence is {full}",
    ],
}


def trainable_alignments(entry: dict) -> list[dict]:
    """Return the entry's alignments whose status is trustworthy for training.

    Args:
        entry: A canonical alignments JSONL entry.

    Returns:
        The sublist of alignments with a status in ``TRAINABLE_STATUSES``. The
        canonical schema carries a ``status`` on every alignment; older
        finalized files without it are treated as all-trainable.
    """
    kept = []
    for alignment in entry["alignments"]:
        status = alignment.get("status")
        if status is None or status in TRAINABLE_STATUSES:
            kept.append(alignment)
    return kept


def token_position(sentence: str, word: str, case_insensitive: bool) -> int:
    """Find the index of a word among a sentence's whitespace tokens.

    Tokens are compared with leading/trailing punctuation stripped. Used only to
    order gloss items in natural reading order, so a miss is non-fatal.

    Args:
        sentence: The source sentence.
        word: The surface form to locate.
        case_insensitive: Whether to compare case-insensitively (English side).

    Returns:
        The zero-based token index of the first match, or a large sentinel
        (``len(tokens)``) when the word is not found, which sorts it to the end.
    """
    tokens = sentence.split()
    needle = word.lower() if case_insensitive else word
    for index, token in enumerate(tokens):
        candidate = token.strip(PUNCTUATION)
        if case_insensitive:
            candidate = candidate.lower()
        if candidate == needle:
            return index
    return len(tokens)


def gloss_pair(
    alignment: dict,
    direction: str,
    word_field: str,
) -> tuple[str, str]:
    """Return the (source_word, translation) pair for one gloss item.

    Args:
        alignment: A single alignment object.
        direction: "got2eng" or "eng2got".
        word_field: The Gothic word field for the active script.

    Returns:
        ``(a, b)`` where ``a`` is the word in the source sentence's language and
        ``b`` is its translation, matching the direction.
    """
    gothic_word = alignment[word_field]
    english_word = alignment["target_word"]
    if direction == "got2eng":
        return gothic_word, english_word
    return english_word, gothic_word


def render_response(
    items: list[tuple[str, str]],
    full_translation: str,
    direction: str,
    rng: random.Random,
) -> str:
    """Render the CoT response: a gloss chain followed by the full translation.

    A single gloss-item style and a single join style are chosen per response
    (consistency reads more naturally than mixing styles within one sentence),
    while the choices vary across examples.

    Args:
        items: Ordered (source_word, translation) gloss pairs.
        full_translation: The full target-language sentence.
        direction: "got2eng" or "eng2got".
        rng: Seeded RNG.

    Returns:
        The response string (without the leading space added by the caller).
    """
    item_style = rng.choice(GLOSS_ITEM_STYLES)
    rendered_items = [
        item_style.format(a=source_word, b=translation)
        for source_word, translation in items
    ]

    conclusion = rng.choice(CONCLUSION_TEMPLATES[direction]).format(
        full=full_translation,
    )

    join_style = rng.choice(["comma_and", "period"])
    if join_style == "comma_and":
        if len(rendered_items) == 1:
            gloss_clause = rendered_items[0]
        else:
            gloss_clause = (
                ", ".join(rendered_items[:-1]) + ", and " + rendered_items[-1]
            )
        return f"{gloss_clause}, {conclusion}"

    # period style: each gloss is its own sentence and the conclusion is
    # capitalised to start a new one.
    gloss_clause = ". ".join(rendered_items)
    capitalised_conclusion = conclusion[0].upper() + conclusion[1:]
    return f"{gloss_clause}. {capitalised_conclusion}"


def make_cot_example(
    entry: dict,
    direction: str,
    script_key: str,
    rng: random.Random,
    min_words: int,
) -> dict | None:
    """Build one CoT translation example for a verse in one direction/script.

    Args:
        entry: A canonical alignments entry.
        direction: "got2eng" or "eng2got".
        script_key: "roman" or "gothic".
        rng: Seeded RNG for word subset, ordering-independent style choices.
        min_words: Minimum number of words to gloss.

    Returns:
        A {"prompt": ..., "response": ...} dict, or None if the verse has fewer
        than ``min_words`` trainable alignments.
    """
    alignments = trainable_alignments(entry)
    if len(alignments) < min_words:
        return None

    sentence_field, word_field = SCRIPT_FIELDS[script_key]
    gothic_sentence = entry[sentence_field]
    english_sentence = entry["english_sentence"]

    if direction == "got2eng":
        source_sentence = gothic_sentence
        full_translation = english_sentence
        order_case_insensitive = False
    else:
        source_sentence = english_sentence
        full_translation = gothic_sentence
        order_case_insensitive = True

    # sample how many words to gloss, then which ones
    count = rng.randint(min_words, len(alignments))
    chosen = rng.sample(alignments, count)

    # order gloss items by their source word's position in the source sentence
    def sort_key(alignment: dict) -> int:
        source_word, _ = gloss_pair(alignment, direction, word_field)
        return token_position(source_sentence, source_word, order_case_insensitive)

    chosen.sort(key=sort_key)
    items = [gloss_pair(alignment, direction, word_field) for alignment in chosen]

    prompt = rng.choice(PROMPT_TEMPLATES[direction]).format(
        source_sentence=source_sentence,
    )
    response = render_response(items, full_translation, direction, rng)
    return {"prompt": prompt, "response": f" {response}"}


def expand_entry(
    entry: dict,
    directions: list[str],
    scripts: list[str],
    pick_one_script: bool,
    variants: int,
    min_words: int,
    rng: random.Random,
) -> tuple[list[dict], int]:
    """Expand one verse into CoT translation examples.

    Args:
        entry: A canonical alignments entry.
        directions: Translation directions to emit.
        scripts: Script keys to consider ("roman", "gothic").
        pick_one_script: If True, choose one script per (direction, variant);
            otherwise emit every script.
        variants: Number of independently sampled examples per direction/script.
        min_words: Minimum number of words to gloss.
        rng: Seeded RNG.

    Returns:
        An (examples, skipped) tuple, where skipped counts direction/script/
        variant combinations that could not be built (too few alignments).
    """
    examples: list[dict] = []
    skipped = 0
    for direction in directions:
        for _ in range(variants):
            if pick_one_script:
                selected_scripts = [rng.choice(scripts)]
            else:
                selected_scripts = scripts
            for script_key in selected_scripts:
                example = make_cot_example(
                    entry,
                    direction,
                    script_key,
                    rng,
                    min_words,
                )
                if example is None:
                    skipped += 1
                else:
                    examples.append(example)
    return examples, skipped


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Expand canonical word-spotting alignments into chain-of-thought "
            "translation instruction examples."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to canonical alignments JSONL (verse-grouped, with status).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for instruction JSONL. Defaults to stdout.",
    )
    parser.add_argument(
        "--script",
        choices=["roman", "gothic", "both", "random"],
        default="both",
        help=(
            "Which Gothic script(s) to use. 'both' emits one example per script "
            "per direction per variant. 'random' picks one script. (default: both)"
        ),
    )
    parser.add_argument(
        "--directions",
        choices=["got2eng", "eng2got", "both"],
        default="both",
        help="Translation direction(s) to emit (default: both).",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help=(
            "Independently sampled CoT examples per direction per script. "
            "Keep small to avoid overfitting verse memory (default: 1)."
        ),
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Minimum number of words to gloss per example (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for subset/template/style sampling (default: 1).",
    )
    args = parser.parse_args()

    if args.variants < 1:
        print("Error: --variants must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.min_words < 1:
        print("Error: --min-words must be >= 1.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if args.directions == "both":
        directions = list(DIRECTIONS)
    else:
        directions = [args.directions]

    if args.script == "random":
        scripts = ["roman", "gothic"]
        pick_one_script = True
    elif args.script == "both":
        scripts = ["roman", "gothic"]
        pick_one_script = False
    else:
        scripts = [args.script]
        pick_one_script = False

    rng = random.Random(args.seed)

    examples: list[dict] = []
    total_skipped = 0
    for entry in entries:
        entry_examples, skipped = expand_entry(
            entry,
            directions,
            scripts,
            pick_one_script,
            args.variants,
            args.min_words,
            rng,
        )
        examples.extend(entry_examples)
        total_skipped += skipped

    if args.output:
        out_file = open(args.output, "w", encoding="utf-8")
    else:
        out_file = sys.stdout

    try:
        for example in examples:
            out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out_file.close()

    skipped_note = f", {total_skipped} skipped" if total_skipped else ""
    print(
        f"Expanded {len(entries)} verses into {len(examples)} CoT translation "
        f"examples (directions={','.join(directions)}, script={args.script}, "
        f"variants={args.variants}{skipped_note})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
