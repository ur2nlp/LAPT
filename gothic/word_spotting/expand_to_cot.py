#!/usr/bin/env python3
"""
Expand canonical word-spotting alignments into chain-of-thought translation data.

Where ``expand_to_instruction.py`` projects each *alignment* into a short
vocabulary query, this module works at the *verse* level: it consumes all of a
verse's trainable alignments to build a single chain-of-thought (CoT)
translation example. The response first glosses a handful of the sentence's key
words, then states the full translation. The gloss wording is direction-specific
so it reads naturally each way, e.g. for got2eng:

    "siponjos" means "disciples", "iddja" means "went", and "skip" means
    "boat", so the sentence means: "He went from there with his disciples by
    boat."

and for eng2got (the target language is named on the first gloss only, then
subsequent glosses are shortened):

    The Gothic for "disciples" is "siponjos", "went" is "iddja", and "boat" is
    "skip", so the sentence in Gothic is: "..."

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

The *output* format is deliberately unified (as of v2.2.0): a single gloss
phrasing per direction, a single conclusion per direction, and one join style
(comma / "a, b, and c"). Keeping the response shape consistent lowers the
task's surface variability so a small model can learn one target form rather
than reverse-engineering an interchangeable set of templates. Robustness is
instead carried on the *input* side, which stays diverse:
    - Several prompt templates per direction, chosen with the seeded RNG.
    - The number of glossed words is sampled per example, with a floor of
      ``min_words`` (default 2) up to all trainable alignments. The floor skews
      the chain toward 2-3 anchors; it is softened to the available count so a
      single-alignment verse still emits a one-word example.
    - Which words are glossed is a random subset per example.

Input format (canonical {train,test}_alignments.jsonl from assign_alignment_ids):
    {"sentence_id": "...", "english_sentence": "...",
     "gothic_sentence_roman": "...", "gothic_sentence_gothic": "...",
     "alignments": [{"alignment_id": "...", "status": "...",
                      "target_word": "...", "gothic_word_roman": "...",
                      "gothic_word_gothic": "..."}, ...]}

Output format (instruction_jsonl, one line per example):
    {"prompt": "... Response:", "response": " ..."}

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

from gothic.instruction_format import flatten_prompt
from gothic.word_spotting.canonical import trainable_alignments

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

# Single canonical conclusion per direction. {full} is the full target
# translation. (Unified in v2.2.0; see module docstring.)
CONCLUSION_TEMPLATES = {
    "got2eng": 'so the sentence means: "{full}"',
    "eng2got": 'so the sentence in Gothic is: "{full}"',
}


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


def render_gloss_items(
    items: list[tuple[str, str]],
    direction: str,
) -> list[str]:
    """Render each (source_word, translation) pair as a gloss clause.

    The phrasing is direction-specific so it reads naturally each way. For
    got2eng, ``"a" means "b"`` reads correctly ("hunds" means "dog"). For
    eng2got the same phrasing reads oddly ("dog" means "hunds"), so the target
    language is named explicitly -- but only on the first gloss, with subsequent
    glosses shortened to avoid a repetitive "The Gothic for ..." on every item.

    Args:
        items: Ordered (source_word, translation) gloss pairs.
        direction: "got2eng" or "eng2got".

    Returns:
        The rendered gloss clauses, one per item, in the given order.
    """
    rendered_items: list[str] = []
    for index, (source_word, translation) in enumerate(items):
        if direction == "got2eng":
            rendered_items.append(f'"{source_word}" means "{translation}"')
        elif index == 0:
            rendered_items.append(
                f'The Gothic for "{source_word}" is "{translation}"'
            )
        else:
            rendered_items.append(f'"{source_word}" is "{translation}"')
    return rendered_items


def render_response(
    items: list[tuple[str, str]],
    full_translation: str,
    direction: str,
) -> str:
    """Render the CoT response: a gloss chain followed by the full translation.

    The output format is unified (v2.2.0): one gloss phrasing per direction, one
    conclusion per direction, joined in a single comma / "a, b, and c" style. No
    per-example randomness is consumed here -- response variability was moved to
    the input side (see module docstring).

    Args:
        items: Ordered (source_word, translation) gloss pairs.
        full_translation: The full target-language sentence.
        direction: "got2eng" or "eng2got".

    Returns:
        The response string (without the leading space added by the caller).
    """
    rendered_items = render_gloss_items(items, direction)
    conclusion = CONCLUSION_TEMPLATES[direction].format(full=full_translation)

    if len(rendered_items) == 1:
        gloss_clause = rendered_items[0]
    else:
        gloss_clause = ", ".join(rendered_items[:-1]) + ", and " + rendered_items[-1]
    return f"{gloss_clause}, {conclusion}"


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
        min_words: Target floor on the number of words to gloss. Softened to the
            verse's trainable-alignment count when the verse has fewer, so a
            verse with a single alignment still emits a one-word example rather
            than being dropped.

    Returns:
        A {"prompt": ..., "response": ...} dict, or None if the verse has no
        trainable alignments.
    """
    alignments = trainable_alignments(entry)
    if not alignments:
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

    # sample how many words to gloss, then which ones. The floor skews the
    # gloss chain toward >=2 anchors (the uniform-from-1 default over-produced
    # single-link examples); it is softened to the available count so
    # single-alignment verses still contribute.
    effective_min = min(min_words, len(alignments))
    count = rng.randint(effective_min, len(alignments))
    chosen = rng.sample(alignments, count)

    # order gloss items by their source word's position in the source sentence
    def sort_key(alignment: dict) -> int:
        source_word, _ = gloss_pair(alignment, direction, word_field)
        return token_position(source_sentence, source_word, order_case_insensitive)

    chosen.sort(key=sort_key)
    items = [gloss_pair(alignment, direction, word_field) for alignment in chosen]

    prompt = flatten_prompt(
        rng.choice(PROMPT_TEMPLATES[direction]).format(
            source_sentence=source_sentence,
        )
    )
    response = render_response(items, full_translation, direction)
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
        variant combinations that could not be built (the verse has no trainable
        alignments).
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
        default=2,
        help=(
            "Target floor on words glossed per example, softened to a verse's "
            "trainable-alignment count when fewer are available (default: 2). "
            "The floor skews the gloss chain toward 2-3 anchors; verses with a "
            "single alignment still emit a one-word example."
        ),
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

    with open(input_path, encoding="utf-8") as f:
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
