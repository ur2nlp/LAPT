#!/usr/bin/env python3
"""
Expand finalized word-spotting JSONL into instruction-tuning format.

Each alignment in the input is projected into one or more instruction examples.
A "projection" is a query that fixes some fields of the underlying
(English sentence, Gothic sentence, English word, Gothic word) relation and asks
for another. Traversing the relation in several directions teaches the model the
*relation* rather than a memorized one-way function, which is the main lever
against the low-resource conditioning failure (see
`.claude/gothic/scaling_instruction_tuning.md`).

Supported projections:
    forward        English word + sentences -> Gothic word (original behaviour)
    reverse        Gothic word + sentences  -> English meaning
    cloze          English word + blanked Gothic sentence -> missing Gothic word
    discrimination Gothic word + two English candidates -> correct English meaning

Each projection has a few interchangeable prompt templates; one is chosen per
example with a seeded RNG, giving robustness to prompt phrasing without the
combinatorial blow-up of emitting every template.

Input format (from verify_word_spotting.py --finalize):
    {"english_sentence": "...", "gothic_sentence_roman": "...",
     "gothic_sentence_gothic": "...",
     "alignments": [{"target_word": "...", "gothic_word_roman": "...",
                      "gothic_word_gothic": "..."}, ...]}

Output format (instruction_jsonl, one line per example):
    {"prompt": "...\\nResponse:", "response": " answer"}

Usage:
    python tools/word-spotting/expand_to_instruction.py \
        --input data/gothic_word_spotting/train_verified_a.jsonl \
        --output data/gothic_word_spotting/train_word_spotting.jsonl

    # Original forward-only behaviour
    python tools/word-spotting/expand_to_instruction.py \
        --input finalized.jsonl --projections forward

    # Roman script only, random script per alignment
    python tools/word-spotting/expand_to_instruction.py \
        --input finalized.jsonl --script roman
"""

import argparse
import json
import random
import string
import sys
from pathlib import Path


# Marker substituted for the missing word in cloze prompts.
BLANK = "____"

# Characters stripped from sentence tokens when locating a word to blank.
PUNCTUATION = string.punctuation + "·"

SCRIPT_FIELDS = {
    "roman": ("gothic_sentence_roman", "gothic_word_roman"),
    "gothic": ("gothic_sentence_gothic", "gothic_word_gothic"),
}

# Prompt templates per projection. All four sentence/word placeholders are made
# available in the format context, so a template draws on whichever it needs.
TEMPLATES = {
    "forward": [
        (
            'In the following Gothic sentence, find the Gothic word for '
            '"{target_word}".\n'
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence}\n"
            "Response:"
        ),
        (
            'Which word in this Gothic sentence means "{target_word}"?\n'
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence}\n"
            "Response:"
        ),
        (
            'Read the Gothic sentence and identify the word that translates '
            '"{target_word}".\n'
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence}\n"
            "Response:"
        ),
    ],
    "reverse": [
        (
            'In the following Gothic sentence, what does the word '
            '"{gothic_word}" mean?\n'
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence}\n"
            "Response:"
        ),
        (
            'The Gothic sentence below contains the word "{gothic_word}". '
            "Give its English meaning.\n"
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence}\n"
            "Response:"
        ),
        (
            'Translate the Gothic word "{gothic_word}" as it is used in this '
            "sentence.\n"
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence}\n"
            "Response:"
        ),
    ],
    "cloze": [
        (
            "Fill in the blank in the Gothic sentence with the Gothic word for "
            '"{target_word}".\n'
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence_blanked}\n"
            "Response:"
        ),
        (
            "The Gothic sentence below is missing one word. Given that it "
            'means "{target_word}", supply the missing Gothic word.\n'
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence_blanked}\n"
            "Response:"
        ),
    ],
    "discrimination": [
        (
            'In the Gothic sentence below, does the word "{gothic_word}" mean '
            '"{option_a}" or "{option_b}"?\n'
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence}\n"
            "Response:"
        ),
        (
            'Choose the correct English meaning of the Gothic word '
            '"{gothic_word}" in this sentence: "{option_a}" or "{option_b}".\n'
            "English: {english_sentence}\n"
            "Gothic: {gothic_sentence}\n"
            "Response:"
        ),
    ],
}

PROJECTIONS = list(TEMPLATES.keys())


def blank_gothic_word(sentence: str, gothic_word: str) -> str | None:
    """Replace the first occurrence of a word in a Gothic sentence with a blank.

    Matches at the whitespace-token level, comparing each token with its leading
    and trailing punctuation stripped. Surrounding punctuation is preserved on
    the blanked token.

    Args:
        sentence: The Gothic sentence (Roman or Gothic script).
        gothic_word: The surface form to blank out.

    Returns:
        The sentence with the word replaced by ``BLANK``, or None if the word
        was not found as a token.
    """
    tokens = sentence.split()
    blanked_tokens = []
    replaced = False
    for token in tokens:
        if not replaced and token.strip(PUNCTUATION) == gothic_word:
            stripped_left = token.lstrip(PUNCTUATION)
            prefix = token[: len(token) - len(stripped_left)]
            stripped_right = token.rstrip(PUNCTUATION)
            suffix = token[len(stripped_right):]
            blanked_tokens.append(f"{prefix}{BLANK}{suffix}")
            replaced = True
        else:
            blanked_tokens.append(token)

    if not replaced:
        return None
    return " ".join(blanked_tokens)


def make_example(
    projection: str,
    alignment: dict,
    entry: dict,
    script_key: str,
    rng: random.Random,
    distractor_pool: list[str],
) -> dict | None:
    """Build a single instruction example for one projection of one alignment.

    Args:
        projection: One of the keys in TEMPLATES.
        alignment: A single alignment object from the entry.
        entry: The parent word-spotting entry.
        script_key: "roman" or "gothic".
        rng: Seeded RNG for template choice, discrimination option order, and
            distractor sampling.
        distractor_pool: Candidate English words for discrimination distractors,
            drawn from the same entry's other alignments when available and
            otherwise falling back to a global pool (assembled by the caller).

    Returns:
        A {"prompt": ..., "response": ...} dict, or None if the projection
        cannot be built for this alignment (currently only cloze, when the
        Gothic word is not found as a token in the sentence).
    """
    sentence_field, word_field = SCRIPT_FIELDS[script_key]
    gothic_sentence = entry[sentence_field]
    gothic_word = alignment[word_field]
    target_word = alignment["target_word"]

    context = {
        "target_word": target_word,
        "english_sentence": entry["english_sentence"],
        "gothic_sentence": gothic_sentence,
        "gothic_word": gothic_word,
    }

    if projection == "cloze":
        blanked = blank_gothic_word(gothic_sentence, gothic_word)
        if blanked is None:
            return None
        context["gothic_sentence_blanked"] = blanked
        response = gothic_word
    elif projection == "forward":
        response = gothic_word
    elif projection == "reverse":
        response = target_word
    elif projection == "discrimination":
        distractor = rng.choice(distractor_pool)
        options = [target_word, distractor]
        rng.shuffle(options)
        context["option_a"], context["option_b"] = options
        response = target_word
    else:
        raise ValueError(f"Unknown projection: {projection}")

    template = rng.choice(TEMPLATES[projection])
    prompt = template.format(**context)
    return {"prompt": prompt, "response": f" {response}"}


def expand_entry(
    entry: dict,
    projections: list[str],
    scripts: list[str],
    pick_one_script: bool,
    rng: random.Random,
    global_pool: list[str],
) -> tuple[list[dict], int]:
    """Expand one word-spotting entry into instruction examples.

    Args:
        entry: A finalized word-spotting JSONL entry.
        projections: Projection keys to emit.
        scripts: Script keys to consider ("roman", "gothic").
        pick_one_script: If True, choose one script per alignment at random
            (the --script random behaviour); otherwise emit every script.
        rng: Seeded RNG.
        global_pool: Global list of target words for distractor fallback.

    Returns:
        A (examples, skipped) tuple, where skipped counts projection/alignment
        combinations that could not be built (e.g. cloze with no token match).
    """
    examples: list[dict] = []
    skipped = 0
    target_words = [a["target_word"] for a in entry["alignments"]]

    for alignment in entry["alignments"]:
        if pick_one_script:
            selected_scripts = [rng.choice(scripts)]
        else:
            selected_scripts = scripts

        # Prefer a sibling target word as the discrimination distractor; fall
        # back to the global pool when the sentence has only one alignment.
        siblings = [
            word for word in target_words if word != alignment["target_word"]
        ]
        if siblings:
            distractor_pool = siblings
        else:
            distractor_pool = [
                word for word in global_pool if word != alignment["target_word"]
            ]

        for projection in projections:
            if projection == "discrimination" and not distractor_pool:
                skipped += 1
                continue
            for script_key in selected_scripts:
                example = make_example(
                    projection,
                    alignment,
                    entry,
                    script_key,
                    rng,
                    distractor_pool,
                )
                if example is None:
                    skipped += 1
                else:
                    examples.append(example)

    return examples, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Expand word-spotting JSONL into instruction-tuning format.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to finalized word-spotting JSONL.",
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
            "Which Gothic script(s) to produce examples for. "
            "'both' emits one example per script per alignment. "
            "'random' picks one script per alignment. (default: both)"
        ),
    )
    parser.add_argument(
        "--projections",
        default=",".join(PROJECTIONS),
        help=(
            "Comma-separated projections to emit, from "
            f"{{{', '.join(PROJECTIONS)}}}. "
            f"(default: all — {','.join(PROJECTIONS)})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for template/script/distractor sampling (default: 1).",
    )
    args = parser.parse_args()

    projections = [p.strip() for p in args.projections.split(",") if p.strip()]
    unknown = [p for p in projections if p not in TEMPLATES]
    if unknown:
        print(
            f"Error: unknown projection(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(PROJECTIONS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

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
    global_pool = [
        alignment["target_word"]
        for entry in entries
        for alignment in entry["alignments"]
    ]

    examples: list[dict] = []
    total_skipped = 0
    for entry in entries:
        entry_examples, skipped = expand_entry(
            entry,
            projections,
            scripts,
            pick_one_script,
            rng,
            global_pool,
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

    total_alignments = sum(len(e["alignments"]) for e in entries)
    skipped_note = f", {total_skipped} skipped" if total_skipped else ""
    print(
        f"Expanded {len(entries)} sentence pairs ({total_alignments} alignments) "
        f"into {len(examples)} instruction examples "
        f"(projections={','.join(projections)}, script={args.script}"
        f"{skipped_note})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
