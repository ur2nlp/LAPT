#!/usr/bin/env python3
"""Rewrite word-aligned pairs into OOV-robustness (hedging) translation examples.

For each canonical alignments entry, a ``verified_correct`` aligned Gothic word
is replaced with a generated non-word, and a got->eng translation target is
produced that (a) flags the unrecognized word and (b) translates the rest,
blanking the aligned English span. This teaches the model to hedge on OOV input
instead of hallucinating a fluent verse. See
`.claude/gothic/oov_robustness_augmentation.md`.

The non-word is produced by the stem-model pipeline (`gothic.oov_augmentation.
stems`): the replaced word's own prefix/suffix are stripped and grafted back onto
a freshly generated, junction-validated stem, so the non-word shares the word's
morphology but not its stem. The generator is built from the prepared monolingual
Gothic corpus, independent of the alignments file.

Response format (CoT fallback, always): the response glosses the verse's *other*
trainable words, flags the non-word as unrecognized, and concludes with the full
translation with the non-word's aligned English span blanked, e.g.

    "waurd" means "word", but I don't recognize "<nonword>", so the sentence
    means: "The _____ sows the word."

The gloss chain is the same "fall back to what you *do* know" reasoning that
`expand_to_cot` produces for plain translation; a verse with no other trainable
alignment degrades gracefully to a bare hedge. The response is emitted under
**both** prompt styles:
  * the plain translate prompt (drawn from the *same* distribution as the base
    translation task via ``prepare_gothic_data.build_instruction_prompt``), so
    the model hedges under a bare "Translate: X" prompt -- the failure demo's
    prompt -- rather than only when explicitly asked to reason;
  * the CoT prompt (``expand_to_cot.PROMPT_TEMPLATES``), matching CoT translation.

Morphology is not labeled in the response (portability / MVP regime): the hedge
names the whole non-word, never its stem or affix.

Input format: canonical {train,test}_alignments.jsonl (see expand_to_instruction).
Output format (one line per example):
    {"prompt": "... Response:", "response": " ..."}

Usage:
    python -m gothic.oov_augmentation.augment \
        --input data/gothic_word_spotting/train_alignments.jsonl \
        --output data/gothic_word_spotting/train_oov_hedge.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from gothic.data.prepare_gothic_data import build_instruction_prompt
from gothic.instruction_format import flatten_prompt
from gothic.oov_augmentation.stems import StemModel, build_stem_model, generate_nonword
from gothic.orthography import transliterate_latin_to_gothic
from gothic.word_spotting.canonical import trainable_alignments
from gothic.word_spotting.expand_to_cot import (
    CONCLUSION_TEMPLATES,
    SCRIPT_FIELDS,
    gloss_pair,
    render_gloss_items,
    token_position,
)
from gothic.word_spotting.expand_to_cot import (
    PROMPT_TEMPLATES as COT_PROMPT_TEMPLATES,
)

BLANK = "_____"

# Direction is fixed: we corrupt a Gothic source word, so hedging only makes
# sense translating Gothic -> English.
DIRECTION = "got2eng"

# Ways to flag the non-word as unrecognized. {word} is the non-word as it
# appears in the (active-script) sentence. Kept whole-word and morphology-silent
# (no stem/affix claim) per the portability regime. Written to read naturally
# both after "but " (mid-response) and capitalized at the start of a clause.
HEDGE_ITEM_STYLES = [
    'I don\'t recognize the word "{word}"',
    'I\'m not familiar with the word "{word}"',
    'the word "{word}" is unfamiliar to me',
    'I don\'t know the word "{word}"',
]


def blank_english_span(sentence: str, target_word: str) -> str:
    """Replace the aligned English word with a blank, case-insensitively.

    Args:
        sentence: The gold English sentence.
        target_word: The aligned English word to blank.

    Returns:
        The sentence with the first whole-word match of ``target_word`` blanked,
        or the sentence unchanged if no match was found.
    """
    pattern = re.compile(rf"\b{re.escape(target_word)}\b", flags=re.IGNORECASE)
    return pattern.sub(BLANK, sentence, count=1)


def capitalize_first(text: str) -> str:
    """Uppercase the first character of ``text`` (leaving the rest untouched)."""
    if not text:
        return text
    return text[0].upper() + text[1:]


def render_hedge_response(
    gloss_items: list[tuple[str, str]],
    hedge_clause: str,
    blanked_translation: str,
) -> str:
    """Render the CoT-fallback hedge response.

    The known words are glossed (reusing ``expand_to_cot.render_gloss_items``),
    the non-word is flagged with ``hedge_clause``, and the conclusion states the
    full translation with the non-word's English span blanked. Mirrors
    ``expand_to_cot.render_response`` but routes the corrupted word to a hedge
    instead of a gloss.

    Like ``expand_to_cot.render_response`` (v2.2.0), the response side is a
    single canonical form and consumes no RNG; hedge variety lives in the
    caller's choice of ``hedge_clause`` and in the input prompts.

    Args:
        gloss_items: Ordered (gothic_word, english_word) pairs for the *known*
            words to gloss. May be empty (bare hedge).
        hedge_clause: The rendered "I don't recognize ..." clause.
        blanked_translation: The gold English sentence with the aligned span
            blanked.

    Returns:
        The response string (without the leading space added by the caller).
    """
    conclusion = CONCLUSION_TEMPLATES[DIRECTION].format(full=blanked_translation)

    if not gloss_items:
        # No other trainable word to lean on: bare hedge + conclusion.
        return f"{capitalize_first(hedge_clause)}, {conclusion}"

    # "a means x, b means y, but <hedge>, <conclusion>". The hedge is the final
    # element of the chain, so the glosses take a plain comma join rather than
    # the "a, b, and c" join expand_to_cot uses when a gloss ends the chain.
    gloss_clause = ", ".join(render_gloss_items(gloss_items, DIRECTION))
    return f"{gloss_clause}, but {hedge_clause}, {conclusion}"


def build_prompt(sentence: str, prompt_style: str, rng: random.Random) -> str:
    """Build the prompt (ending in ' Response:') for the given style.

    Args:
        sentence: The corrupted source sentence (active script).
        prompt_style: "plain" (base-translation distribution) or "cot".
        rng: Seeded RNG.

    Returns:
        The prompt string ending with 'Response:'.
    """
    if prompt_style == "plain":
        # Draw from the exact base-translation prompt distribution (templates +
        # quote coin flip) so a bare "Translate: X" prompt sometimes translates
        # and sometimes hedges, rather than the hedge keying on a novel phrasing.
        return build_instruction_prompt("to_english", sentence, rng)

    template = rng.choice(COT_PROMPT_TEMPLATES[DIRECTION])
    return flatten_prompt(template.format(source_sentence=sentence))


def make_hedge_example(
    entry: dict,
    target: dict,
    alignments: list[dict],
    stem_model: StemModel,
    rng: random.Random,
    script_key: str,
    prompt_style: str,
    min_gloss: int,
) -> dict | None:
    """Build one hedging example corrupting ``target`` in ``entry``.

    Args:
        entry: A canonical alignments entry.
        target: The trainable alignment whose Gothic word is replaced.
        alignments: All of the entry's trainable alignments (for the gloss chain).
        stem_model: A stem model from ``build_stem_model``.
        rng: Seeded RNG for generation/gloss/template selection.
        script_key: "roman" or "gothic".
        prompt_style: "plain" or "cot".
        min_gloss: Floor on the number of known words glossed (softened to the
            number available), i.e. how much fallback reasoning to include.

    Returns:
        An ``{"prompt", "response"}`` dict, or None if generation/substitution
        failed.
    """
    sentence_field, word_field = SCRIPT_FIELDS[script_key]
    source_sentence = entry[sentence_field]
    target_surface = target[word_field]
    if target_surface not in source_sentence:
        return None

    source_roman = target["gothic_word_roman"].lower()
    nonword_roman = generate_nonword(stem_model, source_roman)
    if nonword_roman is None:
        return None
    if script_key == "gothic":
        nonword_surface = transliterate_latin_to_gothic(nonword_roman)
    else:
        nonword_surface = nonword_roman

    corrupted_sentence = source_sentence.replace(target_surface, nonword_surface, 1)
    blanked = blank_english_span(entry["english_sentence"], target["target_word"])
    if BLANK not in blanked:
        return None

    # gloss a sample of the *other* trainable words as the fallback reasoning.
    others = [alignment for alignment in alignments if alignment is not target]
    if others:
        effective_min = min(min_gloss, len(others))
        count = rng.randint(effective_min, len(others))
        chosen = rng.sample(others, count)
    else:
        chosen = []

    # order the glossed words by their position in the corrupted sentence.
    def sort_key(alignment: dict) -> int:
        gothic_word, _ = gloss_pair(alignment, DIRECTION, word_field)
        return token_position(corrupted_sentence, gothic_word, case_insensitive=False)

    chosen.sort(key=sort_key)
    gloss_items = [gloss_pair(alignment, DIRECTION, word_field) for alignment in chosen]

    hedge_clause = rng.choice(HEDGE_ITEM_STYLES).format(word=nonword_surface)
    response = render_hedge_response(gloss_items, hedge_clause, blanked)
    prompt = build_prompt(corrupted_sentence, prompt_style, rng)
    return {"prompt": prompt, "response": f" {response}"}


def expand_entry(
    entry: dict,
    stem_model: StemModel,
    rng: random.Random,
    scripts: list[str],
    prompt_styles: list[str],
    max_per_verse: int | None,
    min_gloss: int,
) -> list[dict]:
    """Expand one verse into hedging examples (one per corrupted word).

    Args:
        entry: A canonical alignments entry.
        stem_model: A stem model from ``build_stem_model``.
        rng: Seeded RNG.
        scripts: Script keys to emit ("roman", "gothic").
        prompt_styles: Prompt styles to emit ("plain", "cot").
        max_per_verse: Cap on how many distinct words to corrupt per verse
            (a random subset); None corrupts every trainable word.
        min_gloss: Floor on glossed known words per example.

    Returns:
        A list of ``{"prompt", "response"}`` examples.
    """
    alignments = trainable_alignments(entry)
    if not alignments:
        return []

    if max_per_verse is None or max_per_verse >= len(alignments):
        targets = alignments
    else:
        targets = rng.sample(alignments, max_per_verse)

    examples: list[dict] = []
    for target in targets:
        for script_key in scripts:
            for prompt_style in prompt_styles:
                example = make_hedge_example(
                    entry,
                    target,
                    alignments,
                    stem_model,
                    rng,
                    script_key,
                    prompt_style,
                    min_gloss,
                )
                if example is not None:
                    examples.append(example)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--script",
        choices=["roman", "gothic", "both"],
        default="both",
        help="Which Gothic script(s) to corrupt in (default: both).",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["plain", "cot", "both"],
        default="both",
        help=(
            "Prompt distribution(s) to emit hedges under: 'plain' (base "
            "translation prompts), 'cot' (chain-of-thought prompts), or 'both' "
            "(default)."
        ),
    )
    parser.add_argument(
        "--max-per-verse",
        type=int,
        default=None,
        help=(
            "Cap on distinct words corrupted per verse (a random subset). "
            "Default: corrupt every trainable word."
        ),
    )
    parser.add_argument(
        "--min-gloss",
        type=int,
        default=1,
        help=(
            "Floor on known words glossed as fallback reasoning per example, "
            "softened to the number available (default: 1)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.max_per_verse is not None and args.max_per_verse < 1:
        print("Error: --max-per-verse must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.min_gloss < 0:
        print("Error: --min-gloss must be >= 0.", file=sys.stderr)
        sys.exit(1)

    scripts = ["roman", "gothic"] if args.script == "both" else [args.script]
    prompt_styles = (
        ["plain", "cot"] if args.prompt_style == "both" else [args.prompt_style]
    )

    entries = [json.loads(line) for line in args.input.read_text().splitlines() if line]

    stem_model = build_stem_model(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    rng = random.Random(args.seed)
    written = 0
    with args.output.open("w") as handle:
        for entry in entries:
            for example in expand_entry(
                entry,
                stem_model,
                rng,
                scripts,
                prompt_styles,
                args.max_per_verse,
                args.min_gloss,
            ):
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1

    print(
        f"Wrote {written} hedging examples to {args.output} "
        f"(script={args.script}, prompt_style={args.prompt_style})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
