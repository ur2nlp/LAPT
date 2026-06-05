#!/usr/bin/env python3
"""
Expand the Wright glossary JSON into dictionary-lookup instruction examples.

Produces vocabulary lemma↔gloss pairs to patch the model's lexical-coverage
hallucinations (the bible corpus exposes almost no general vocabulary, so the
model invents forms). This is a coverage patch only — see
`.claude/gothic/dictionary_lookup.md` and `scaling_instruction_tuning.md`.

Two directions over each (Gothic lemma, English gloss) pair:
    forward   English meaning -> Gothic word    ("Gothic for 'apostle'?")
    reverse   Gothic word     -> English meaning ("What does 'apaustaulus' mean?")

Each direction has a few interchangeable prompt templates, one drawn per example
with a seeded RNG (phrasing robustness without combinatorial blow-up).

Form/gloss handling:
- Variant spellings (entry `variant_forms`) are all emitted: in forward as
  separate valid targets for the same gloss, in reverse as separate inputs. SFT
  loss is against a single target string, so "multiple acceptable answers" must
  be expressed as multiple rows, not a set in one row.
- Forward emits one row per (gloss × form): each gloss is a distinct valid
  English query for the word.
- Reverse picks the gloss per `--reverse-gloss-mode`: `sample` (one sampled
  gloss per row, default), `each` (one row per gloss — "good noise" that pairs
  with Gothic upsampling), or `join` (all glosses comma-joined into one row).
- Reconstructed (`*`) headwords are excluded by default (unattested forms);
  `--include-reconstructed` keeps them.
- The query term is quoted per a `--quote-prob` coin flip (default 0.5) rather
  than always quoted, so the model does not become dependent on quoting (cf. the
  inference-time quote-sensitivity mitigation in translation/transliteration).

Input:  data/gothic_dictionaries/wright_gothic_english.json
Output format (instruction_jsonl, one line per example):
    {"prompt": "... Response:", "response": " answer"}

Usage:
    python -m gothic.dictionary_lookup.expand_to_instruction \
        --input data/gothic_dictionaries/wright_gothic_english.json \
        --output data/gothic_instruct/dictionary_lookup.jsonl

    # Reverse only, one row per gloss, Roman script
    python -m gothic.dictionary_lookup.expand_to_instruction \
        --directions reverse --reverse-gloss-mode each --script roman
"""

import argparse
import json
import random
import sys
from pathlib import Path

from gothic.instruction_format import flatten_prompt
from gothic.orthography import transliterate_latin_to_gothic

SCRIPTS = ["roman", "gothic"]
DIRECTIONS = ["forward", "reverse"]
REVERSE_GLOSS_MODES = ["sample", "each", "join"]

# The language label disambiguates the answer's script: "Gothic" -> native
# Gothic script, "romanized Gothic" -> Roman. This matches the translation-data
# convention (unmarked Gothic = Gothic script) and keeps the label a modifier of
# the *language*, never a script you "translate into" (which would conflate with
# the transliteration task).
LANG_LABELS = {"roman": "romanized Gothic", "gothic": "Gothic"}

# Prompt templates. Forward fills {lang} and {gloss}; reverse fills {lang} and
# {word}. The query placeholder is bare (no hard-coded quotes) so quoting can be
# diversified per example by maybe_quote — see the quote-sensitivity rigidity
# mitigation in prepare_gothic_data.build_instruction_prompt. Templates end with
# a "\nResponse:" delimiter, but flatten_prompt collapses the newline to a space
# at assembly (the PTEx tokenizer has no newline piece); responses begin with a
# leading space.
FORWARD_TEMPLATES = [
    "What is the {lang} word for {gloss}?\nResponse:",
    "Give the {lang} word that means {gloss}.\nResponse:",
    "What is the {lang} for {gloss}?\nResponse:",
    "Translate {gloss} into {lang}.\nResponse:",
]
REVERSE_TEMPLATES = [
    "What does the {lang} word {word} mean?\nResponse:",
    "Give the English meaning of the {lang} word {word}.\nResponse:",
    "Translate the {lang} word {word} into English.\nResponse:",
    "What is the {lang} word {word} in English?\nResponse:",
]


def maybe_quote(text: str, rng: random.Random, quote_prob: float) -> str:
    """Wrap text in double quotes with probability ``quote_prob``.

    Diversifies quoting so the model does not become dependent on the query term
    always being quoted (cf. the inference-time quote-sensitivity mitigation in
    translation/transliteration). Skips wrapping if the text already contains a
    double quote, to avoid nesting.
    """
    if '"' in text:
        return text
    if rng.random() < quote_prob:
        return f'"{text}"'
    return text


def render_form(form: str, script: str) -> str:
    """Render a normalized (Roman) Gothic form in the requested script."""
    if script == "gothic":
        return transliterate_latin_to_gothic(form)
    return form


def forms_for_entry(entry: dict, include_reconstructed: bool) -> list[str]:
    """Return the acceptable Gothic surface forms for an entry (deduplicated).

    The primary `headword_normalized` plus any `variant_forms`. Reconstructed
    entries return no forms unless ``include_reconstructed`` is set.
    """
    if entry.get("reconstructed") and not include_reconstructed:
        return []
    candidates = [entry["headword_normalized"]] + entry.get("variant_forms", [])
    forms: list[str] = []
    seen: set[str] = set()
    for form in candidates:
        if form and form not in seen:
            seen.add(form)
            forms.append(form)
    return forms


def reverse_gloss_sets(
    glosses: list[str],
    mode: str,
    rng: random.Random,
) -> list[str]:
    """Select the response gloss(es) for one reverse example per the mode."""
    if mode == "each":
        return list(glosses)
    if mode == "join":
        return [", ".join(glosses)]
    # sample
    return [rng.choice(glosses)]


def expand_entry(
    entry: dict,
    directions: list[str],
    scripts: list[str],
    pick_one_script: bool,
    reverse_gloss_mode: str,
    variants: int,
    include_reconstructed: bool,
    quote_prob: float,
    rng: random.Random,
) -> list[dict]:
    """Expand one glossary entry into instruction examples."""
    glosses = entry.get("glosses") or []
    if not glosses:
        return []
    forms = forms_for_entry(entry, include_reconstructed)
    if not forms:
        return []

    if pick_one_script:
        selected_scripts = [rng.choice(scripts)]
    else:
        selected_scripts = scripts

    examples: list[dict] = []
    for script in selected_scripts:
        rendered_forms = [render_form(form, script) for form in forms]
        lang = LANG_LABELS[script]

        if "forward" in directions:
            for gloss in glosses:
                for word in rendered_forms:
                    for _ in range(variants):
                        template = rng.choice(FORWARD_TEMPLATES)
                        prompt = flatten_prompt(template.format(
                            lang=lang,
                            gloss=maybe_quote(gloss, rng, quote_prob),
                        ))
                        examples.append({
                            "prompt": prompt,
                            "response": f" {word}",
                        })

        if "reverse" in directions:
            for word in rendered_forms:
                for _ in range(variants):
                    for gloss in reverse_gloss_sets(glosses, reverse_gloss_mode, rng):
                        template = rng.choice(REVERSE_TEMPLATES)
                        prompt = flatten_prompt(template.format(
                            lang=lang,
                            word=maybe_quote(word, rng, quote_prob),
                        ))
                        examples.append({
                            "prompt": prompt,
                            "response": f" {gloss}",
                        })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Expand the Wright glossary JSON into dictionary-lookup "
        "instruction examples.",
    )
    parser.add_argument(
        "--input",
        default="data/gothic_dictionaries/wright_gothic_english.json",
        help="Path to the parsed Wright glossary JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for instruction JSONL. Defaults to stdout.",
    )
    parser.add_argument(
        "--directions",
        default=",".join(DIRECTIONS),
        help=(
            "Comma-separated directions to emit, from "
            f"{{{', '.join(DIRECTIONS)}}}. (default: both)"
        ),
    )
    parser.add_argument(
        "--script",
        choices=["roman", "gothic", "both", "random"],
        default="both",
        help=(
            "Which Gothic script(s) to produce. 'both' emits one example per "
            "script; 'random' picks one script per entry. (default: both)"
        ),
    )
    parser.add_argument(
        "--reverse-gloss-mode",
        choices=REVERSE_GLOSS_MODES,
        default="sample",
        help=(
            "How a reverse example chooses its English response from a "
            "multi-gloss entry: 'sample' one gloss (default), 'each' one row "
            "per gloss, 'join' all glosses comma-joined."
        ),
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Independently-sampled examples per base row (default: 1).",
    )
    parser.add_argument(
        "--include-reconstructed",
        action="store_true",
        help="Include reconstructed (*) headwords (excluded by default).",
    )
    parser.add_argument(
        "--quote-prob",
        type=float,
        default=0.5,
        help=(
            "Probability of wrapping the query term in double quotes, for "
            "anti-rigidity (1.0 = always quote, 0.0 = never). (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for template/script/gloss sampling (default: 1).",
    )
    args = parser.parse_args()

    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    unknown = [d for d in directions if d not in DIRECTIONS]
    if unknown:
        print(
            f"Error: unknown direction(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(DIRECTIONS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.variants < 1:
        print("Error: --variants must be >= 1", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.quote_prob <= 1.0:
        print("Error: --quote-prob must be in [0.0, 1.0]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)

    if args.script == "random":
        scripts = SCRIPTS
        pick_one_script = True
    elif args.script == "both":
        scripts = SCRIPTS
        pick_one_script = False
    else:
        scripts = [args.script]
        pick_one_script = False

    rng = random.Random(args.seed)
    examples: list[dict] = []
    for entry in entries:
        examples.extend(expand_entry(
            entry,
            directions,
            scripts,
            pick_one_script,
            args.reverse_gloss_mode,
            args.variants,
            args.include_reconstructed,
            args.quote_prob,
            rng,
        ))

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

    print(
        f"Expanded {len(entries)} glossary entries into {len(examples)} "
        f"instruction examples (directions={','.join(directions)}, "
        f"script={args.script}, reverse-gloss-mode={args.reverse_gloss_mode}, "
        f"variants={args.variants})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
