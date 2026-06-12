# `gothic/` — Gothic project pipeline

Project-specific code for the Gothic pilot, built on the language-agnostic **LAPT**
framework in `src/`. Everything here is about turning raw Old-Germanic corpora and
the Gothic Bible into the two-stage training data the framework consumes:

- **Stage 1 (CPT):** natural text — English C4 + historical Germanic (mainly Old
  Norse) + monolingual Gothic.
- **Stage 2 (instruction tuning):** English chat (No Robots) + Gothic tasks —
  translation, transliteration, word-spotting/alignment, chain-of-thought
  translation, dictionary lookup.

The framework code stays generic; the Gothic-specific corpus cleaning, parallel-data
prep, orthography, and task generation live here. The training entry points
themselves are framework configs (`configs/`), not scripts in this package.

## Layout

```
gothic/
├── data/                    corpus cleaning + unified data prep
│   ├── prepare_gothic_data.py   THE prep entry point: parallel verses →
│   │                            monolingual / translation / transliteration
│   │                            (plaintext or instruction JSONL)
│   ├── clean_sagas.py           Old Norse saga corpus cleaning
│   ├── clean_icepahc.py         IcePaHC (Old Norse) cleaning
│   ├── clean_beowulf.py         Old English (dropped from the mix; kept for record)
│   └── filter_flan.py           FLAN filtering (legacy English-IT source)
│
├── word_spotting/           word-spotting alignment task  ── see its own README.md
│                            (canonical alignments file + propose / iterative
│                             verify / diversify / expand around it)
│
├── dictionary_lookup/       Wright glossary → lemma↔gloss IT task
│   ├── parse_wright_glossary.py   public-domain glossary HTML/text → JSON
│   └── expand_to_instruction.py   glossary → instruction examples
│
├── orthography.py           Latin↔Gothic-script transliteration; romanization
│                            normalization; clean_gothic_artifacts (editorial-mark
│                            cleanup, shared with word_spotting)
└── instruction_format.py    flatten_prompt — single-line ` Response:` prompt
                             stopgap (PTEx tokenizer has no newline piece)
```

## Submodules

- **`data/`** — corpus-specific cleaners (one per source) plus
  `prepare_gothic_data.py`, the single unified prep script that emits monolingual,
  translation, and transliteration data in either plaintext or instruction-JSONL
  form, across Roman / Gothic scripts and both directions. It feeds the framework's
  dataset loaders directly *and* produces the parallel input that the word-spotting
  pipeline annotates.
- **`word_spotting/`** — the largest subpackage: the multi-stage pipeline that
  produces the canonical alignment data and expands it into the alignment and
  chain-of-thought instruction datasets. It has its own
  [`README.md`](word_spotting/README.md) — start there for anything alignment-related.
- **`dictionary_lookup/`** — parses Wright's public-domain Gothic glossary and
  expands lemma↔gloss pairs into a dictionary-lookup IT task (both directions and
  scripts).

## Shared modules

- **`orthography.py`** — bidirectional Latin↔Gothic-script transliteration,
  romanization normalization (macrons / accents / the ƕ digraph → corpus surface
  forms), and `clean_gothic_artifacts` (unwrap restoration parens, strip
  Arabic-numeral glosses, join enclitic tildes). Usable as a library or CLI.
- **`instruction_format.py`** — `flatten_prompt`, which collapses newlines to
  spaces so prompts tokenize cleanly under the newline-less PTEx tokenizer. Every
  instruction-data generator calls it at prompt-assembly time. This is a stopgap;
  the proper fix is a newline-aware tokenizer.

## Environment

Run everything with the project env: `$CONDA_ENVS/lapt/bin/python` (see the
repo-root environment notes). Scripts are runnable as modules, e.g.
`python -m gothic.word_spotting.expand_to_instruction …`.
