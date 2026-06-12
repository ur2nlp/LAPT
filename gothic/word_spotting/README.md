# Gothic word-spotting pipeline

This package builds the **word-spotting alignment** data for the Gothic project:
for each parallel English↔Gothic sentence pair, one or more pairings of an English
target word to its Gothic surface form. Those alignments are the source for two
instruction-tuning datasets — the alignment/word-spotting tasks and the
chain-of-thought translation tasks.

For the **data** side — canonical schema, status taxonomy, per-version statistics,
known quirks — see
[`data/gothic_word_spotting/README.md`](../../data/gothic_word_spotting/README.md)
and its `CHANGELOG.md`. This file is about the **code**.

## The intended flow

The design centers on **one persistent reference**, the canonical
`{train,test}_alignments.jsonl`: one record per sentence pair, every alignment
carrying a stable `alignment_id`, a `source`, and a verification `status`. Once an
alignment has an id, every later operation patches it **in place** — the git diff
then shows exactly which alignments changed in each round.

Around that reference, the proper cycle is **propose → (diversify ⇄ verify),
iterating, then expand** — keeping the canonical file authoritative throughout:

```
        ┌──────────────────────────────────────────────┐
        │   canonical {train,test}_alignments.jsonl     │   ← the persistent reference
        │   (stable alignment_id + source + status)     │
        └──────────────────────────────────────────────┘
            ▲ propose            ▲ diversify           │ expand
            │                    │ (rebalance over-    │
   prepare_batches          prepare_batches           ▼
     --mode original          --mode diversify    canonical.py  (trainable filter)
   run_annotation           run_annotation        expand_to_instruction.py  → alignment data
                                                  expand_to_cot.py          → CoT translation data
            └──────── iterative verify ───────┘
              review_unverified.py  (export → human review → --apply)
              clean_alignments.py   (dedupe + artifact cleanup, in place)
              scored against the Köbler dictionary (parse_koebler_dictionary.py)
```

The key idea: **proposal and diversification are not one-shot stages with a single
review at the end. They interleave with verification, and each verification round
patches the same canonical file.** `review_unverified.py` is the workhorse — it can
re-target any status set (`--statuses`), so you can re-audit already-accepted rows,
not just freshly-proposed ones.

## What's realized vs. what's fossilized

The flow above is the *intended* model. Be aware which parts the code actually
implements that way today, and which are historical residue from before the
canonical format and the iterative round-trip existed:

- **Realized (go-forward path).** The review / clean / expand half already works
  the ideal way — `review_unverified.py`, `clean_alignments.py`, `canonical.py`,
  and the two expanders all operate **on the canonical file, in place, keyed by
  `alignment_id`.** This is where ongoing work happens.

- **Bootstrap / fossilized.** The canonical file was first assembled *retroactively*
  by `assign_alignment_ids.py`, which fused a set of pre-canonical flat
  intermediates (first-pass annotations, a `verify_word_spotting.py` review TSV, and
  a separately-merged diversification file) into the first
  `{train,test}_alignments.jsonl`. That linear bootstrap reflects the original
  order of events — first-pass annotation wasn't anticipating diversification or
  iterative review — not the intended cycle. Its artifacts are kept only as
  provenance:
  - `verify_word_spotting.py` — the first-pass review tool, which **rebuilds** a
    flat JSONL rather than patching in place. Superseded by `review_unverified.py`
    for any iterative work.
  - `assign_alignment_ids.py` — a one-time fusion. The canonical file now already
    exists; you don't re-run this in normal operation.
  - the intermediate flat files `*_verified_b.jsonl` /
    `*_verified_b_diversified.jsonl` — inputs to that fusion, not live targets.

- **Known gap.** There is currently **no tool that merges a fresh proposal or
  diversification round directly into the canonical file** (assigning new
  `_o`/`_d`/`_m` ids and statuses in place). `prepare_batches` + `run_annotation` +
  `merge_diversification` still emit flat JSONL, which historically fed
  `assign_alignment_ids.py`. Closing this gap — so a new annotation round patches
  the canonical file the way `review_unverified.py` already does — is what would
  make the proposal/diversification half match the intended flow.

## Script reference

### Live — the canonical-file path

- **`canonical.py`** — defines the trainable subset **once**:
  `TRAINABLE_STATUSES = {verified_correct, kept_edited}` and `trainable_alignments()`.
  Both expanders import it, so they can never train on different subsets.
  (No-`status` records — older files — are treated as all-trainable.)
- **`review_unverified.py`** — the in-place review round-trip. *Export* filters the
  canonical file to a status set (default `unverified`; `--statuses` re-targets any
  set), scores each alignment against Köbler, and writes a review TSV joined on
  `alignment_id`. *`--apply`* patches only the named rows: `reject`→`rejected`,
  `correct` unchanged→`verified_correct`, `correct` edited→`kept_edited` (pre-edit
  surfaces stashed under `original`). Untouched alignments stay byte-identical.
- **`clean_alignments.py`** — in-place maintenance over the canonical file: (1)
  de-duplicate within-sentence identical live alignments; (2) clean Gothic-side
  editorial artifacts (restoration parens, Arabic-numeral glosses, enclitic tildes,
  trailing dashes) via `gothic.orthography.clean_gothic_artifacts` — English fields
  untouched; (3) emit an English-target-mismatch review TSV in
  `review_unverified.py`'s apply schema, for hand review (reported, not auto-dropped).
- **`expand_to_instruction.py`** — projects each trainable alignment into
  instruction examples in four directions (`forward`, `reverse`, `cloze`,
  `discrimination`), with seeded prompt templates and Roman/Gothic script. →
  `alignment-instruct` dataset.
- **`expand_to_cot.py`** — verse-level: consumes all of a verse's trainable
  alignments to build one chain-of-thought example (gloss a few key words, then the
  full translation), both directions and scripts; a `min_words=2` soft floor skews
  the gloss chain toward 2–3 anchors. → CoT translation dataset.

Both expanders emit single-line prompts ending in ` Response:` (newlines collapsed
via `gothic.instruction_format.flatten_prompt`). Outputs land in
`data/gothic_instruct/`, version-tracked by that folder's `MANIFEST.yaml`.

### Proposal / diversification (LLM round)

- **`prepare_batches.py`** — builds Anthropic Batch API request JSONL.
  `--mode original` reads a prepared parallel-translation file and emits first-pass
  alignment requests; `--mode diversify` reads existing alignments, finds over-used
  "focal" surface pairs (`qaþ↔said`, …), and emits realignment requests plus a
  `.manifest.json`. Prompts: `prompt.txt` / `prompt_diversify.txt`.
- **`run_annotation.py`** — submits a request JSONL to Claude (`--mode sync` for
  small tests, `--mode batch` for the cheap Batch API), collects responses, and
  extracts the model's JSONL. Used for both passes.
- **`merge_diversification.py`** — merges diversification annotations back in,
  replacing focal-pair alignments under a retention proportion (default 0.3, so a
  pair's signal is reduced not erased). *(Currently writes a flat
  `*_diversified.jsonl`; see "Known gap" above.)*

### Scoring, diagnostics, and the bootstrap

- **`parse_koebler_dictionary.py`** — one-time: Köbler Gothic–English dictionary
  HTML → `koebler_gothic_english.json`, the edit-distance scoring reference used by
  both review tools.
- **`alignment_frequencies.py`** — diagnostic histograms of surface-form / pair
  frequencies; informs the diversification focal/avoid thresholds.
- **`verify_word_spotting.py`** *(fossilized)* — first-pass review: *verify mode*
  scores a flat annotations JSONL into a tiered TSV; *`--finalize`* drops
  `delete`/`unchecked` rows and **rebuilds** `*_verified_b.jsonl`. Superseded by
  `review_unverified.py` for in-place review.
- **`assign_alignment_ids.py`** *(fossilized, one-time)* — the retroactive fusion
  that produced the first canonical file from the bootstrap intermediates. Kept for
  provenance / reproducibility; not part of normal operation.

## Two review tools — when to use which

| | `verify_word_spotting.py` (bootstrap) | `review_unverified.py` (live) |
|---|---|---|
| Operates on | flat per-sentence JSONL | canonical `*_alignments.jsonl` |
| Join key | regroups by sentence text | `alignment_id` |
| On finalize/apply | **rebuilds** a new file | **patches in place**; untouched rows byte-identical |
| Role | original first-pass review | any iterative re-review (any status) |

## Regenerating the training data

After any change to `{train,test}_alignments.jsonl`, regenerate both downstream
datasets from the canonical file (seed 1):

```bash
PY=$CONDA_ENVS/lapt/bin/python
WS=data/gothic_word_spotting
OUT=data/gothic_instruct

# alignment-instruct  (repeat with test_alignments.jsonl)
$PY -m gothic.word_spotting.expand_to_instruction \
    --input $WS/train_alignments.jsonl \
    --output $OUT/alignment_both-scripts_train_vX.Y.Z.jsonl --seed 1

# translation-cot-instruct  (repeat with test_alignments.jsonl)
$PY -m gothic.word_spotting.expand_to_cot \
    --input $WS/train_alignments.jsonl \
    --output $OUT/translation-cot_both-scripts_both-directions_train_vX.Y.Z.jsonl --seed 1
```

Then record sha256 / counts in `data/gothic_instruct/MANIFEST.yaml`, add a
`CHANGELOG.md` entry, and repoint the dataset/holdout configs. Bump semantics
(MAJOR = source facts changed, MINOR = rendering only, PATCH = corrective) are at
the top of that MANIFEST.

## Related code outside this package

- `gothic/data/prepare_gothic_data.py` — produces the parallel-translation input to
  the proposal pass, and the plain translation/transliteration/monolingual IT data
  the CoT data is meant to be *mixed with*.
- `gothic/orthography.py` — transliteration + `clean_gothic_artifacts`.
- `gothic/instruction_format.py` — `flatten_prompt`, the single-line prompt stopgap.
