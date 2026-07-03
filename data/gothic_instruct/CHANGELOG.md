# Gothic instruction-data changelog

Human-readable history of the instruction-tuning artifacts in this folder.
`MANIFEST.yaml` is the machine-readable companion (sha256, per-file provenance,
`used_by`). Newest first.

## 2026-07-03 — new artifact: oov-hedge-instruct 2.0.0

**New OOV-robustness (hedging) artifact**, teaching the model to flag an
unrecognized source word and translate the rest rather than hallucinate a
fluent-but-wrong verse. Built from the v0.5 canonical alignments
(`data/gothic_word_spotting/{train,test}_alignments.jsonl` at `cf21d31`), same
source facts as the alignment/CoT 2.0.0 family — so it **starts at 2.0.0** (not
1.0.0) to co-version with them, despite having no literal 1.0.0 predecessor.

Mechanism: for each `{verified_correct, kept_edited}` aligned word, replace it in
the Gothic sentence with a **generated non-word** (novel stem + the replaced
word's own grafted prefix/suffix, so it reads as unknown-but-well-formed Gothic,
not gibberish — see `.claude/gothic/oov_robustness_augmentation.md`), and emit a
got->eng target that names the non-word as unrecognized and blanks its aligned
English span. The response always **falls back to CoT**: it glosses the verse's
other trainable words before hedging. Emitted under **both** the plain
translation prompt (same distribution as `translation-instruct`, so a bare
"Translate: X" prompt sometimes translates and sometimes hedges) and the CoT
prompt.

- **`oov-hedge-instruct` 2.0.0** — `gothic.oov_augmentation.augment`,
  `script=both`, `prompt-style=both`, one example per trainable word, `min-gloss=1`,
  non-word T=0.6, seed 1 → **16,487 train / 4,088 test** examples. got2eng only.
  Not yet trained on / not yet wired into any train config.

Eval caveat carried into the manifest: once mixed in, **plain-translation bpc is
no longer cross-regime comparable** (the same prompt now branches to a hedge).
The discriminating signals are clean-holdout over-hedge rate + the `saijands` OOV
probe, **not** a synthetic non-word holdout (a spelling-cue heuristic would ace
the latter while failing real OOV).

## 2026-06-11 — v0.5 regeneration: alignment 2.0.0, CoT 2.0.0

**MAJOR bumps for both word-spotting-derived artifacts**, regenerated from the
re-reviewed v0.5 canonical alignments (`data/gothic_word_spotting/{train,test}_alignments.jsonl`
at `cf21d31`) — the underlying facts changed, so MAJOR by the settled semantics.
Both expanders now consume the status-bearing canonical files directly, taking the
trainable subset `{verified_correct, kept_edited}` (4,140 train / 1,023 test
alignments) via the shared `gothic.word_spotting.canonical` filter, rather than the
old v0.4 diversified file.

- **`alignment-instruct` 2.0.0** — default projections (forward/reverse/cloze/
  discrimination), `script=both`, seed 1 → **33,049 train / 8,172 test** examples
  (71 train / 12 test projection/alignment combos skipped, chiefly cloze with no
  token match). Resolves the `next:` note that was pending on this artifact.
- **`translation-cot-instruct` 2.0.0** — `script=both`, `directions=both`,
  `variants=1`, `min_words=2`, seed 1 → **5,312 train / 1,312 test** examples. The
  1.1.0 sampler is unchanged (min_words=2 soft floor); verified empirically on the
  v0.5 source that every verse with ≥2 trainable alignments glosses ≥2 words, and
  single-alignment verses still emit a one-word example. Count moved 5288→5312 train
  / 1312→1312 test purely from the changed trainable set.

Neither is trained on yet. Live train configs still point at the prior versions —
repoint them (and decide the holdout-eval comparability spine) before the next run.

## 2026-06-11 — Bump semantics defined; CoT 1.1.0

**Version semantics settled** (the deferred discussion from 2026-06-05), now
recorded at the top of `MANIFEST.yaml`. Keyed on what changes for a model trained
on the artifact: **MAJOR** = the underlying facts change (new/edited/removed
source content, source-file switch); **MINOR** = same facts, different rendering
(prompts, schema, sampling/templating, added/removed projections); **PATCH** =
corrective regeneration with no intended change to content or rendering (bugfix to
match documented behavior, whitespace normalization, dropping a corrupt record,
manifest-only fix).

**`translation-cot-instruct` 1.1.0 cut** (MINOR over 1.0.0). The CoT generator's
gloss-count sampler (`expand_to_cot.py`) was changed from uniform-from-1 to a soft
floor of `min_words=2` (capped at a verse's trainable-alignment count, so the 84
genuine single-alignment train verses still emit a one-word example). This skews
the gloss chain toward 2–3 anchors, targeting the observed single-link CoT bias
(`.claude/gothic/it_generation_observations.md`, obs. 4). Single-link share
dropped 36% → 6%. **Same source facts, seed, and example count** (5288 train /
1312 test) as 1.0.0 — only gloss density changed; verified the current source
reproduces 1.0.0 exactly under the old `min_words=1` sampler, so this is a pure
sampler diff. Live train configs (`gothic_instruct.yaml`, `gothic_instruct_1b.yaml`)
repointed to 1.1.0; the holdout-eval config is intentionally left on the 1.0.0
test set as a comparability spine (see `scaling_instruction_tuning.md` § Evaluation
— an eval-set change is the thing that rebases the metric). Not yet trained on.

## 2026-06-05 — Reliable versioning begins

Established `MANIFEST.yaml` + this changelog, started git-tracking
`data/gothic_instruct/` (incl. `archive/`), and cut the first reliably-versioned
releases. Context: the prompt/response boundary was switched from `\nResponse:`
to ` Response:` because the PTEx tokenizer has no newline piece and maps `\n` to
`<unk>` (see `.claude/TODO.md`). That separator change is what makes these new
releases distinct from the as-run bytes.

**1.0.0 cut for four artifacts:**
- `translation-instruct` 1.0.0, `transliteration-instruct` 1.0.0 — bytes
  unchanged this session (already space-form); these are exactly what the
  v74L-i8..i15 / v139-i18..i21 runs consumed.
- `translation-cot-instruct` 1.0.0, `dictionary-lookup-instruct` 1.0.0 —
  space-normalized form of the newline data the runs consumed; **content-identical**
  (verified `flatten(old)==new`). The newline-form is archived.

**`alignment-instruct` 1.0.0 cut** (space-form, content-identical to the as-run
newline archive — same pattern as cot/dict). Resolved a provenance scare from
earlier in the session: the v0.4 alignment files were thought un-reproducible
because regenerating from `train_alignments.jsonl` gave 4,718 alignments vs the
4,178 reflected in v0.4. The expander was in fact never run on
`train_alignments.jsonl` (a newer, status-bearing format); it consumes the
verified+diversified files `data/gothic_word_spotting/diversification/{train,test}_verified_b_diversified.jsonl`
(4,178 train / 1,031 test alignments). Expanding those with default args + seed 1
reproduces v0.4 exactly (33,327 train / 8,236 test, zero diffs after separator
normalization). 1.0.0 is the space-normalization of that; not itself trained on,
but directly comparable to what the v74L/v139 runs consumed.

**Archive populated** (newline-form / pre-pd predecessors, fetched from the
cluster where needed): authentic newline-form alignment v0.4 (the newline form the
runs ate; its space-form is the 1.0.0 release), legacy `_a` alignment, newline-form cot and
dictionary, and the pre-pd translation/transliteration.

**Live files renamed to the versioned scheme** `<descriptors>_<split>_v1.0.0.jsonl`
(version after the split suffix, matching the archive's `_v0.4` convention),
dropping `_pd_` per "keep all but pd". Added `aliases:` to each manifest entry
mapping the old config/registry-referenced filenames to their current versioned
file (or to the archived as-run form, for `alignment` `_v0.4`/`_a` and the
even-older `word-spotting_*_a`). Configs for finished runs are left untouched;
the aliases are the bridge. Validated: every config-referenced path resolves to a
current file or an alias.

### Still pending (next session)
- Define MAJOR/MINOR/PATCH bump semantics (deferred discussion).
- Optionally: emit a `.meta.json` sidecar from each generator so provenance is
  captured automatically on every run.
