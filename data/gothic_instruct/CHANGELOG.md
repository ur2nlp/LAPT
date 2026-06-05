# Gothic instruction-data changelog

Human-readable history of the instruction-tuning artifacts in this folder.
`MANIFEST.yaml` is the machine-readable companion (sha256, per-file provenance,
`used_by`). Newest first.

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

**`alignment-instruct` 1.0.0 deliberately NOT cut.** It must be the
status-filtered expansion (`verified_correct` + `kept_edited`), which requires
adding a status filter to `gothic.word_spotting.expand_to_instruction` first
(the expander currently filters nothing — so a naive regen would train on
rejected/unverified alignments and would make the v0.4→v0.5 re-review a no-op).
Only the archived as-run input is registered for now.

**Archive populated** (newline-form / pre-pd predecessors, fetched from the
cluster where needed): authentic newline-form alignment v0.4 (the un-reproducible
4,178-alignment input the runs ate), legacy `_a` alignment, newline-form cot and
dictionary, and the pre-pd translation/transliteration.

### Still pending (next session)
- Cut `alignment-instruct` 1.0.0 after adding the status filter.
- Rename live files into the versioned scheme (`..._vMAJOR.MINOR.PATCH.jsonl`,
  dropping `_pd_`; keep other descriptors per "keep all but pd"), adding
  `aliases:` to the manifest for the old config-referenced paths.
- Define MAJOR/MINOR/PATCH bump semantics (deferred discussion).
- Optionally: emit a `.meta.json` sidecar from each generator so provenance is
  captured automatically on every run.
