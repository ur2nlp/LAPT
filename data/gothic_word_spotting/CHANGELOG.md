# Changelog

Stage-based versioning; see [README.md](README.md#versioning).

## v0.4-diversified-unverified — 2026-05-20

Initial documented state. Established the canonical
`{train,test}_alignments.jsonl` schema with stable per-sentence and
per-alignment IDs, status taxonomy, and provenance fields.

### Added
- `train_alignments.jsonl` / `test_alignments.jsonl`: canonical files with
  one record per sentence pair (LLM-original-ordered), each carrying all
  alignments ever proposed for that sentence with `source`, `status`, and
  (where applicable) `original` / `replaced_focal_pair` fields.
- Folder reorganization into `raw_llm/`, `verification/`,
  `verification/archive/`, `diversification/` subfolders.
- This CHANGELOG and the dataset README.
- Diversification pipeline:
  - `gothic/word_spotting/prompt_diversify.txt`
  - `gothic/word_spotting/prepare_batches.py --mode diversify` (extends
    the original-mode script)
  - `gothic/word_spotting/merge_diversification.py`
- `gothic/word_spotting/assign_alignment_ids.py` for retroactive
  ID/status assignment.
- `gothic/word_spotting/alignment_frequencies.py` for distribution
  diagnostics.

### Changed
- Diversification reduced the worst Gothic↔English pair offenders by
  ~70% (e.g., `qaþ↔said` from 45 to 14 occurrences across train+test) and
  added 604 distinct new pair types in train + 141 in test, flattening
  the alignment frequency distribution.

## v0.3-verified-b — earlier

Full first-pass manual review of all LLM-proposed alignments. Reviewer
marked each TSV row `okay`, `unchecked`, or `delete`; some `okay` rows
had inline english_word edits, sometimes including parenthetical notes
on the rejected/unverified rows.

## v0.2-verified-a — archived

Partial earlier review pass, superseded by v0.3 but kept in
`verification/archive/` because earlier model-training experiments used
this version. Not for new use.

## v0.1-llm-raw — earlier

Initial LLM proposals from Claude (Sonnet 4.6) via the Anthropic Batch
API. Prompt: `gothic/word_spotting/prompt.txt`. Stored as
`raw_llm/{train,test}_annotations.jsonl`.
