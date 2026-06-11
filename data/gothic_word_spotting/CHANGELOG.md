# Changelog

Stage-based versioning; see [README.md](README.md#versioning).

## v0.5-diversified-verified — 2026-06-11

Resolved the `unverified` subset and cleaned editorial artifacts, graduating the
canonical files to a single sound trainable set. Trainable
(`verified_correct` + `kept_edited`) is now **4140 train / 1023 test**.

### Changed
- **Re-reviewed the `unverified` subset** via the `review_unverified.py`
  round-trip (`verification/{train,test}_v05_review.tsv`):
  - train: 630 → `verified_correct`, 19 → `kept_edited`, 18 → `rejected`
  - test: 164 → `verified_correct`, 5 → `kept_edited`, 9 → `rejected`
- **Cleaned Gothic-side editorial artifacts** (`clean_alignments.py` /
  `orthography.clean_gothic_artifacts`) across both sentence and surface fields:
  unwrapped reconstruction parentheses (`wa(i)ros` → `wairos`,
  `(𐍃𐌿)𐌽𐌹𐍅𐌴` → `𐍃𐌿𐌽𐌹𐍅𐌴`), removed Arabic-numeral glosses (`(2222)`), joined
  enclitic tildes (`qaþuþ~þan` → `qaþuþþan`), trimmed trailing fragment dashes.
  76 train / 20 test sentences affected; every Gothic surface was re-validated to
  occur in its sentence.
- **De-duplicated diversification-pass duplicates**: collapsed identical
  within-sentence alignments that `assign_alignment_ids.py` had given distinct
  `_dN` ids (90 train / 18 test alignments dropped, keeping the best-verified
  member).
- **English-target mismatch review**
  (`verification/{train,test}_english_mismatch_v05.tsv`): flagged trainable
  alignments whose English `target_word` does not occur verbatim as a token in
  the translation, then reviewed by hand. Policy: the listed English word must
  appear (or mostly appear) in the English sentence. Rejected genuine semantic
  mismatches where the gloss conflicts with the translation wording (e.g.
  "messenger" vs the verse's "angel", or `dagam` glossed "days" where the verse
  reads "years"); 25 train / 8 test → `rejected`. Kept morphological near-misses
  where the content word is still present and only the surface form or
  surrounding phrasing differs (e.g. `þiujo` glossed "maid" in a verse reading
  "one of the maids").

### Added
- `gothic/word_spotting/clean_alignments.py` — dedupe + Gothic-artifact cleanup
  + English-target mismatch export (in `review_unverified`'s apply schema).
- `review_unverified.py --statuses` (review any status, not just `unverified`)
  and `--no-coverage-check` (apply a content-defined subset TSV).

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
