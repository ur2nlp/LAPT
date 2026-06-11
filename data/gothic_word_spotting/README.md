# Gothic Word-Spotting Dataset

A parallel English–Gothic alignment dataset for instruction-tuning language
models on Gothic vocabulary recognition. For each pair of parallel sentences
(English ↔ Gothic, both in Roman transliteration and Gothic script), the
dataset provides one or more **alignments** — pairings of an English target
word/phrase to its Gothic surface form.

Source corpus: the Gothic Bible (Wulfila), aligned to English by verse, drawn
from multiple codices.

## Layout

```
data/gothic_word_spotting/
├── README.md
├── CHANGELOG.md
├── train_alignments.jsonl          ← canonical
├── test_alignments.jsonl           ← canonical
├── raw_llm/                        Stage 1: LLM first-pass annotation
│   ├── train_annotations.jsonl     ← extracted alignments (defines _o{N} order)
│   ├── test_annotations.jsonl
│   ├── train_batch_requests.jsonl  ← Anthropic Batch API inputs
│   └── test_batch_requests.jsonl
├── verification/                   Stage 2: manual review
│   ├── train_verified_b.tsv        ← reviewed spreadsheet export
│   ├── test_verified_b.tsv
│   ├── train_verified_b.jsonl      ← finalized post-review (okay rows only)
│   ├── test_verified_b.jsonl
│   └── archive/                    Superseded earlier review pass; kept
│       ├── train_verified_a.tsv    because prior experiments trained on it
│       ├── test_verified_a.tsv
│       ├── train_verified_a.jsonl
│       └── test_verified_a.jsonl
└── diversification/                Stage 3: focal-pair diversification
    ├── diversify_batch_requests.jsonl       ← Batch API inputs
    ├── diversify_batch_requests.manifest.json
    ├── diversify_annotations.jsonl          ← extracted new alignments
    ├── train_verified_b_diversified.jsonl   ← post-diversification state
    └── test_verified_b_diversified.jsonl
```

### Canonical

- `train_alignments.jsonl`, `test_alignments.jsonl` — **the source of
  truth**. Every LLM-proposed or LLM-diversified alignment that ever
  existed, tagged with provenance (`source`) and verification outcome
  (`status`). See schema below.

### Derived (training-ready)

- *Not yet generated.* The trainable subset is obtained by filtering
  `*_alignments.jsonl` to rows with `status ∈ {verified_correct,
  kept_edited}`. A future script will materialize `train.jsonl` /
  `test.jsonl` from this filter.

## Record schema

Each line of `{train,test}_alignments.jsonl` is one sentence pair:

```json
{
  "sentence_id": "train_0042",
  "english_sentence": "...",
  "gothic_sentence_roman": "...",
  "gothic_sentence_gothic": "...",
  "alignments": [
    {
      "alignment_id": "train_0042_o0",
      "source": "llm_original",
      "status": "verified_correct",
      "target_word": "...",
      "gothic_word_roman": "...",
      "gothic_word_gothic": "..."
    }
  ]
}
```

### sentence_id

`{split}_{NNNN}` where `NNNN` is the position in `*_annotations.jsonl` (the
LLM's first-pass output, which preserves the input order). Stable across
versions.

### alignment_id

`{sentence_id}_{prefix}{N}` where:

- `o` = LLM original (position in the LLM's per-sentence proposal list)
- `d` = added during diversification (position within that sentence's
  diversification response)
- `m` = manually added (reserved; not currently used)

`N` is zero-padded only when consistency matters; treat IDs as opaque
strings.

### source

| source | meaning |
|---|---|
| `llm_original` | proposed by Claude during the first annotation pass |
| `llm_diversify` | proposed by Claude during the diversification pass |

### status

| status | in trainable subset? | meaning |
|---|---|---|
| `verified_correct` | yes | reviewer marked `okay`; surface forms identical to LLM proposal |
| `kept_edited` | yes | reviewer marked `okay`; one of the surface forms was edited (the `original` field carries the pre-edit values) |
| `unverified` | optional | reviewer marked `unchecked`, *or* alignment never made it into the verification TSV, *or* added during diversification and not yet reviewed |
| `rejected` | no | reviewer marked `delete` |
| `replaced_in_diversification` | no | LLM-original alignment whose surface pair was an over-used "focal pair" during diversification, and which the retention pass did *not* keep |

`kept_edited` records carry an `original` field:

```json
"original": {
  "target_word": "strong",
  "gothic_word_roman": "swinþnoda",
  "gothic_word_gothic": "𐍃𐍅𐌹𐌽𐌸𐌽𐍉𐌳𐌰"
}
```

`llm_diversify` records carry a `replaced_focal_pair` field naming the
over-used pair they were generated to replace.

## Construction pipeline

### Stage 1 — LLM proposal

For each parallel sentence pair, Claude (Sonnet 4.6) was asked to propose
1–4 word-level alignments via the Anthropic Batch API. Prompt:
`gothic/word_spotting/prompt.txt`. Output: `*_annotations.jsonl`.

### Stage 2 — Manual verification

LLM proposals were validated against the Köbler Gothic–English dictionary
and tier-ranked by edit distance, then exported to TSV
(`verify_word_spotting.py` in verification mode). The reviewer marked each
row `okay`, `unchecked`, or `delete`, and occasionally edited surface
forms inline. The TSV was finalized back to JSONL
(`verify_word_spotting.py --finalize`) to produce `*_verified_b.jsonl`.

### Stage 3 — Diversification

A frequency analysis identified Gothic↔English surface pairs occurring
≥10 times in the verified data (the "focal pairs" — `qaþ↔said`,
`frauja↔lord`, etc.). For each focal pair, Claude was asked to propose
alternative alignments from the same sentences, with a global avoid-list
of all pairs occurring ≥5 times. Prompt:
`gothic/word_spotting/prompt_diversify.txt`. A retention pass kept ~30% of
each focal pair's occurrences (prioritizing records where the model
abstained or had the fewest other alignments). Result:
`*_verified_b_diversified.jsonl`.

### Stage 4 — Canonical ID assignment

`gothic/word_spotting/assign_alignment_ids.py` cross-references all
preceding artifacts to produce `*_alignments.jsonl` with stable IDs and
status provenance.

## Versioning

This dataset uses **stage-based version names** rather than semantic
versioning (which doesn't map well to data lifecycles).

- `v0.1-llm-raw` — raw LLM proposals, no review
- `v0.2-verified-a` — partial first-pass review (archived, not released)
- `v0.3-verified-b` — full first-pass review (earlier "verified" baseline)
- `v0.4-diversified-unverified` — post-diversification, diversification
  additions not yet re-reviewed
- `v0.5-diversified-verified` — diversification additions re-reviewed, Gothic
  editorial artifacts cleaned, within-sentence duplicates deduped, English-target
  mismatches rejected (current state)
- `v1.0` — first public release (planned)

Git tags should mark each version.

## Known quirks

### Codex duplication

The source Gothic Bible draws from multiple codices that sometimes preserve
identical Greek-source verses with identical or near-identical Gothic
text. Such verses appear as **distinct sentence_ids** in the dataset
(e.g., train_1265 and train_1266 both contain "afar leitil þan
atgaggandans…"). This is intentional — they're separate corpus
occurrences — but worth knowing for deduplication if you ever need it.

A side effect: during finalize from TSV→JSONL, `verify_word_spotting.py`
groups rows by `(english_sentence, gothic_sentence_roman)`. Codex-duplicate
sentences share that key, so their separate sets of alignment rows merge
into one `verified_b.jsonl` record. In the canonical alignments output,
both sentence_ids reference this merged record, which produces correct IDs
for both copies but caused some "orphan" warnings during ID assignment
that are not real data issues.

### Untraceable wholesale edits

If the reviewer edited *both* the gothic_surface and english_word of a
TSV row (rather than just the english_word with a parenthetical note),
the resulting verified-b alignment cannot be matched back to its
LLM-original counterpart. In the current dataset this happened exactly
once: `ni waihts ↔ there is nothing` in `train_0801`, which replaced the
LLM original `waihts ↔ nothing`. The LLM original is marked `unverified`
in the canonical output, and the edited form is dropped — at the cost of
one alignment.

For future review rounds, prefer in-place english_word edits (with a
parenthetical or appended note) over wholesale rewrites, or explicitly add
new rows for manual additions so they can be assigned `_m{N}` IDs.

## Statistics (v0.5-diversified-verified)

| | Train | Test |
|---|---|---|
| Sentences | 1,343 | 333 |
| Total alignments (all statuses) | 4,628 | 1,162 |
| `verified_correct` | 4,084 | 1,011 |
| `kept_edited` | 56 | 12 |
| `unverified` (llm_original) | 8 | 5 |
| `unverified` (llm_diversify) | 20 | 5 |
| `rejected` | 59 | 24 |
| `replaced_in_diversification` | 401 | 105 |
| **Trainable subset** (verified_correct + kept_edited) | **4,140** | **1,023** |

Total alignments dropped from v0.4 (4,718 / 1,180) because the v0.5 cleanup
removed 90 train / 18 test within-sentence diversification duplicates. A residue
of `unverified` rows remains (review-skipped, not yet adjudicated).
