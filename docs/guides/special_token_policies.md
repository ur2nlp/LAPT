# Special tokens in vocabulary replacement

Replacing a model's vocabulary is mostly a question about the *learned* pieces:
how many, trained on what, initialized how. The special tokens are the part that
quietly decides whether the adapted checkpoint still works.

They are not just four more entries. Other parts of a checkpoint point at them
**by id** — a `TemplateProcessing` post-processor, `generation_config.json`, a
chat template's stop condition, and, on some architectures, whole maps of ids.
Replacing the vocabulary moves those targets. What you do about that is a
choice, and there is more than one right answer.

## The question that decides it

> Does anything downstream depend on the special tokens' **absolute ids**, or on
> the **internal structure** of the special-token block?

Answer that for your base model and the policy follows.

## The three policies

| Policy | Absolute ids | Block structure | Use when |
|---|---|---|---|
| Mint at base ids | preserved | n/a | base ids fit inside the new vocabulary |
| Mint positionally | lost | n/a | they don't fit, and nothing downstream cares |
| Append above | lost | preserved | the block has internal arithmetic |

### Mint at base ids

SentencePiece is told to place each role at the id it already had —
`unk_id`/`bos_id`/`eos_id`/`pad_id`. For a base like XGLM, whose roles sit at
3/0/2/1, everything downstream that hardcodes an id keeps working, so the base
tokenizer's post-processor can be copied verbatim.

This is the best outcome when you can get it, and it is LAPT's default.

### Mint positionally

The base ids are unusable when a role has no id, when two roles share one, or
when an id falls outside the target vocabulary. The last case is the norm for a
large-vocabulary base: a model whose eos is 151643 cannot put it inside a 32k
vocabulary.

The roles are then assigned 0, 1, 2… in role order. Absolute ids change, so
anything that hardcoded them is now wrong — which is why LAPT checks
(`_base_special_token_ids_preserved`) and skips copying the post-processor
rather than copying a broken one.

For a base model whose specials are four role tokens, this costs nothing real.
There is no structure to lose.

### Append above the learned block

Some base models have a special-token block that is *structured*, and the
structure is load-bearing. Whisper is the clearest case: its added-token block
holds `<|startoftranscript|>`, a contiguous run of language tokens,
`<|transcribe|>`/`<|translate|>`, `<|notimestamps|>`, `<|endoftext|>`, and 1501
timestamp tokens — and its generation config addresses them through index
arithmetic and through `lang_to_id`/`task_to_id` maps, not just by name.

Minting those into SentencePiece scatters them through the learned vocabulary
and destroys the arithmetic. The alternative is to train only `<unk>`, then
re-attach the base block **after** the learned pieces, in its original order.
Every id in the block then shifts by the same constant, so relative offsets
survive.

Absolute ids are *not* preserved here — the point is that the block stays
internally consistent. Which means this policy is only half a mechanism:

!!! warning "Append needs a companion remap step"
    Every stored id that addressed the old vocabulary is now stale. A model-side
    pass has to rewrite `config` and `generation_config` — including any
    language/task maps — by looking each token back up **by name**. Without it
    you get a tokenizer that is internally correct and a checkpoint whose
    configs point at the wrong rows.

!!! note "Reference implementation"
    [`ur2nlp/ASR`](https://github.com/ur2nlp/ASR) applies this policy to Whisper
    in [`src/focus.py`](https://github.com/ur2nlp/ASR/blob/main/src/focus.py).
    `_append_base_special_tokens` re-attaches the block and verifies that it
    landed contiguously and in order; `remap_special_token_ids` is the companion
    pass that rewrites `config` and `generation_config` by name.

## What LAPT implements

The first two, chosen automatically by `_assign_special_token_ids` in
`lapt/tokenizer.py`. It tries to preserve base ids and falls back to
positional assignment, reporting on stderr when it does.

LAPT does **not** implement the third. The decoder-only bases it targets have no
structured special block, and the policy would be incomplete without the
model-side remap described above. If you are adapting LAPT to a base model whose
special block carries arithmetic, that is the gap to fill, and you need both
halves — the reference implementation above is a worked example.

## Do not collapse these into one policy

They look like three solutions to one problem, so the natural instinct is to
pick the best one and delete the others. There isn't a best one.

* Force *append* onto a model with small, fixed role ids and you lose absolute
  id preservation for nothing — the post-processor stops being copyable.
* Force *mint* onto a model with a structured block and you break its
  arithmetic.

Both failures are quiet. Neither raises at build time; they surface as a model
that underperforms or never stops generating. If you are refactoring this area,
treat the three as a fixed taxonomy rather than as drift to be unified.

## Checking your work

The ids in the special-tokens config are a **request** made to SentencePiece,
not a fact about the tokenizer you got back. They have to survive the conversion
to a HuggingFace backend and role resolution, which deliberately drops a role to
`None` when its piece is absent from the trained vocabulary.

`_validate_special_token_ids` asserts the post-condition after the build, so a
role that silently went missing fails loudly instead of turning into a model
that never emits a stop token. If you add a policy, extend that check to cover
it.
