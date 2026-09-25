# `lapt_core.spm`

Build HuggingFace tokenizer backends from a trained SentencePiece model.

Policy-free: where the *special* tokens land in the id space is a separate
decision, covered in
[Special tokens in vocabulary replacement](../guides/special_token_policies.md).
Every policy builds its backend through the functions here.

Requires the `tokenizers` extra (`pip install "lapt-core[tokenizers]"`).

::: lapt_core.spm
