"""Stateless helpers for turning corpus text into tokenized rows.

A peer of `lapt/sources/`, not a member of it. `sources` is the untokenized
layer -- it produces text -- and `sources/text_processing.py` holds helpers
that the source types themselves call. Nothing here is called by a source;
these belong to the stage that consumes one. Keeping them out of `sources`
also keeps `transformers` out of it: that package depends only on `datasets`
today, and it is the layer nearest to being shared with sibling projects.

Everything here is a pure function of its arguments. The artifact classes in
`dataset_utils` supply the caching, the config record, and the cache-or-build
decision; this module supplies the transformations they run.
"""

from datasets import DatasetDict, load_from_disk
from transformers import PreTrainedTokenizer


def tokenize_plaintext_with_labels(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> dict:
    """
    Tokenize plaintext examples and add labels for causal LM loss.

    Used for plaintext splits in mixed instruction/plaintext datasets, where the
    DataCollatorForInstructionTuning expects all examples to have 'labels'.
    For plaintext, labels = input_ids (loss on all tokens).

    Args:
        examples: Batch with 'text' field
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels' fields
    """
    tokenized = tokenizer(
        examples['text'], max_length=max_length, truncation=True
    )
    # For plaintext, labels = input_ids (standard causal LM loss on all tokens)
    tokenized['labels'] = [ids.copy() for ids in tokenized['input_ids']]
    return tokenized


def tokenize_instruction_examples(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> dict:
    """
    Tokenize instruction examples with label masking.

    For each example, tokenizes prompt and response separately, then concatenates.
    Creates labels where prompt tokens are masked (-100) and only response tokens
    contribute to the loss.

    Also handles mixed datasets where some examples have prompt/response (instruction)
    and others have only text (plaintext). Plaintext examples get labels = input_ids
    (standard causal LM loss on all tokens).

    Args:
        examples: Batch with 'prompt' and 'response' fields, optionally 'text'
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length (prompt + response combined)

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels' fields
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    # Get text column if it exists (for mixed datasets)
    texts = examples.get('text', [None] * len(examples['prompt']))

    for prompt, response, text in zip(examples['prompt'], examples['response'], texts):
        # Check if this is an instruction example or plaintext
        is_instruction = prompt is not None and response is not None

        if is_instruction:
            # Instruction example: tokenize prompt and response separately
            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False
            )

            response_tokens = tokenizer(
                response,
                add_special_tokens=False,
                truncation=False
            )

            # Append EOS so the model learns to terminate responses
            response_ids = response_tokens['input_ids'] + [tokenizer.eos_token_id]
            response_mask = response_tokens['attention_mask'] + [1]

            # Concatenate
            # TODO: fix linting issue here
            input_ids = prompt_tokens['input_ids'] + response_ids
            attention_mask = prompt_tokens['attention_mask'] + response_mask

            # Create labels: -100 for prompt (masked), actual tokens for response
            prompt_length = len(prompt_tokens['input_ids'])
            labels = [-100] * prompt_length + response_ids
        else:
            # Plaintext example: standard tokenization, labels = input_ids
            if text is None:
                raise ValueError(
                    "Example has neither valid prompt/response nor text. "
                    "Mixed datasets must have 'text' for plaintext examples."
                )

            tokens = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )

            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            # Standard causal LM: predict all tokens
            labels = list(input_ids)

        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'labels': all_labels
    }


# Plan / per-source tokenization for multinomial mixes.
# The training-time multinomial pipeline upsamples by repeating row indices
# rather than duplicating tokenized rows. The pieces below implement that:
#
#   <cache_dir>/<source_id>/untokenized/                      (text, mix-agnostic)
#   <cache_dir>/<source_id>/tokenized_<tok>_ml<L>_{labels,nolabels}/  (mix-agnostic)
#   <mix_dir>/train_plan.npz                                  (shuffled global indices)
#   <mix_dir>/dev/                                            (per-source tokenized dev)
#
# The training Dataset is built as
# `concatenate_datasets(per_source_tokenized).select(global_indices)`,
# which produces an Arrow indices-mapped view; no rows are duplicated.


def tokenized_source_dirname(
    tokenizer_id: str,
    max_length: int,
    add_labels: bool,
    variant_suffix: str = "",
) -> str:
    """
    Build the per-source tokenized cache directory name.

    The optional ``variant_suffix`` carries the untokenized variant (e.g.
    ``_sub_<hash>`` for a substituted source) so a substituted source tokenizes
    into a distinct cache rather than colliding with the raw one. An empty
    suffix reproduces the original ``tokenized_<id>_ml<L>_<labels>`` name, so
    pre-existing raw caches stay valid.
    """
    label_suffix = "labels" if add_labels else "nolabels"
    return f"tokenized{variant_suffix}_{tokenizer_id}_ml{max_length}_{label_suffix}"


def dev_splits_dirname(
    tokenizer_id: str,
    max_length: int,
    add_labels: bool,
) -> str:
    """
    Build the mix-level tokenized dev-splits cache directory name.

    The dev splits hold token ids, so the cache must be keyed by the same
    parameters as the per-source tokenized caches. The mix slug that names the
    parent directory deliberately excludes the tokenizer (sources and plans are
    shared across tokenizers), so without this suffix a dev cache written by one
    model's tokenizer would be silently reused by a model with a different
    tokenizer.
    """
    label_suffix = "labels" if add_labels else "nolabels"
    return f"dev_{tokenizer_id}_ml{max_length}_{label_suffix}"


def source_has_instruction_columns(untokenized_path: str) -> bool:
    """Return True if the source's untokenized 'train' split has prompt/response columns."""
    data = load_from_disk(untokenized_path)
    if isinstance(data, DatasetDict):
        split = data['train'] if 'train' in data else data[list(data.keys())[0]]
    else:
        split = data
    cols = split.column_names
    return 'prompt' in cols and 'response' in cols
