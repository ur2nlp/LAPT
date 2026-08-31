"""
Custom Trainer variants with alternative loss formulations.

Provides `FlooredPerExampleLossTrainer`, which computes the training loss as a
mean over examples rather than a mean over tokens. Each example's loss is the
sum of its response-token cross-entropies divided by `max(t_i, k)`, where `t_i`
is the number of non-masked (non-(-100)) label tokens in that example and `k`
is the `per_example_loss_floor`.

Motivation: with the HuggingFace default (mean over all non-masked tokens in
the batch), long examples dominate the gradient. In an instruction-tuning mix
with heterogeneous response lengths — long translation/transliteration
responses alongside short FLAN or word-spotting responses — short tasks get
washed out. Per-example normalization gives each example equal weight; a floor
`k > 1` caps how much a very short example can punch above its length so that
a single noisy token cannot dominate an update.

Only the training loss is overridden. Evaluation delegates to the parent
implementation so that `eval_loss` (and therefore BPC) remains comparable
across runs trained with different `loss_type` settings.
"""

import torch
import torch.nn as nn
from transformers import Trainer


def floored_per_example_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    per_example_loss_floor: int,
) -> torch.Tensor:
    """Compute the floored per-example causal-LM loss.

    For each example in the batch, the loss is the sum of response-token
    cross-entropies divided by `max(t_i, k)`, where `t_i` is the number of
    non-masked label tokens and `k = per_example_loss_floor`. The batch loss is
    the mean of per-example losses over examples with at least one valid token.

    Args:
        logits: (batch, seq, vocab) pre-shift model logits.
        labels: (batch, seq) labels with -100 marking positions to ignore.
        per_example_loss_floor: `k`, must be >= 1.

    Returns:
        Scalar loss tensor.
    """
    if per_example_loss_floor < 1:
        raise ValueError(
            f"per_example_loss_floor must be >= 1, got {per_example_loss_floor}"
        )

    # causal-LM shift: predict token t+1 given token t
    # (batch, seq - 1, vocab)
    shift_logits = logits[..., :-1, :].contiguous()
    # (batch, seq - 1)
    shift_labels = labels[..., 1:].contiguous()

    batch_size, seq_len_minus_1 = shift_labels.shape

    # per-token cross-entropy in fp32 for numerical stability under bf16/fp16
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    # (batch * (seq - 1),)
    per_token_loss_flat = loss_fct(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    # (batch, seq - 1)
    per_token_loss = per_token_loss_flat.view(batch_size, seq_len_minus_1)

    # mask of valid (non-ignored) positions
    # (batch, seq - 1)
    valid_mask = (shift_labels != -100).to(per_token_loss.dtype)

    # (batch,)
    per_example_loss_sum = (per_token_loss * valid_mask).sum(dim=-1)
    # (batch,)
    per_example_token_count = valid_mask.sum(dim=-1)

    # floor the denominator to cap very-short-example gradient weight
    denominator = torch.clamp(
        per_example_token_count,
        min=float(per_example_loss_floor),
    )

    # exclude examples with zero valid tokens from both numerator and count
    valid_example_mask = (per_example_token_count > 0).to(per_token_loss.dtype)
    num_valid_examples = valid_example_mask.sum().clamp(min=1.0)

    # (batch,)
    per_example_loss = per_example_loss_sum / denominator
    return (per_example_loss * valid_example_mask).sum() / num_valid_examples


class FlooredPerExampleLossTrainer(Trainer):
    """Trainer with floored per-example training loss.

    Args:
        per_example_loss_floor: Minimum denominator `k` applied to each example's
            token-count divisor. Must be >= 1. With `k = 1` the loss is a pure
            per-example mean; with `k > 1`, examples shorter than `k` have their
            gradient weight capped.
    """

    def __init__(
        self,
        *args,
        per_example_loss_floor: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if per_example_loss_floor < 1:
            raise ValueError(
                f"per_example_loss_floor must be >= 1, got {per_example_loss_floor}"
            )
        self.per_example_loss_floor = per_example_loss_floor

        # Our compute_loss returns a mean over the micro-batch, not a
        # num_items_in_batch-normalized sum. Force HF's training_step to divide
        # by gradient_accumulation_steps so the accumulated gradient and logged
        # loss match the true mean over the effective batch. Without this, both
        # are inflated by a factor of gradient_accumulation_steps.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # fall back to HF default during eval so eval_loss / BPC stay comparable
        # across runs trained with different loss formulations
        if not model.training:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # pop labels so the model does not compute its own (token-mean) loss
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        loss = floored_per_example_causal_lm_loss(
            logits=outputs.logits,
            labels=labels,
            per_example_loss_floor=self.per_example_loss_floor,
        )

        return (loss, outputs) if return_outputs else loss
