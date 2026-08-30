"""
Gradient correction for continued training of vocabulary-pruned models.

After the vocabulary is pruned from the base set to a kept subset S, the
cross-entropy loss is computed over a smaller softmax:

    L'(z) = log Z_S - z_t,    Z_S = sum_{j in S} exp(z_j)

The resulting gradient w.r.t. logits is distorted by a factor of Z/Z_S
compared to the unpruned gradient (larger by Z/Z_S >= 1 because the pruned
softmax over-concentrates mass on the kept tokens):

    dL'/dz_i = (Z/Z_S) * dL/dz_i

Multiplying the pruned gradients by Z_S/Z therefore recovers the unpruned
training signal. Z_S is read live from each forward pass of `lm_head`;
Z is approximated by a fixed scalar estimate E[log Z] computed before
pruning on reference text (see tools/estimate_partition_function.py).

As continued training redistributes probability mass onto S, Z_S naturally
grows toward Z, the ratio approaches 1, and the correction self-eliminates.

Usage:
    install_gradient_correction(model, mean_log_z=12.34)
    trainer.add_callback(GradientCorrectionLogCallback(model))
"""

import json
import os
import sys

import torch
import torch.nn as nn
from transformers import TrainerCallback

Z_ESTIMATE_FILENAME = "z_estimate.json"


def load_z_estimate(tokenizer_path: str) -> float | None:
    """
    Look for z_estimate.json in the tokenizer directory and return its
    mean_log_z field. Returns None if the file is not found.
    """
    path = os.path.join(tokenizer_path, Z_ESTIMATE_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return float(data['mean_log_z'])


def install_gradient_correction(
    model: nn.Module,
    mean_log_z: float,
) -> None:
    """
    Install a forward hook on `model.lm_head` that rescales the gradient
    flowing back through the logits by Z_S / exp(E[log Z]) per position.

    The per-position ratio is computed from the live pruned logits, so it
    adapts automatically as the model redistributes probability mass onto
    the kept vocabulary.

    Side effects:
        - Attaches a forward hook to model.lm_head.
        - Sets model._gradient_correction_log_z = mean_log_z (introspection).
        - Initializes running accumulators on the model, read and reset by
          GradientCorrectionLogCallback:
              model._gradient_correction_ratio_sum (float)
              model._gradient_correction_ratio_count (int)

    Args:
        model: A CausalLM with a `lm_head` attribute (checked).
        mean_log_z: E[log Z] estimated from the unpruned base model on
            reference text.
    """
    if not hasattr(model, 'lm_head'):
        raise AttributeError(
            "install_gradient_correction expects model to have an `lm_head` module"
        )

    model._gradient_correction_log_z = mean_log_z
    model._gradient_correction_ratio_sum = 0.0
    model._gradient_correction_ratio_count = 0

    def forward_hook(module, inputs, output):
        # output: logits with shape (batch, seq_len, vocab_kept)
        # If grads aren't being tracked (eval mode, no_grad context), skip.
        if not torch.is_tensor(output) or not output.requires_grad:
            return

        # Per-position log Z_S from the live pruned logits, in fp32 for stability.
        with torch.no_grad():
            log_z_s = torch.logsumexp(output.float(), dim=-1)
            # Ratio is exp(log Z_S - E[log Z]); stays roughly ~1 near convergence.
            ratio = torch.exp(log_z_s - mean_log_z)
            model._gradient_correction_ratio_sum += ratio.mean().item()
            model._gradient_correction_ratio_count += 1

        # Register a tensor-level hook on the logits that multiplies the
        # incoming gradient by the per-position ratio. Tensor grad hooks
        # fire once per backward, so there's no leak across iterations.
        ratio_for_grad = ratio.unsqueeze(-1)  # (batch, seq_len, 1)

        def grad_hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * ratio_for_grad.to(grad.dtype)

        output.register_hook(grad_hook)

    model.lm_head.register_forward_hook(forward_hook)
    print(
        f"Gradient correction installed on lm_head (E[log Z] = {mean_log_z:.4f})",
        file=sys.stderr,
    )


class GradientCorrectionLogCallback(TrainerCallback):
    """
    Callback that surfaces the mean Z_S/exp(E[log Z]) gradient-correction ratio
    into the trainer log history on every logging step.

    The ratio is averaged across all forward passes since the last log event
    and reset afterward. Values far from 1 indicate the correction is doing
    significant work; values near 1 indicate the pruned model has redistributed
    most of its mass onto the kept vocabulary and the correction can be removed.

    Args:
        model: The model on which `install_gradient_correction` was called.
            Used only for its accumulator attributes.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def on_log(self, args, state, control, logs=None, **kwargs):
        count = getattr(self.model, '_gradient_correction_ratio_count', 0)
        if count == 0:
            return
        total = self.model._gradient_correction_ratio_sum
        mean_ratio = total / count
        # Reset for the next window.
        self.model._gradient_correction_ratio_sum = 0.0
        self.model._gradient_correction_ratio_count = 0

        # Inject into both the live logs dict (so it shows in console/wandb)
        # and the last log_history entry (so it lands in trainer_state.json).
        if logs is not None:
            logs['grad_correction_ratio'] = mean_ratio
        if state.log_history:
            state.log_history[-1]['grad_correction_ratio'] = mean_ratio
