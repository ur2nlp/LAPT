"""
Evaluation utilities for measuring prediction diversity and detecting model collapse.

This module provides metrics for diagnosing pathological model behavior, particularly
degenerate generation where models collapse to repeatedly predicting a small set of
high-frequency tokens. Also provides bits-per-character (BPC) computation for
tokenizer-agnostic evaluation.
"""

import math
import sys

import numpy as np
import torch
from transformers import TrainerCallback
from typing import Dict


def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess logits before accumulation to reduce memory usage.

    This function is called by HuggingFace Trainer after each eval batch,
    before accumulating predictions. By converting full logits
    (batch, seq, vocab_size) to argmax token IDs (batch, seq), we reduce
    memory usage by ~256x for models with large vocabularies like XGLM.

    Without this, Trainer accumulates full logits from all eval batches in GPU
    memory, causing OOM errors with large vocabularies (256K tokens).

    Args:
        logits: (batch_size, seq_len, vocab_size) prediction logits from model
        labels: (batch_size, seq_len) ground truth labels (unused, but required by interface)

    Returns:
        (batch_size, seq_len) tensor of predicted token IDs
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_ttr_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute type-token ratio (TTR) to measure prediction diversity.

    This function measures whether the model is collapsing to a small set of
    majority-class predictions, which can happen after embedding reinitialization
    (e.g., with FOCUS). TTR (unique tokens / total tokens) captures vocabulary
    diversity within each sequence.

    Args:
        eval_pred: EvalPrediction object with:
            - predictions: (batch_size, seq_len) token IDs (already argmaxed by preprocess_logits_for_metrics)
            - label_ids: (batch_size, seq_len) with -100 for padding

    Returns:
        Dictionary with:
            - ttr-seq: Average unique tokens / seq_len per sequence

    Note:
        This function expects predictions to already be argmaxed token IDs, not logits.
        Use preprocess_logits_for_metrics with Trainer to ensure this.
    """
    predictions, labels = eval_pred

    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        pred_tokens = predictions.cpu().numpy()
    else:
        pred_tokens = predictions

    # Predictions should already be argmaxed by preprocess_logits_for_metrics
    # Shape: (batch_size, seq_len)
    if len(pred_tokens.shape) == 3:
        raise ValueError(
            "compute_ttr_metrics received 3D predictions (logits), but expects "
            "2D token IDs. Make sure to use preprocess_logits_for_metrics with Trainer."
        )

    # Create mask for non-padding positions (labels == -100 indicates padding)
    if labels is not None:
        mask = labels != -100
    else:
        # No padding information, use all positions
        mask = np.ones_like(pred_tokens, dtype=bool)

    # Compute TTR (type-token ratio) within each sequence, then average
    ttr_per_seq = []
    for i in range(len(pred_tokens)):
        seq_tokens = pred_tokens[i][mask[i]]
        seq_len = len(seq_tokens)
        if seq_len > 0:
            seq_unique = len(np.unique(seq_tokens))
            ttr_per_seq.append(seq_unique / seq_len)

    avg_ttr = np.mean(ttr_per_seq) if ttr_per_seq else 0.0

    return {
        'ttr-seq': avg_ttr,
    }


def compute_chars_per_token(
    eval_dataset,
    tokenizer,
) -> dict[str, float]:
    """Compute the average characters per loss-contributing token for each eval split.

    This ratio enables converting per-token eval loss (nats) to bits-per-character (BPC),
    which is comparable across tokenizers.

    For CLM, the model predicts token[i+1] from token[i], so the number of loss-contributing
    tokens per example is (non-masked labels) - 1.

    Args:
        eval_dataset: Either a single Dataset or a dict of Datasets (keyed by split name).
        tokenizer: The tokenizer used to encode the dataset, needed for decoding back to text.

    Returns:
        Mapping from metric prefix to chars_per_token ratio.
        For a single dataset: {"eval": 4.2}
        For a dict: {"eval_got": 4.2, "eval_ang": 3.8, ...}
    """
    if isinstance(eval_dataset, dict):
        splits = eval_dataset
    else:
        splits = {"__single__": eval_dataset}

    ratios = {}
    for split_name, split_ds in splits.items():
        total_chars = 0
        total_tokens = 0

        input_ids_list = split_ds["input_ids"]
        has_labels = "labels" in split_ds.column_names if hasattr(split_ds, "column_names") else False
        labels_list = split_ds["labels"] if has_labels else None

        for i in range(len(split_ds)):
            input_ids = input_ids_list[i]
            decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
            total_chars += len(decoded)

            if labels_list is not None:
                labels = labels_list[i]
                # Count non-masked labels, minus 1 for CLM shift
                non_masked = sum(1 for label in labels if label != -100)
            else:
                non_masked = len(input_ids)

            # CLM shift: model predicts next token, so loss tokens = non_masked - 1
            loss_tokens = max(non_masked - 1, 0)
            total_tokens += loss_tokens

        if total_tokens == 0:
            ratio = 0.0
        else:
            ratio = total_chars / total_tokens

        if split_name == "__single__":
            ratios["eval"] = ratio
        else:
            ratios[f"eval_{split_name}"] = ratio

    return ratios


class BPCCallback(TrainerCallback):
    """Trainer callback that injects bits-per-character (BPC) into eval metrics.

    BPC = eval_loss / (chars_per_token * ln(2))

    This normalizes per-token cross-entropy loss by character count, enabling fair
    comparison across different tokenizers.

    Args:
        chars_per_token_ratios: Mapping from metric prefix (e.g. "eval", "eval_got")
            to the chars_per_token ratio for that split.
    """

    def __init__(self, chars_per_token_ratios: dict[str, float]):
        self.chars_per_token_ratios = chars_per_token_ratios

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        for prefix, chars_per_token in self.chars_per_token_ratios.items():
            loss_key = f"{prefix}_loss"
            bpc_key = f"{prefix}_bpc"
            if loss_key in metrics and chars_per_token > 0:
                metrics[bpc_key] = metrics[loss_key] / (chars_per_token * math.log(2))
