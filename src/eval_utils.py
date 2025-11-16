"""
Evaluation utilities for measuring prediction diversity and detecting model collapse.

This module provides metrics for diagnosing pathological model behavior, particularly
degenerate generation where models collapse to repeatedly predicting a small set of
high-frequency tokens.
"""

import numpy as np
import torch
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
