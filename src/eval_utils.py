"""
Evaluation utilities for measuring prediction diversity and detecting model collapse.

This module provides metrics for diagnosing pathological model behavior, particularly
degenerate generation where models collapse to repeatedly predicting a small set of
high-frequency tokens.
"""

import numpy as np
from typing import Dict


def compute_distinctness_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute distinct-n metrics to measure prediction diversity.

    This function measures whether the model is collapsing to a small set of
    majority-class predictions, which can happen after embedding reinitialization
    (e.g., with FOCUS). Unlike per-prediction confidence metrics, distinct-n
    captures uniformity vs. diversity across predictions.

    Two complementary metrics are computed:
    1. distinct_1_batch: Fraction of unique tokens across entire batch
       - Detects: "Every sequence predicts the same tokens"
       - Healthy models: >0.3-0.5
       - Pathological: <0.1 (collapsed to majority class)

    2. distinct_1_within_seq: Average fraction of unique tokens per sequence
       - Detects: "Each sequence repeats itself"
       - Healthy models: >0.2-0.4 depending on domain
       - Pathological: <0.1 (degenerate repetition like "true true true...")

    Args:
        eval_pred: EvalPrediction object with:
            - predictions: (batch_size, seq_len, vocab_size) logits
            - label_ids: (batch_size, seq_len) with -100 for padding

    Returns:
        Dictionary with:
            - distinct_1_batch: Unique tokens / total tokens across batch
            - distinct_1_within_seq: Average unique tokens / seq_len per sequence
            - num_eval_tokens: Total number of non-padding tokens evaluated
    """
    predictions, labels = eval_pred

    # Take argmax to get predicted token IDs
    # predictions shape: (batch_size, seq_len, vocab_size) -> (batch_size, seq_len)
    if len(predictions.shape) == 3:
        pred_tokens = predictions.argmax(axis=-1)
    else:
        # Already argmaxed
        pred_tokens = predictions

    # Create mask for non-padding positions (labels == -100 indicates padding)
    if labels is not None:
        mask = labels != -100
    else:
        # No padding information, use all positions
        mask = np.ones_like(pred_tokens, dtype=bool)

    # Compute distinct-1 across entire batch
    valid_pred_tokens = pred_tokens[mask]
    total_tokens = len(valid_pred_tokens)
    unique_tokens = len(np.unique(valid_pred_tokens))
    distinct_1_batch = unique_tokens / total_tokens if total_tokens > 0 else 0.0

    # Compute distinct-1 within each sequence, then average
    distinct_1_per_seq = []
    for i in range(len(pred_tokens)):
        seq_tokens = pred_tokens[i][mask[i]]
        seq_len = len(seq_tokens)
        if seq_len > 0:
            seq_unique = len(np.unique(seq_tokens))
            distinct_1_per_seq.append(seq_unique / seq_len)

    distinct_1_within_seq = np.mean(distinct_1_per_seq) if distinct_1_per_seq else 0.0

    return {
        'distinct_1_batch': distinct_1_batch,
        'distinct_1_within_seq': distinct_1_within_seq,
        'num_eval_tokens': total_tokens,
    }
