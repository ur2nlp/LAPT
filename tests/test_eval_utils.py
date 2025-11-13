"""
Tests for evaluation utilities, particularly distinct-n metrics.
"""

import numpy as np
import pytest
import torch
from collections import namedtuple

from eval_utils import compute_distinctness_metrics, preprocess_logits_for_metrics


# Create a simple EvalPrediction-like object for testing
EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids'])


def test_preprocess_logits():
    """Test that preprocess_logits_for_metrics correctly converts logits to token IDs."""
    batch_size = 2
    seq_len = 5
    vocab_size = 100

    # Create logits with clear argmax
    logits = torch.zeros((batch_size, seq_len, vocab_size))
    logits[0, :, 10] = 1.0  # All positions predict token 10
    logits[1, :, 20] = 1.0  # All positions predict token 20

    labels = torch.ones((batch_size, seq_len), dtype=torch.long)

    result = preprocess_logits_for_metrics(logits, labels)

    # Should return token IDs, not logits
    assert result.shape == (batch_size, seq_len)
    assert result[0, 0].item() == 10
    assert result[1, 0].item() == 20


def test_preprocess_logits_tuple():
    """Test that preprocess handles tuple input (some models return tuples)."""
    batch_size = 2
    seq_len = 5
    vocab_size = 100

    logits = torch.zeros((batch_size, seq_len, vocab_size))
    logits[:, :, 42] = 1.0

    labels = torch.ones((batch_size, seq_len), dtype=torch.long)

    # Wrap in tuple like some models do
    result = preprocess_logits_for_metrics((logits, None), labels)

    assert result.shape == (batch_size, seq_len)
    assert (result == 42).all()


def test_distinctness_perfect_diversity():
    """Test case where every prediction is unique (perfect diversity)."""
    batch_size = 4
    seq_len = 10

    # Create token ID predictions: 0, 1, 2, 3, ... (all unique)
    predictions = np.arange(batch_size * seq_len).reshape(batch_size, seq_len)

    # No padding
    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # With all unique tokens, both metrics should be 1.0
    assert metrics['distinct-1-batch'] == 1.0
    assert metrics['distinct-1-seq'] == 1.0
    assert metrics['num_eval_tokens'] == batch_size * seq_len


def test_distinctness_complete_collapse():
    """Test case where model predicts same token everywhere (complete collapse)."""
    batch_size = 4
    seq_len = 10

    # All predictions are token 42
    predictions = np.full((batch_size, seq_len), 42)

    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # Only one unique token, so both metrics should be very low
    # distinct-1-batch = 1 unique / 40 total = 0.025
    # distinct-1-seq = 1 unique / 10 per seq = 0.1
    assert metrics['distinct-1-batch'] == 1.0 / (batch_size * seq_len)
    assert metrics['distinct-1-seq'] == 1.0 / seq_len
    assert metrics['num_eval_tokens'] == batch_size * seq_len


def test_distinctness_repetitive_sequences():
    """Test case where each sequence repeats a few tokens (like 'true true text text')."""
    batch_size = 4
    seq_len = 8

    # Each sequence repeats 2 tokens (4 times each)
    # Seq 0: [10, 10, 10, 10, 20, 20, 20, 20]
    # Seq 1: [30, 30, 30, 30, 40, 40, 40, 40]
    # etc.
    predictions = np.zeros((batch_size, seq_len), dtype=int)
    for i in range(batch_size):
        token_a = i * 20 + 10
        token_b = i * 20 + 20
        predictions[i, :4] = token_a
        predictions[i, 4:] = token_b

    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # Each sequence has 2 unique tokens out of 8 = 0.25
    assert metrics['distinct-1-seq'] == 2.0 / seq_len

    # Across batch: 8 unique tokens (2 per sequence) out of 32 total = 0.25
    assert metrics['distinct-1-batch'] == (batch_size * 2) / (batch_size * seq_len)


def test_distinctness_with_padding():
    """Test that padding tokens are correctly masked out."""
    batch_size = 2
    seq_len = 10

    predictions = np.zeros((batch_size, seq_len), dtype=int)

    # First sequence: tokens 0-7 valid, last 2 padded
    predictions[0, :8] = np.arange(8)
    predictions[0, 8:] = 99  # Padded positions (shouldn't count)

    # Second sequence: tokens 0-5 valid, last 4 padded
    predictions[1, :6] = np.arange(10, 16)
    predictions[1, 6:] = 99  # Padded positions

    # Labels with -100 indicating padding
    labels = np.ones((batch_size, seq_len), dtype=int)
    labels[0, 8:] = -100  # Last 2 positions of seq 0 are padding
    labels[1, 6:] = -100  # Last 4 positions of seq 1 are padding

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # Should only count non-padded tokens: 8 + 6 = 14 tokens
    assert metrics['num_eval_tokens'] == 14

    # All 14 non-padded tokens are unique
    assert metrics['distinct-1-batch'] == 1.0

    # Seq 0: 8 unique / 8 = 1.0
    # Seq 1: 6 unique / 6 = 1.0
    # Average: 1.0
    assert metrics['distinct-1-seq'] == 1.0


def test_distinctness_argmaxed_input():
    """Test that function works with pre-argmaxed predictions."""
    batch_size = 2
    seq_len = 5
    vocab_size = 100

    # Pass token IDs directly instead of logits
    predictions = np.array([
        [10, 10, 20, 20, 30],  # 3 unique tokens
        [40, 40, 40, 50, 50],  # 2 unique tokens
    ])

    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # Across batch: 5 unique tokens (10, 20, 30, 40, 50) / 10 total = 0.5
    assert metrics['distinct-1-batch'] == 0.5

    # Within seq: (3/5 + 2/5) / 2 = 0.5
    assert metrics['distinct-1-seq'] == 0.5


def test_distinctness_realistic_pathological():
    """
    Test realistic pathological case matching observed behavior.

    Simulates model collapsing to predicting a few high-frequency tokens
    like 'following', 'true', 'text' repeatedly.
    """
    batch_size = 8
    seq_len = 20

    # Simulate collapse to 3 tokens: 100 ('following'), 200 ('true'), 300 ('text')
    # Distribute somewhat randomly but with heavy repetition
    collapse_tokens = [100, 200, 300]

    np.random.seed(42)
    predictions = np.zeros((batch_size, seq_len), dtype=int)
    for i in range(batch_size):
        for j in range(seq_len):
            predictions[i, j] = np.random.choice(collapse_tokens)

    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # Should have very low distinctness (only 3 unique tokens out of 160)
    assert metrics['distinct-1-batch'] < 0.05
    assert metrics['distinct-1-seq'] < 0.2  # At most 3/20 = 0.15

    # Exact values depend on random distribution
    print(f"Pathological case metrics: {metrics}")
