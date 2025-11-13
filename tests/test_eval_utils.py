"""
Tests for evaluation utilities, particularly distinct-n metrics.
"""

import numpy as np
import pytest
from collections import namedtuple

from eval_utils import compute_distinctness_metrics


# Create a simple EvalPrediction-like object for testing
EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids'])


def test_distinctness_perfect_diversity():
    """Test case where every prediction is unique (perfect diversity)."""
    batch_size = 4
    seq_len = 10
    vocab_size = 100

    # Create predictions where argmax gives unique tokens: 0, 1, 2, 3, ...
    predictions = np.zeros((batch_size, seq_len, vocab_size))
    for i in range(batch_size):
        for j in range(seq_len):
            token_id = i * seq_len + j
            predictions[i, j, token_id] = 1.0

    # No padding
    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # With all unique tokens, both metrics should be 1.0
    assert metrics['distinct_1_batch'] == 1.0
    assert metrics['distinct_1_within_seq'] == 1.0
    assert metrics['num_eval_tokens'] == batch_size * seq_len


def test_distinctness_complete_collapse():
    """Test case where model predicts same token everywhere (complete collapse)."""
    batch_size = 4
    seq_len = 10
    vocab_size = 100

    # All predictions have highest probability for token 42
    predictions = np.zeros((batch_size, seq_len, vocab_size))
    predictions[:, :, 42] = 1.0

    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # Only one unique token, so both metrics should be very low
    # distinct_1_batch = 1 unique / 40 total = 0.025
    # distinct_1_within_seq = 1 unique / 10 per seq = 0.1
    assert metrics['distinct_1_batch'] == 1.0 / (batch_size * seq_len)
    assert metrics['distinct_1_within_seq'] == 1.0 / seq_len
    assert metrics['num_eval_tokens'] == batch_size * seq_len


def test_distinctness_repetitive_sequences():
    """Test case where each sequence repeats a few tokens (like 'true true text text')."""
    batch_size = 4
    seq_len = 8
    vocab_size = 100

    predictions = np.zeros((batch_size, seq_len, vocab_size))

    # Each sequence repeats 2 tokens (4 times each)
    # Seq 0: [10, 10, 10, 10, 20, 20, 20, 20]
    # Seq 1: [30, 30, 30, 30, 40, 40, 40, 40]
    # etc.
    for i in range(batch_size):
        token_a = i * 20 + 10
        token_b = i * 20 + 20
        predictions[i, :4, token_a] = 1.0
        predictions[i, 4:, token_b] = 1.0

    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # Each sequence has 2 unique tokens out of 8 = 0.25
    assert metrics['distinct_1_within_seq'] == 2.0 / seq_len

    # Across batch: 8 unique tokens (2 per sequence) out of 32 total = 0.25
    assert metrics['distinct_1_batch'] == (batch_size * 2) / (batch_size * seq_len)


def test_distinctness_with_padding():
    """Test that padding tokens are correctly masked out."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    predictions = np.zeros((batch_size, seq_len, vocab_size))

    # First sequence: tokens 0-7 valid, last 2 padded
    # Predict tokens [0, 1, 2, 3, 4, 5, 6, 7] in valid positions
    for j in range(8):
        predictions[0, j, j] = 1.0
    predictions[0, 8:, 99] = 1.0  # Padded positions (shouldn't count)

    # Second sequence: tokens 0-5 valid, last 4 padded
    for j in range(6):
        predictions[1, j, j + 10] = 1.0
    predictions[1, 6:, 99] = 1.0  # Padded positions

    # Labels with -100 indicating padding
    labels = np.ones((batch_size, seq_len), dtype=int)
    labels[0, 8:] = -100  # Last 2 positions of seq 0 are padding
    labels[1, 6:] = -100  # Last 4 positions of seq 1 are padding

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # Should only count non-padded tokens: 8 + 6 = 14 tokens
    assert metrics['num_eval_tokens'] == 14

    # All 14 non-padded tokens are unique
    assert metrics['distinct_1_batch'] == 1.0

    # Seq 0: 8 unique / 8 = 1.0
    # Seq 1: 6 unique / 6 = 1.0
    # Average: 1.0
    assert metrics['distinct_1_within_seq'] == 1.0


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
    assert metrics['distinct_1_batch'] == 0.5

    # Within seq: (3/5 + 2/5) / 2 = 0.5
    assert metrics['distinct_1_within_seq'] == 0.5


def test_distinctness_realistic_pathological():
    """
    Test realistic pathological case matching observed behavior.

    Simulates model collapsing to predicting a few high-frequency tokens
    like 'following', 'true', 'text' repeatedly.
    """
    batch_size = 8
    seq_len = 20
    vocab_size = 32768

    predictions = np.zeros((batch_size, seq_len, vocab_size))

    # Simulate collapse to 3 tokens: 100 ('following'), 200 ('true'), 300 ('text')
    # Distribute somewhat randomly but with heavy repetition
    collapse_tokens = [100, 200, 300]

    np.random.seed(42)
    for i in range(batch_size):
        for j in range(seq_len):
            # Heavily biased toward the 3 collapse tokens
            token = np.random.choice(collapse_tokens)
            predictions[i, j, token] = 1.0

    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_distinctness_metrics(eval_pred)

    # Should have very low distinctness (only 3 unique tokens out of 160)
    assert metrics['distinct_1_batch'] < 0.05
    assert metrics['distinct_1_within_seq'] < 0.2  # At most 3/20 = 0.15

    # Exact values depend on random distribution
    print(f"Pathological case metrics: {metrics}")
