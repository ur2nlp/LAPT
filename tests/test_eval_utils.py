"""
Tests for evaluation utilities: TTR metrics, BPC computation, and logit preprocessing.
"""

import math

import numpy as np
import pytest
import torch
from collections import namedtuple
from unittest.mock import MagicMock
from datasets import Dataset

from eval_utils import (
    compute_ttr_metrics, preprocess_logits_for_metrics,
    compute_chars_per_token, BPCCallback,
)


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
    metrics = compute_ttr_metrics(eval_pred)

    # With all unique tokens, TTR should be 1.0
    assert metrics['ttr-seq'] == 1.0


def test_distinctness_complete_collapse():
    """Test case where model predicts same token everywhere (complete collapse)."""
    batch_size = 4
    seq_len = 10

    # All predictions are token 42
    predictions = np.full((batch_size, seq_len), 42)

    labels = np.ones((batch_size, seq_len), dtype=int)

    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_ttr_metrics(eval_pred)

    # Only one unique token: 1 unique / 10 per seq = 0.1
    assert metrics['ttr-seq'] == 1.0 / seq_len


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
    metrics = compute_ttr_metrics(eval_pred)

    # Each sequence has 2 unique tokens out of 8 = 0.25
    assert metrics['ttr-seq'] == 2.0 / seq_len


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
    metrics = compute_ttr_metrics(eval_pred)

    # Seq 0: 8 unique / 8 = 1.0
    # Seq 1: 6 unique / 6 = 1.0
    # Average: 1.0
    assert metrics['ttr-seq'] == 1.0


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
    metrics = compute_ttr_metrics(eval_pred)

    # Within seq: (3/5 + 2/5) / 2 = 0.5
    assert metrics['ttr-seq'] == 0.5


# --- BPC tests ---

def _make_mock_tokenizer(token_to_text: dict[int, str]):
    """Create a mock tokenizer that decodes token IDs to predetermined strings."""
    tokenizer = MagicMock()

    def mock_decode(input_ids, skip_special_tokens=True):
        return "".join(token_to_text.get(tid, "") for tid in input_ids)

    tokenizer.decode = mock_decode
    return tokenizer


class TestComputeCharsPerToken:
    def test_single_dataset(self):
        """Test chars_per_token with a single eval dataset (non-dict)."""
        # Each token decodes to "ab" (2 chars)
        token_map = {1: "ab", 2: "cd", 3: "ef"}
        tokenizer = _make_mock_tokenizer(token_map)

        ds = Dataset.from_dict({
            "input_ids": [[1, 2, 3], [1, 2, 3]],
        })

        result = compute_chars_per_token(ds, tokenizer)

        # 2 examples, each: 6 chars, 2 loss tokens (3 tokens - 1 for CLM shift)
        # total_chars=12, total_tokens=4, ratio=3.0
        assert result == {"eval": 3.0}

    def test_dict_dataset_multi_split(self):
        """Test chars_per_token with a dict of eval datasets."""
        token_map = {1: "a", 2: "bb", 3: "ccc"}
        tokenizer = _make_mock_tokenizer(token_map)

        split_a = Dataset.from_dict({
            "input_ids": [[1, 2, 3]],
        })
        split_b = Dataset.from_dict({
            "input_ids": [[1, 1, 1, 1]],
        })

        result = compute_chars_per_token({"got": split_a, "ang": split_b}, tokenizer)

        # split_a: 1 example, chars="a"+"bb"+"ccc"=6 chars, tokens=3-1=2, ratio=3.0
        assert result["eval_got"] == pytest.approx(3.0)
        # split_b: 1 example, chars="a"*4=4 chars, tokens=4-1=3, ratio=4/3
        assert result["eval_ang"] == pytest.approx(4.0 / 3.0)

    def test_with_labels_masking(self):
        """Test that -100 labels are excluded from token count."""
        token_map = {1: "ab", 2: "cd", 3: "ef", 4: "gh"}
        tokenizer = _make_mock_tokenizer(token_map)

        ds = Dataset.from_dict({
            "input_ids": [[1, 2, 3, 4]],
            "labels": [[-100, -100, 3, 4]],
        })

        result = compute_chars_per_token(ds, tokenizer)

        # chars = "ab"+"cd"+"ef"+"gh" = 8 chars (decode uses all input_ids)
        # non-masked labels = 2 (tokens 3 and 4), loss tokens = 2-1 = 1
        # ratio = 8/1 = 8.0
        assert result == {"eval": 8.0}

    def test_empty_dataset(self):
        """Test with an empty dataset."""
        tokenizer = _make_mock_tokenizer({})
        ds = Dataset.from_dict({"input_ids": []})
        result = compute_chars_per_token(ds, tokenizer)
        assert result == {"eval": 0.0}


class TestBPCCallback:
    def test_injects_bpc(self):
        """Test that BPC is correctly computed and injected into metrics."""
        callback = BPCCallback({"eval": 4.0})
        metrics = {"eval_loss": 2.0}

        callback.on_evaluate(args=None, state=None, control=None, metrics=metrics)

        expected_bpc = 2.0 / (4.0 * math.log(2))
        assert metrics["eval_bpc"] == pytest.approx(expected_bpc)

    def test_multi_split(self):
        """Test BPC injection for multiple eval splits."""
        callback = BPCCallback({"eval_got": 4.0, "eval_ang": 3.0})
        metrics = {"eval_got_loss": 2.0, "eval_ang_loss": 1.5}

        callback.on_evaluate(args=None, state=None, control=None, metrics=metrics)

        assert "eval_got_bpc" in metrics
        assert "eval_ang_bpc" in metrics
        assert metrics["eval_got_bpc"] == pytest.approx(2.0 / (4.0 * math.log(2)))
        assert metrics["eval_ang_bpc"] == pytest.approx(1.5 / (3.0 * math.log(2)))

    def test_no_metrics(self):
        """Test that callback handles None metrics gracefully."""
        callback = BPCCallback({"eval": 4.0})
        # Should not raise
        callback.on_evaluate(args=None, state=None, control=None, metrics=None)

    def test_missing_loss_key(self):
        """Test that callback skips splits without a matching loss key."""
        callback = BPCCallback({"eval": 4.0, "eval_other": 3.0})
        metrics = {"eval_loss": 2.0}

        callback.on_evaluate(args=None, state=None, control=None, metrics=metrics)

        assert "eval_bpc" in metrics
        assert "eval_other_bpc" not in metrics
