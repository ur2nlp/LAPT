"""
Tests for evaluation utilities: TTR metrics, BPC computation, and logit preprocessing.
"""

import math
from collections import namedtuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from datasets import Dataset

from lapt.eval_utils import (
    BPCCallback,
    GenerationChrfCallback,
    compute_chars_per_token,
    compute_ttr_metrics,
    count_new_tokens,
    load_instruction_prompts,
    preprocess_logits_for_metrics,
    truncate_at_stop,
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

        # Only non-masked positions are decoded, so prompt characters cannot
        # inflate the ratio: chars = "ef"+"gh" = 4
        # non-masked labels = 2 (tokens 3 and 4), loss tokens = 2-1 = 1
        # ratio = 4/1 = 4.0
        assert result == {"eval": 4.0}

    def test_empty_dataset(self):
        """Test with an empty dataset."""
        tokenizer = _make_mock_tokenizer({})
        ds = Dataset.from_dict({"input_ids": []})
        result = compute_chars_per_token(ds, tokenizer)
        assert result == {"eval": 0.0}


def _make_mock_state(log_entry: dict = None):
    """Create a mock TrainerState with a log_history list."""
    state = MagicMock()
    state.log_history = [log_entry] if log_entry is not None else []
    return state


class TestBPCCallback:
    def test_injects_bpc(self):
        """Test that BPC is correctly computed and injected into metrics."""
        callback = BPCCallback({"eval": 4.0})
        metrics = {"eval_loss": 2.0}
        state = _make_mock_state({"eval_loss": 2.0})

        callback.on_evaluate(args=None, state=state, control=None, metrics=metrics)

        expected_bpc = 2.0 / (4.0 * math.log(2))
        assert metrics["eval_bpc"] == pytest.approx(expected_bpc)

    def test_patches_log_history(self):
        """Test that BPC values are patched into state.log_history."""
        callback = BPCCallback({"eval": 4.0})
        metrics = {"eval_loss": 2.0}
        log_entry = {"eval_loss": 2.0, "step": 100}
        state = _make_mock_state(log_entry)

        callback.on_evaluate(args=None, state=state, control=None, metrics=metrics)

        expected_bpc = 2.0 / (4.0 * math.log(2))
        assert state.log_history[-1]["eval_bpc"] == pytest.approx(expected_bpc)

    def test_multi_split(self):
        """Test BPC injection for multiple eval splits."""
        callback = BPCCallback({"eval_got": 4.0, "eval_ang": 3.0})
        metrics = {"eval_got_loss": 2.0, "eval_ang_loss": 1.5}
        state = _make_mock_state({"eval_got_loss": 2.0, "eval_ang_loss": 1.5})

        callback.on_evaluate(args=None, state=state, control=None, metrics=metrics)

        assert "eval_got_bpc" in metrics
        assert "eval_ang_bpc" in metrics
        assert metrics["eval_got_bpc"] == pytest.approx(2.0 / (4.0 * math.log(2)))
        assert metrics["eval_ang_bpc"] == pytest.approx(1.5 / (3.0 * math.log(2)))

    def test_no_metrics(self):
        """Test that callback handles None metrics gracefully."""
        callback = BPCCallback({"eval": 4.0})
        state = _make_mock_state()
        # Should not raise
        callback.on_evaluate(args=None, state=state, control=None, metrics=None)

    def test_missing_loss_key(self):
        """Test that callback skips splits without a matching loss key."""
        callback = BPCCallback({"eval": 4.0, "eval_other": 3.0})
        metrics = {"eval_loss": 2.0}
        state = _make_mock_state({"eval_loss": 2.0})

        callback.on_evaluate(args=None, state=state, control=None, metrics=metrics)

        assert "eval_bpc" in metrics
        assert "eval_other_bpc" not in metrics


class TestCountNewTokens:
    """Tests for measuring the true length of a single generation."""

    def test_halted_before_padding(self):
        """A sequence right-padded with EOS counts only up to the first EOS."""
        # generate() pads finished sequences out to the batch's longest, and PAD
        # is EOS here, so the trailing 2s are padding rather than content.
        row_ids = torch.tensor([5, 6, 7, 2, 2, 2])

        token_count, hit_cap = count_new_tokens(row_ids, eos_token_id=2)

        assert token_count == 4
        assert hit_cap is False

    def test_no_eos_means_hit_cap(self):
        """A sequence with no EOS never halted and was cut off by the cap."""
        row_ids = torch.tensor([5, 6, 7, 8])

        token_count, hit_cap = count_new_tokens(row_ids, eos_token_id=2)

        assert token_count == 4
        assert hit_cap is True

    def test_immediate_eos(self):
        """A model that emits EOS first thing generated exactly one token."""
        row_ids = torch.tensor([2, 2, 2])

        token_count, hit_cap = count_new_tokens(row_ids, eos_token_id=2)

        assert token_count == 1
        assert hit_cap is False

    def test_eos_in_final_position(self):
        """EOS as the last token counts the full row and still counts as halted."""
        row_ids = torch.tensor([5, 6, 2])

        token_count, hit_cap = count_new_tokens(row_ids, eos_token_id=2)

        assert token_count == 3
        assert hit_cap is False


class TestHitCapWarning:
    """Tests for the stderr warning on a high non-halting rate."""

    def _make_callback(self, threshold: float = 0.25):
        """Build a callback without running __init__, which reads data files."""
        callback = GenerationChrfCallback.__new__(GenerationChrfCallback)
        callback._warned_hit_cap = set()
        spec = {
            'name': 'got',
            'max_new_tokens': 128,
            'hit_cap_warn_threshold': threshold,
        }
        return callback, spec

    def test_warns_above_threshold(self, capsys):
        callback, spec = self._make_callback()

        callback._maybe_warn_hit_cap(spec, hit_cap_frac=0.4, global_step=100)

        captured = capsys.readouterr()
        assert "40.0% of generations hit max_new_tokens" in captured.err
        assert captured.out == ""

    def test_silent_below_threshold(self, capsys):
        callback, spec = self._make_callback()

        callback._maybe_warn_hit_cap(spec, hit_cap_frac=0.1, global_step=100)

        assert capsys.readouterr().err == ""

    def test_warns_once_while_persistent(self, capsys):
        """A sustained problem should not spam every eval cycle."""
        callback, spec = self._make_callback()

        callback._maybe_warn_hit_cap(spec, hit_cap_frac=0.4, global_step=100)
        capsys.readouterr()
        callback._maybe_warn_hit_cap(spec, hit_cap_frac=0.5, global_step=200)

        assert capsys.readouterr().err == ""

    def test_rearms_after_recovery(self, capsys):
        """Dropping below the threshold re-arms the warning for a relapse."""
        callback, spec = self._make_callback()

        callback._maybe_warn_hit_cap(spec, hit_cap_frac=0.4, global_step=100)
        callback._maybe_warn_hit_cap(spec, hit_cap_frac=0.05, global_step=200)
        capsys.readouterr()
        callback._maybe_warn_hit_cap(spec, hit_cap_frac=0.4, global_step=300)

        assert "40.0% of generations hit max_new_tokens" in capsys.readouterr().err


class TestTruncateAtStop:
    """Unit tests for truncate_at_stop."""

    def test_returns_text_unchanged_when_no_stop_present(self):
        assert truncate_at_stop("a complete answer", ["\n\n", "###"]) == "a complete answer"

    def test_returns_text_unchanged_for_empty_stop_list(self):
        assert truncate_at_stop("a complete answer", []) == "a complete answer"

    def test_truncates_at_the_stop_string(self):
        assert truncate_at_stop("answer###trailing", ["###"]) == "answer"

    def test_cuts_at_the_earliest_stop_regardless_of_list_order(self):
        """
        The cut is the minimum index over all stop strings, not the first stop
        that happens to match. Reordering the list must not change the result,
        or the truncation would depend on config ordering.
        """
        text = "answer<eos>tail###more"
        assert truncate_at_stop(text, ["###", "<eos>"]) == "answer"
        assert truncate_at_stop(text, ["<eos>", "###"]) == "answer"

    def test_returns_empty_string_when_stop_is_at_the_start(self):
        assert truncate_at_stop("###everything", ["###"]) == ""


class TestLoadInstructionPrompts:
    """Unit tests for load_instruction_prompts."""

    @staticmethod
    def _write(tmp_path, text):
        path = tmp_path / "eval.jsonl"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_loads_parallel_prompt_and_reference_lists(self, tmp_path):
        path = self._write(
            tmp_path,
            '{"prompt": "p1", "response": "r1"}\n'
            '{"prompt": "p2", "response": "r2"}\n',
        )

        prompts, references = load_instruction_prompts(path)

        assert prompts == ["p1", "p2"]
        assert references == ["r1", "r2"]

    def test_skips_blank_lines(self, tmp_path):
        path = self._write(
            tmp_path,
            '{"prompt": "p1", "response": "r1"}\n'
            "\n"
            "   \n"
            '{"prompt": "p2", "response": "r2"}\n',
        )

        prompts, references = load_instruction_prompts(path)

        assert prompts == ["p1", "p2"]
        assert references == ["r1", "r2"]

    def test_raises_file_not_found_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="chrF eval data file not found"):
            load_instruction_prompts(str(tmp_path / "absent.jsonl"))

    def test_reports_the_physical_line_number_of_a_malformed_entry(self, tmp_path):
        """
        Blank lines are skipped but still counted, so the number in the error
        matches what an editor shows for the offending line.
        """
        path = self._write(
            tmp_path,
            '{"prompt": "p1", "response": "r1"}\n'
            "\n"
            '{"prompt": "p2"}\n',
        )

        with pytest.raises(ValueError, match="Line 3"):
            load_instruction_prompts(path)

    def test_raises_when_response_field_is_missing(self, tmp_path):
        path = self._write(tmp_path, '{"response": "r1"}\n')

        with pytest.raises(ValueError, match="missing 'prompt' or 'response'"):
            load_instruction_prompts(path)

    def test_raises_when_no_examples_are_loaded(self, tmp_path):
        path = self._write(tmp_path, "\n   \n\n")

        with pytest.raises(ValueError, match="No examples loaded"):
            load_instruction_prompts(path)


class TestComputeTtrMetricsInputHandling:
    """Input-shape and mask handling in compute_ttr_metrics."""

    def test_accepts_torch_tensor_predictions(self):
        """Trainer may hand back tensors rather than numpy arrays."""
        predictions = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[1, 2, 3, 4]])

        metrics = compute_ttr_metrics(EvalPrediction(predictions, labels))

        assert metrics["ttr-seq"] == pytest.approx(1.0)

    def test_rejects_three_dimensional_logits(self):
        """
        Passing raw logits instead of argmaxed ids is the expected misuse, and
        it would otherwise produce a meaningless TTR rather than an error.
        """
        predictions = np.zeros((2, 4, 8))
        labels = np.zeros((2, 4))

        with pytest.raises(ValueError, match="3D predictions"):
            compute_ttr_metrics(EvalPrediction(predictions, labels))

    def test_uses_all_positions_when_labels_are_absent(self):
        """With no label_ids there is no padding information to mask on."""
        predictions = np.array([[5, 5, 6, 7]])

        metrics = compute_ttr_metrics(EvalPrediction(predictions, None))

        assert metrics["ttr-seq"] == pytest.approx(0.75)

    def test_returns_zero_when_every_position_is_masked(self):
        """An all-padding batch has no tokens to measure diversity over."""
        predictions = np.array([[1, 2, 3, 4]])
        labels = np.full((1, 4), -100)

        metrics = compute_ttr_metrics(EvalPrediction(predictions, labels))

        assert metrics["ttr-seq"] == 0.0
