"""
Tests for evaluation utilities: TTR metrics, BPC computation, and logit preprocessing.
"""

import math
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from datasets import Dataset

from lapt.evaluation import (
    BPCCallback,
    EvalMetrics,
    GenerationChrfCallback,
    batch_nll_by_temperature,
    compute_chars_per_token,
    compute_ttr_metrics,
    count_new_tokens,
    generate_greedy_batched,
    load_external_eval_set,
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


class TestExternalEvalSetLoader:
    """
    Tests for loading and tokenizing external evaluation datasets.

    Testing strategy:
    - Use real files (via tmp_path) to test actual I/O
    - Mock tokenizer for controlled tokenization behavior
    - Test both plaintext and JSONL formats
    - Test error cases (missing file, invalid format, missing columns)
    """

    def test_load_external_eval_plaintext(self, tmp_path):
        """
        Test loading a plaintext external eval set.

        Strategy: Create real text file, mock tokenizer, verify tokenization.
        """
        # Setup: Create test plaintext file
        test_file = tmp_path / "eval_data.txt"
        test_lines = [
            "First eval example",
            "Second eval example",
            "Third eval example",
        ]
        test_file.write_text("\n".join(test_lines))

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3]] * 3,
            'attention_mask': [[1, 1, 1]] * 3
        }

        eval_config = {
            'name': 'held_out',
            'path': str(test_file),
            'format': 'plaintext'
        }

        # Act: Load and tokenize
        dataset = load_external_eval_set(
            eval_config=eval_config,
            tokenizer=mock_tokenizer,
            max_length=512
        )

        # Assert: Verify dataset structure
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 3

        # Verify tokenizer was called correctly
        assert mock_tokenizer.called
        call_args = mock_tokenizer.call_args
        assert call_args[1]['truncation'] is True
        assert call_args[1]['max_length'] == 512

    def test_load_external_eval_jsonl(self, tmp_path):
        """
        Test loading a JSONL external eval set.

        Strategy: Create JSONL file, verify correct field extraction.
        """
        import json

        # Setup: Create test JSONL file
        test_file = tmp_path / "eval_data.jsonl"
        test_data = [
            {"text": "Example 1", "label": "A"},
            {"text": "Example 2", "label": "B"},
        ]
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2]] * 2,
            'attention_mask': [[1, 1]] * 2
        }

        eval_config = {
            'name': 'test_set',
            'path': str(test_file),
            'format': 'jsonl'
        }

        # Act: Load and tokenize
        dataset = load_external_eval_set(
            eval_config=eval_config,
            tokenizer=mock_tokenizer,
            max_length=256
        )

        # Assert: Verify dataset
        assert len(dataset) == 2
        assert mock_tokenizer.called

    def test_load_external_eval_jsonl_custom_column(self, tmp_path):
        """
        Test loading JSONL with custom text column name.

        Strategy: Use different field name, verify it's read correctly.
        """
        import json

        test_file = tmp_path / "eval_data.jsonl"
        test_data = [
            {"content": "Custom field 1"},
            {"content": "Custom field 2"},
        ]
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': [[1]] * 2,
            'attention_mask': [[1]] * 2
        }

        eval_config = {
            'name': 'custom',
            'path': str(test_file),
            'format': 'jsonl',
            'text_column': 'content'
        }

        # Act: Load and tokenize
        dataset = load_external_eval_set(
            eval_config=eval_config,
            tokenizer=mock_tokenizer,
            max_length=256
        )

        # Assert: Should succeed
        assert len(dataset) == 2

    def test_load_external_eval_missing_file(self, tmp_path):
        """
        Test that loading non-existent file raises appropriate error.
        """
        mock_tokenizer = MagicMock()

        eval_config = {
            'name': 'missing',
            'path': str(tmp_path / 'nonexistent.txt')
        }

        with pytest.raises(ValueError) as exc_info:
            load_external_eval_set(
                eval_config=eval_config,
                tokenizer=mock_tokenizer,
                max_length=512
            )

        assert "not found" in str(exc_info.value).lower()

    def test_load_external_eval_invalid_format(self, tmp_path):
        """
        Test that unsupported format raises appropriate error.
        """
        test_file = tmp_path / "data.csv"
        test_file.write_text("col1,col2\nval1,val2")

        mock_tokenizer = MagicMock()

        eval_config = {
            'name': 'csv_test',
            'path': str(test_file),
            'format': 'csv'  # Unsupported format
        }

        with pytest.raises(ValueError) as exc_info:
            load_external_eval_set(
                eval_config=eval_config,
                tokenizer=mock_tokenizer,
                max_length=512
            )

        assert "unsupported format" in str(exc_info.value).lower()

    def test_load_external_eval_jsonl_missing_column(self, tmp_path):
        """
        Test that JSONL with missing text column raises error.
        """
        import json

        test_file = tmp_path / "bad_data.jsonl"
        test_data = [{"wrong_field": "value"}]
        with open(test_file, 'w') as f:
            f.write(json.dumps(test_data[0]) + '\n')

        mock_tokenizer = MagicMock()

        eval_config = {
            'name': 'bad_jsonl',
            'path': str(test_file),
            'format': 'jsonl',
            'text_column': 'text'  # Expected column that doesn't exist
        }

        with pytest.raises(ValueError) as exc_info:
            load_external_eval_set(
                eval_config=eval_config,
                tokenizer=mock_tokenizer,
                max_length=512
            )

        assert "missing" in str(exc_info.value).lower()

    def test_load_external_eval_strips_empty_lines(self, tmp_path):
        """
        Test that empty lines are filtered out from plaintext files.
        """
        test_file = tmp_path / "data.txt"
        test_file.write_text("Line 1\n\n\nLine 2\n   \n")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': [[1]] * 2,
            'attention_mask': [[1]] * 2
        }

        eval_config = {
            'name': 'test',
            'path': str(test_file)
        }

        # Act: Load and tokenize
        dataset = load_external_eval_set(
            eval_config=eval_config,
            tokenizer=mock_tokenizer,
            max_length=512
        )

        # Assert: Should only have 2 non-empty lines
        assert len(dataset) == 2


class TestDataCollatorForInstructionTuning:
    """
    Tests for DataCollatorForInstructionTuning.

    This collator handles batching of instruction-tuning data:
    - Pads input_ids with pad_token_id
    - Pads attention_mask with 0
    - Pads labels with -100 (so padded positions don't contribute to loss)

    Testing strategy:
    - Test with features of different lengths to verify padding
    - Test single-example batch (no padding needed)
    - Verify tensor shapes and dtypes
    """

    def test_basic_padding(self, base_tokenizer):
        """
        Test that features of different lengths are padded correctly.

        Verifies:
        1. All sequences padded to same length
        2. input_ids padded with pad_token_id
        3. attention_mask padded with 0
        4. labels padded with -100
        """
        import torch

        from lapt.evaluation import DataCollatorForInstructionTuning

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        # Features with different lengths
        features = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                'attention_mask': [1, 1, 1, 1, 1],
                'labels': [-100, -100, 3, 4, 5]
            },
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [-100, 2, 3]
            }
        ]

        batch = collator(features)

        # Check shapes - should be padded to longest (5)
        assert batch['input_ids'].shape == (2, 5)
        assert batch['attention_mask'].shape == (2, 5)
        assert batch['labels'].shape == (2, 5)

        # Check dtypes
        assert batch['input_ids'].dtype == torch.long
        assert batch['labels'].dtype == torch.long

        # Check padding values for second (shorter) sequence
        # Last 2 positions should be padded
        pad_token_id = base_tokenizer.pad_token_id or base_tokenizer.eos_token_id
        assert batch['input_ids'][1, 3].item() == pad_token_id
        assert batch['input_ids'][1, 4].item() == pad_token_id
        assert batch['attention_mask'][1, 3].item() == 0
        assert batch['attention_mask'][1, 4].item() == 0
        assert batch['labels'][1, 3].item() == -100
        assert batch['labels'][1, 4].item() == -100

    def test_left_padding_keeps_labels_aligned(self, base_tokenizer):
        """
        Test that labels are padded on the tokenizer's padding side.

        A tokenizer saved after left-padded batch generation reloads with
        padding_side='left'. If labels were still right-padded, every shorter
        example's labels would be shifted relative to its input_ids.
        """
        from lapt.evaluation import DataCollatorForInstructionTuning

        original_padding_side = base_tokenizer.padding_side
        base_tokenizer.padding_side = 'left'
        try:
            collator = DataCollatorForInstructionTuning(base_tokenizer)
            features = [
                {
                    'input_ids': [1, 2, 3, 4, 5],
                    'attention_mask': [1, 1, 1, 1, 1],
                    'labels': [-100, -100, 3, 4, 5]
                },
                {
                    'input_ids': [1, 2, 3],
                    'attention_mask': [1, 1, 1],
                    'labels': [-100, 2, 3]
                }
            ]
            batch = collator(features)
        finally:
            base_tokenizer.padding_side = original_padding_side

        # every unmasked label must sit under its own token
        for row_input_ids, row_labels in zip(batch['input_ids'], batch['labels']):
            for token_id, label in zip(row_input_ids.tolist(), row_labels.tolist()):
                if label != -100:
                    assert label == token_id
        assert batch['labels'][1].tolist() == [-100, -100, -100, 2, 3]

    def test_single_example_batch(self, base_tokenizer):
        """
        Test batch with single example (no padding needed).
        """
        from lapt.evaluation import DataCollatorForInstructionTuning

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        features = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [-100, 2, 3]
            }
        ]

        batch = collator(features)

        # Should have batch size 1
        assert batch['input_ids'].shape == (1, 3)
        assert batch['labels'].shape == (1, 3)

        # Values should be unchanged
        assert batch['input_ids'][0].tolist() == [1, 2, 3]
        assert batch['labels'][0].tolist() == [-100, 2, 3]

    def test_preserves_masked_labels(self, base_tokenizer):
        """
        Test that -100 labels are preserved (not overwritten by padding logic).
        """
        from lapt.evaluation import DataCollatorForInstructionTuning

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        features = [
            {
                'input_ids': [1, 2, 3, 4],
                'attention_mask': [1, 1, 1, 1],
                'labels': [-100, -100, 3, 4]  # First two are masked
            }
        ]

        batch = collator(features)

        # Original -100 values should be preserved
        assert batch['labels'][0, 0].item() == -100
        assert batch['labels'][0, 1].item() == -100
        assert batch['labels'][0, 2].item() == 3
        assert batch['labels'][0, 3].item() == 4

    def test_handles_all_masked_labels(self, base_tokenizer):
        """
        Test handling of sequence where all labels are -100.

        This can happen with empty responses or very long prompts.
        """
        from lapt.evaluation import DataCollatorForInstructionTuning

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        features = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [-100, -100, -100]  # All masked
            }
        ]

        batch = collator(features)

        # Should work without errors
        assert batch['labels'].shape == (1, 3)
        assert batch['labels'][0].tolist() == [-100, -100, -100]

    def test_returns_pytorch_tensors(self, base_tokenizer):
        """
        Test that output is PyTorch tensors, not lists.
        """
        import torch

        from lapt.evaluation import DataCollatorForInstructionTuning

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        features = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [-100, 2, 3]
            }
        ]

        batch = collator(features)

        assert isinstance(batch['input_ids'], torch.Tensor)
        assert isinstance(batch['attention_mask'], torch.Tensor)
        assert isinstance(batch['labels'], torch.Tensor)


class _RecordingGenerationModel:
    """Stand-in causal LM whose generate() records its inputs and appends two EOS
    tokens, so generate_greedy_batched can run without real weights."""

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id
        self.config = SimpleNamespace(use_cache=False)
        self.generate_calls = []

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, input_ids, attention_mask, **kwargs):
        self.generate_calls.append({'input_ids': input_ids, 'attention_mask': attention_mask})
        # (batch_size, 2)
        new_tokens = torch.full((input_ids.shape[0], 2), self.eos_token_id)
        return torch.cat([input_ids, new_tokens], dim=1)


class TestGenerateGreedyBatched:
    """Tests for in-training batched generation's handling of the tokenizer."""

    def _run(self, base_tokenizer, prompts, max_prompt_length=64):
        model = _RecordingGenerationModel(base_tokenizer.eos_token_id)
        generate_greedy_batched(
            model=model,
            tokenizer=base_tokenizer,
            prompts=prompts,
            max_new_tokens=2,
            max_prompt_length=max_prompt_length,
            batch_size=len(prompts),
            stop_strings=[],
        )
        return model.generate_calls[0]

    def test_leaves_tokenizer_state_untouched(self, base_tokenizer):
        """
        Generation must not change the tokenizer's padding or truncation.

        This is the training tokenizer, which the Trainer saves into every
        checkpoint. A padding setting left on the fast tokenizer's backend was
        written into tokenizer.json and reloaded as padding_side='left'.
        """
        original_padding_side = base_tokenizer.padding_side
        self._run(base_tokenizer, ["a short prompt", "a much longer prompt than the first"])

        assert base_tokenizer.padding_side == original_padding_side
        assert base_tokenizer.backend_tokenizer.padding is None
        assert base_tokenizer.backend_tokenizer.truncation is None

    def test_left_pads_and_truncates_prompts(self, base_tokenizer):
        """Prompts are left-padded to the batch width and truncated to the cap."""
        prompts = ["a short prompt", "a much longer prompt than the first"]
        encoded = [base_tokenizer(prompt)['input_ids'] for prompt in prompts]
        call = self._run(base_tokenizer, prompts)

        width = max(len(ids) for ids in encoded)
        assert call['input_ids'].shape == (2, width)
        short_ids = encoded[0]
        assert call['input_ids'][0, width - len(short_ids):].tolist() == short_ids
        assert call['attention_mask'][0].tolist() == (
            [0] * (width - len(short_ids)) + [1] * len(short_ids)
        )

        truncated_call = self._run(base_tokenizer, prompts[1:], max_prompt_length=3)
        assert truncated_call['input_ids'][0].tolist() == encoded[1][:3]


class TestBatchNllByTemperature:
    """Tests for the per-batch temperature-scaled NLL used by calibrated bpc."""

    def test_unit_temperature_matches_causal_lm_loss(self):
        """At T = 1 the result is the shifted, masked token-mean cross-entropy."""
        torch.manual_seed(1)
        # (batch_size, seq_len, vocab_size)
        logits = torch.randn(3, 7, 11)
        labels = torch.randint(0, 11, (3, 7))
        labels[0, :3] = -100
        labels[2, 5:] = -100

        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 11),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        # a small chunk size exercises chunk boundaries
        result = batch_nll_by_temperature(logits, labels, [1.0, 2.0], chunk_size=4)

        assert result[0].item() == pytest.approx(expected.item(), rel=1e-5)
        assert result[1].item() != pytest.approx(expected.item(), rel=1e-3)

    def test_temperature_divides_logits(self):
        """Scoring logits at T equals scoring logits / T at T = 1."""
        torch.manual_seed(1)
        logits = torch.randn(2, 5, 9)
        labels = torch.randint(0, 9, (2, 5))

        scaled = batch_nll_by_temperature(logits, labels, [2.5])
        reference = batch_nll_by_temperature(logits / 2.5, labels, [1.0])

        assert scaled[0].item() == pytest.approx(reference[0].item(), rel=1e-5)

    def test_no_loss_tokens_gives_nan(self):
        logits = torch.randn(1, 4, 5)
        labels = torch.full((1, 4), -100)

        result = batch_nll_by_temperature(logits, labels, [1.0, 2.0])

        assert torch.isnan(result).all()


class TestEvalMetrics:
    """Tests for the combined TTR / calibration Trainer metric hooks."""

    def test_recovers_overconfidence_temperature(self):
        """Logits sharpened by 4x around the true distribution are best at T = 4."""
        torch.manual_seed(1)
        # (batch_size, seq_len, vocab_size)
        true_logits = torch.randn(16, 128, 20)
        # sample each next token from the true distribution, then sharpen the logits
        sampled = torch.distributions.Categorical(logits=true_logits).sample()
        labels = torch.full((16, 128), -100)
        labels[:, 1:] = sampled[:, :-1]

        eval_metrics = EvalMetrics(
            compute_ttr=False,
            calibration_temperatures=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        predictions = eval_metrics.preprocess_logits(4.0 * true_logits, labels)
        metrics = eval_metrics.compute_metrics((predictions.numpy(), labels.numpy()))

        assert metrics['calibrated_temperature'] == 4.0
        unit_loss = batch_nll_by_temperature(4.0 * true_logits, labels, [1.0])[0].item()
        assert metrics['calibrated_loss'] < unit_loss

    def test_ttr_and_calibration_together(self):
        """With both enabled, preprocess returns a tuple and both metrics appear."""
        torch.manual_seed(1)
        logits = torch.randn(2, 6, 8)
        labels = torch.randint(0, 8, (2, 6))

        eval_metrics = EvalMetrics(compute_ttr=True, calibration_temperatures=[1.0, 2.0])
        argmax_ids, losses = eval_metrics.preprocess_logits(logits, labels)
        assert argmax_ids.shape == (2, 6)
        assert losses.shape == (2, 2)

        metrics = eval_metrics.compute_metrics(
            ((argmax_ids.numpy(), losses.numpy()), labels.numpy())
        )
        assert 'ttr-seq' in metrics
        assert metrics['calibrated_temperature'] in (1.0, 2.0)

    def test_requires_a_diagnostic(self):
        with pytest.raises(ValueError):
            EvalMetrics(compute_ttr=False, calibration_temperatures=None)

    def test_unit_temperature_reproduces_trainer_eval_loss(self, tmp_path):
        """
        Through a real Trainer, calibrated loss on a T = 1 grid equals eval_loss.

        Uses uneven example lengths and a partial final batch, so this checks
        both that the hook sees correctly aligned labels and that the per-example
        expansion reproduces the Trainer's batch-size weighting.
        """
        from transformers import Trainer, TrainingArguments, XGLMConfig, XGLMForCausalLM

        from lapt.evaluation import DataCollatorForInstructionTuning

        torch.manual_seed(1)
        config = XGLMConfig(
            vocab_size=32, d_model=16, num_layers=1, attention_heads=2, ffn_dim=32,
            max_position_embeddings=64, pad_token_id=1,
        )
        model = XGLMForCausalLM(config)

        examples = []
        for length in [5, 9, 7, 12, 4]:
            input_ids = torch.randint(2, 32, (length,)).tolist()
            examples.append({
                'input_ids': input_ids,
                'attention_mask': [1] * length,
                'labels': [-100, -100] + input_ids[2:],
            })
        dataset = Dataset.from_list(examples)

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 1
        tokenizer.padding_side = 'right'

        def pad(features, padding, max_length, return_tensors):
            width = max(len(feature['input_ids']) for feature in features)
            return {
                'input_ids': torch.tensor([
                    feature['input_ids'] + [1] * (width - len(feature['input_ids']))
                    for feature in features
                ]),
                'attention_mask': torch.tensor([
                    feature['attention_mask'] + [0] * (width - len(feature['input_ids']))
                    for feature in features
                ]),
            }

        tokenizer.pad = pad
        eval_metrics = EvalMetrics(compute_ttr=False, calibration_temperatures=[1.0])
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(tmp_path), per_device_eval_batch_size=2,
                report_to=[], use_cpu=True,
            ),
            data_collator=DataCollatorForInstructionTuning(tokenizer),
            compute_metrics=eval_metrics.compute_metrics,
            preprocess_logits_for_metrics=eval_metrics.preprocess_logits,
        )
        metrics = trainer.evaluate(eval_dataset=dataset)

        assert metrics['eval_calibrated_loss'] == pytest.approx(metrics['eval_loss'], rel=1e-5)
        assert metrics['eval_calibrated_temperature'] == 1.0


class TestBPCCallbackCalibrated:
    def test_converts_calibrated_loss(self):
        """Calibrated loss is converted with the same chars-per-token ratio."""
        callback = BPCCallback({"eval_got": 4.0})
        metrics = {"eval_got_loss": 2.0, "eval_got_calibrated_loss": 1.5}
        state = _make_mock_state(dict(metrics))

        callback.on_evaluate(args=None, state=state, control=None, metrics=metrics)

        expected = 1.5 / (4.0 * math.log(2))
        assert metrics["eval_got_calibrated_bpc"] == pytest.approx(expected)
        assert state.log_history[-1]["eval_got_calibrated_bpc"] == pytest.approx(expected)
