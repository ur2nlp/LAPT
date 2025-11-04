"""Tests for tokenizer_utils module."""

import pytest

from tokenizer_utils import (
    _detect_tokenizer_algorithm,
    _extract_special_tokens,
    _validate_tokenizer,
    train_new_tokenizer,
)


class TestTokenizerAlgorithmDetection:
    """
    Tests for detecting whether a tokenizer uses BPE or Unigram algorithm.

    Testing strategy:
    - Use real tokenizer from HF (via fixture) rather than mocking
    - Test known tokenizers: XGLM (Unigram), GPT-2 (BPE)
    """

    def test_detect_unigram_xglm(self, base_tokenizer):
        """
        Test that XGLM tokenizer is correctly identified as Unigram.

        Design note: We pass base_tokenizer as a parameter, and pytest
        automatically injects the fixture we defined in conftest.py.
        """
        algorithm = _detect_tokenizer_algorithm(base_tokenizer)
        assert algorithm == "unigram"

    def test_detect_bpe_gpt2(self):
        """
        Test that GPT-2 tokenizer is correctly identified as BPE.

        Strategy note: We import here rather than using a fixture because
        we only need GPT-2 for this one test, so no need to cache it.
        """
        from transformers import AutoTokenizer

        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        algorithm = _detect_tokenizer_algorithm(gpt2_tokenizer)
        assert algorithm == "bpe"


class TestSpecialTokenExtraction:
    """
    Tests for extracting special token configuration for SentencePiece training.

    Testing strategy:
    - Test both modes: with and without additional special tokens
    - Verify the returned dict has correct SentencePiece parameter names
    - Test with XGLM which has <madeupword0-6> as additional tokens
    """

    def test_extract_with_additional_tokens(self, base_tokenizer):
        """
        Test extraction with inherit_additional=True (includes <madeupword> tokens).

        What we're verifying:
        1. Core special tokens (BOS, EOS, UNK, PAD) are extracted
        2. Additional tokens (like <madeupword0-6>) are included
        3. Output format matches SentencePiece expectations
        """
        config = _extract_special_tokens(base_tokenizer, inherit_additional=True)

        # Check core special tokens are present with correct SentencePiece parameter names
        assert "bos_piece" in config
        assert "eos_piece" in config
        assert "unk_piece" in config
        assert "pad_piece" in config
        assert config["bos_piece"] == "<s>"
        assert config["eos_piece"] == "</s>"
        assert config["unk_piece"] == "<unk>"
        assert config["pad_piece"] == "<pad>"

        # Check token IDs are present
        assert "bos_id" in config
        assert "eos_id" in config
        assert config["bos_id"] == 0
        assert config["eos_id"] == 2

        # Check that additional special tokens are included
        # XGLM has <madeupword0> through <madeupword6>
        assert "user_defined_symbols" in config
        assert "<madeupword0>" in config["user_defined_symbols"]

    def test_extract_without_additional_tokens(self, base_tokenizer):
        """
        Test extraction with inherit_additional=False (excludes <madeupword> tokens).

        This mode maximizes vocabulary space for target language by not
        reserving slots for unused placeholder tokens.
        """
        config = _extract_special_tokens(base_tokenizer, inherit_additional=False)

        # Core tokens should still be present
        assert "bos_piece" in config
        assert "eos_piece" in config
        assert "unk_piece" in config
        assert "pad_piece" in config

        # Additional tokens should NOT be included
        assert "user_defined_symbols" not in config


class TestTokenizerValidation:
    """
    Tests for tokenizer validation logic.

    Testing strategy:
    - Test happy path: validation passes with correct tokenizer
    - Test error cases: validation fails with appropriate error messages
    - Use pytest.raises to verify exceptions are raised correctly

    Design decision: We test _validate_tokenizer() separately rather than
    only testing it as part of train_new_tokenizer() because:
    1. Faster: don't need to train a tokenizer to test validation
    2. Clearer: failures point directly to validation logic
    3. More thorough: easier to test edge cases
    """

    def test_validate_correct_tokenizer(self, base_tokenizer):
        """
        Test that validation passes for a correctly configured tokenizer.

        The base_tokenizer has 256,008 tokens, so we validate against that.
        If validation passes, the function returns None (doesn't raise).
        """
        expected_size = len(base_tokenizer)
        _validate_tokenizer(base_tokenizer, expected_size)

    def test_validate_wrong_size(self, base_tokenizer):
        """
        Test that validation fails when expected size doesn't match actual size.

        pytest.raises is a context manager that:
        1. Expects the code inside to raise the specified exception
        2. Fails the test if no exception is raised
        3. Fails the test if a different exception is raised
        """
        wrong_size = 1000

        with pytest.raises(ValueError) as exc_info:
            _validate_tokenizer(base_tokenizer, wrong_size)

        # We can also check the error message
        assert "vocab size" in str(exc_info.value).lower()

    def test_validate_non_contiguous_ids_mock(self):
        """
        Test that validation catches non-contiguous token IDs.

        Strategy decision: We need to mock a broken tokenizer here because
        real tokenizers from HF are always valid. This is one case where
        mocking is necessary to test error handling.

        We create a minimal mock that:
        - Has len() = 10
        - Has get_vocab() returning IDs 1-10 (missing 0!)
        """
        class MockBrokenTokenizer:
            def __len__(self):
                return 10

            def get_vocab(self):
                # Missing ID 0 - should trigger validation error
                return {f"token_{i}": i for i in range(1, 11)}

        broken_tokenizer = MockBrokenTokenizer()

        with pytest.raises(ValueError) as exc_info:
            _validate_tokenizer(broken_tokenizer, expected_vocab_size=10)

        error_msg = str(exc_info.value).lower()
        assert "contiguous" in error_msg or "range" in error_msg


class TestTokenizerTraining:
    """
    Integration tests for the full tokenizer training pipeline.

    Testing strategy:
    - Actually run SentencePiece training (fast enough on small data)
    - Use tmp_path for outputs (automatic cleanup)
    - Test both with/without additional special tokens
    - Verify caching behavior
    - Use small vocab (64 tokens) for speed

    These are integration tests because they exercise the full pipeline:
    JSONL → text conversion → SentencePiece training → HF tokenizer wrapping
    """

    def test_train_tokenizer_with_additional_tokens(
        self, sample_jsonl_path, base_tokenizer, tmp_path
    ):
        """
        Test training a tokenizer that inherits additional special tokens.

        Fixtures used:
        - sample_jsonl_path: Path to our 10-line fixture data
        - base_tokenizer: XGLM tokenizer for algorithm/special tokens
        - tmp_path: Temporary directory for outputs (auto-cleaned)

        What we verify:
        1. Training completes without errors
        2. Output files are created (tokenizer.json, spm.model, etc.)
        3. Vocab size matches request
        4. Additional special tokens are present
        5. Tokenizer can actually tokenize text
        """
        vocab_size = 64
        output_dir = tmp_path / "tokenizer_with_additional"

        tokenizer = train_new_tokenizer(
            jsonl_path=str(sample_jsonl_path),
            base_tokenizer_name="facebook/xglm-564M",
            vocab_size=vocab_size,
            output_path=str(output_dir),
            inherit_additional_special_tokens=True
        )

        # Verify tokenizer object
        assert tokenizer is not None
        assert len(tokenizer) == vocab_size

        # Verify additional special tokens were inherited
        assert hasattr(tokenizer, 'additional_special_tokens')
        assert tokenizer.additional_special_tokens is not None
        assert len(tokenizer.additional_special_tokens) > 0
        assert "<madeupword0>" in tokenizer.additional_special_tokens

        # Verify output files were created
        assert (output_dir / "tokenizer.json").exists()
        assert (output_dir / "spm.model").exists()
        assert (output_dir / "spm.vocab").exists()

        # Verify tokenizer actually works
        test_text = "This is a test sentence."
        tokens = tokenizer.tokenize(test_text)
        assert len(tokens) > 0

        token_ids = tokenizer.encode(test_text)
        assert len(token_ids) > 0
        assert all(0 <= tid < vocab_size for tid in token_ids)

    def test_train_tokenizer_without_additional_tokens(self, sample_jsonl_path, tmp_path):
        """
        Test training a tokenizer without additional special tokens.

        This mode maximizes vocabulary space for the target language.
        """
        vocab_size = 64
        output_dir = tmp_path / "tokenizer_without_additional"

        tokenizer = train_new_tokenizer(
            jsonl_path=str(sample_jsonl_path),
            base_tokenizer_name="facebook/xglm-564M",
            vocab_size=vocab_size,
            output_path=str(output_dir),
            inherit_additional_special_tokens=False
        )

        assert len(tokenizer) == vocab_size

        # Additional special tokens should be empty or None
        if hasattr(tokenizer, 'additional_special_tokens'):
            assert tokenizer.additional_special_tokens is None or \
                   len(tokenizer.additional_special_tokens) == 0

    def test_tokenizer_caching(self, sample_jsonl_path, tmp_path):
        """
        Test that train_new_tokenizer() reuses existing tokenizer instead of retraining.

        Strategy: Train once, then call again with same output_path.
        Second call should be much faster and return the same tokenizer.

        We verify caching by checking that output files have old timestamps.
        """
        vocab_size = 64
        output_dir = tmp_path / "tokenizer_cached"

        # First call: actually trains
        tokenizer1 = train_new_tokenizer(
            jsonl_path=str(sample_jsonl_path),
            base_tokenizer_name="facebook/xglm-564M",
            vocab_size=vocab_size,
            output_path=str(output_dir),
            inherit_additional_special_tokens=True
        )

        # Get timestamp of created file
        tokenizer_file = output_dir / "tokenizer.json"
        mtime_before = tokenizer_file.stat().st_mtime

        # Second call: should load from cache
        tokenizer2 = train_new_tokenizer(
            jsonl_path=str(sample_jsonl_path),
            base_tokenizer_name="facebook/xglm-564M",
            vocab_size=vocab_size,
            output_path=str(output_dir),
            inherit_additional_special_tokens=True
        )

        # Verify file wasn't modified (indicates it was loaded, not retrained)
        mtime_after = tokenizer_file.stat().st_mtime
        assert mtime_after == mtime_before

        # Both tokenizers should have same vocab size
        assert len(tokenizer1) == len(tokenizer2) == vocab_size
