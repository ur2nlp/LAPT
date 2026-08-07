"""Tests for tokenizer_utils module."""

import pytest

from artifact_configs import TokenizerConfig
from tokenizer_utils import (
    _detect_tokenizer_algorithm,
    _extract_special_tokens,
    _resolve_hf_special_tokens,
    _validate_tokenizer,
    train_new_tokenizer,
)


class StubTokenizer:
    """
    Minimal stand-in exposing only the attributes _extract_special_tokens reads.

    Used to cover base tokenizers whose special-token layout differs from XGLM's
    without downloading them. The Qwen3 values below were read off
    ``Qwen/Qwen3-0.6B-Base`` with transformers 4.57.6 (2026-08-07).
    """

    def __init__(
        self,
        unk_token=None,
        unk_token_id=None,
        bos_token=None,
        bos_token_id=None,
        eos_token=None,
        eos_token_id=None,
        pad_token=None,
        pad_token_id=None,
        additional_special_tokens=None,
    ):
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id
        self.bos_token = bos_token
        self.bos_token_id = bos_token_id
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.additional_special_tokens = additional_special_tokens or []


def make_qwen_stub() -> StubTokenizer:
    """Stub matching Qwen3's layout: no unk, no bos, pad aliased to eos at 151643."""
    return StubTokenizer(
        eos_token="<|endoftext|>",
        eos_token_id=151643,
        pad_token="<|endoftext|>",
        pad_token_id=151643,
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )


def make_tokenizer_config(
    vocab_size: int,
    num_samples: int = None,
    inherit_additional_special_tokens: bool = True,
    use_seed_vocabulary: bool = False,
    seed_lambda: float = 0.5,
    seed_vocab_multiplier: float = 5.0,
    seed_score_mode: str = "count",
    character_coverage: float = 1.0,
    hf_model: str = "facebook/xglm-564M",
    tokenizer_algorithm: str = None,
) -> TokenizerConfig:
    """Helper to create TokenizerConfig for tests with sensible defaults."""
    return TokenizerConfig(
        hf_model=hf_model,
        vocab_size=vocab_size,
        num_samples=num_samples,
        character_coverage=character_coverage,
        inherit_additional_special_tokens=inherit_additional_special_tokens,
        tokenizer_algorithm=tokenizer_algorithm,
        use_seed_vocabulary=use_seed_vocabulary,
        seed_vocab_multiplier=seed_vocab_multiplier,
        seed_lambda=seed_lambda,
        seed_min_frequency=1,
        seed_round_mode="round",
        seed_score_mode=seed_score_mode,
        fasttext_model_min_count=4,
        seed=42,
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
        # The newline piece is always reserved regardless of inherited tokens
        assert "\n" in config["user_defined_symbols"]

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

        # The newline piece is always reserved, so user_defined_symbols is present
        # even without additional tokens — but the <madeupword> placeholders are not.
        assert "user_defined_symbols" in config
        assert "\n" in config["user_defined_symbols"]
        assert not any(
            token.startswith("<madeupword") for token in config["user_defined_symbols"]
        )

    def test_base_ids_preserved_when_usable(self, base_tokenizer):
        """
        XGLM's own ids fit in the target vocab and are unique, so they survive.

        This pins the backward-compatibility guarantee: adapting the extraction
        logic for other base models must not silently renumber XGLM tokenizers,
        which would invalidate every cached tokenizer and FOCUS embedding artifact.
        """
        config = _extract_special_tokens(base_tokenizer, vocab_size=32768)

        assert config["unk_id"] == base_tokenizer.unk_token_id
        assert config["bos_id"] == base_tokenizer.bos_token_id
        assert config["eos_id"] == base_tokenizer.eos_token_id
        assert config["pad_id"] == base_tokenizer.pad_token_id


class TestSpecialTokensNonXglmBase:
    """
    Tests for base tokenizers that do not have XGLM's special-token layout.

    Qwen3 is the motivating case: byte-level BPE with no unk and no bos, pad
    aliased to eos, and ids far outside any target vocab size. Passing those
    values to SentencePiece verbatim is a hard training crash, so this class
    covers the three ways the extraction has to diverge from the base.
    """

    def test_out_of_range_ids_are_reassigned(self):
        """Base ids beyond vocab_size are replaced with positional ids from 0."""
        config = _extract_special_tokens(make_qwen_stub(), vocab_size=32768)

        emitted_ids = [
            config[f"{role}_id"]
            for role in ("unk", "bos", "eos", "pad")
            if config[f"{role}_id"] >= 0
        ]
        assert all(token_id < 32768 for token_id in emitted_ids)
        assert len(set(emitted_ids)) == len(emitted_ids)

    def test_aliased_pad_is_disabled_not_duplicated(self):
        """
        pad_token == eos_token must yield one piece, since SentencePiece rejects
        a duplicate meta piece ("<|endoftext|> is already defined").
        """
        config = _extract_special_tokens(make_qwen_stub(), vocab_size=32768)

        assert config["eos_piece"] == "<|endoftext|>"
        assert "pad_piece" not in config
        assert config["pad_id"] == -1

    def test_missing_unk_is_synthesized_and_missing_bos_disabled(self):
        """
        A byte-level base has no unk, but the trained non-byte tokenizer needs one;
        it has no real bos, and inventing one is worse than disabling it.
        """
        config = _extract_special_tokens(make_qwen_stub(), vocab_size=32768)

        assert config["unk_piece"] == "<unk>"
        assert config["unk_id"] >= 0
        assert "bos_piece" not in config
        assert config["bos_id"] == -1

    def test_aliased_pad_recovered_for_huggingface_wrapper(self):
        """
        The pad role is dropped for SentencePiece but re-attached on the wrapper,
        where HuggingFace resolves the shared string to the eos id.
        """
        stub = make_qwen_stub()
        config = _extract_special_tokens(stub, vocab_size=32768)
        vocab = {"<unk>", "<|endoftext|>", "▁a", "▁b"}

        resolved = _resolve_hf_special_tokens(stub, config, vocab)

        assert resolved["eos_token"] == "<|endoftext|>"
        assert resolved["pad_token"] == "<|endoftext|>"
        assert resolved["unk_token"] == "<unk>"
        assert resolved["bos_token"] is None

    def test_roles_absent_from_trained_vocab_resolve_to_none(self):
        """A piece missing from the trained vocab must not be registered."""
        stub = make_qwen_stub()
        config = _extract_special_tokens(stub, vocab_size=32768)

        resolved = _resolve_hf_special_tokens(stub, config, {"▁a", "▁b"})

        assert all(token is None for token in resolved.values())


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

        config = make_tokenizer_config(
            vocab_size=vocab_size,
            inherit_additional_special_tokens=True
        )
        tokenizer = train_new_tokenizer(
            config=config,
            jsonl_path=str(sample_jsonl_path),
            output_path=str(output_dir)
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

    def test_train_tokenizer_bpe_override(self, sample_jsonl_path, tmp_path):
        """
        Test the fresh-tokenizer BPE branch via tokenizer_algorithm override.

        XGLM's base algorithm is Unigram, so tokenizer_algorithm='bpe' forces the
        override path. This is a regression guard: the BPE branch was previously
        unreachable (PTEx is Unigram-only) and had a broken .from_file() call.

        We verify the produced backend uses a BPE model and shares the same
        SentencePiece pipeline (empty normalizer + Metaspace pre-tokenizer/decoder)
        as the Unigram branch.
        """
        vocab_size = 64
        output_dir = tmp_path / "tokenizer_bpe"

        config = make_tokenizer_config(
            vocab_size=vocab_size,
            tokenizer_algorithm="bpe",
        )
        tokenizer = train_new_tokenizer(
            config=config,
            jsonl_path=str(sample_jsonl_path),
            output_path=str(output_dir),
        )

        assert len(tokenizer) == vocab_size

        backend = tokenizer.backend_tokenizer
        assert type(backend.model).__name__ == "BPE"

        # Shared pipeline: matches the Unigram branch for comparability
        assert "Metaspace" in str(backend.pre_tokenizer)
        assert "Metaspace" in str(backend.decoder)

        # Tokenizer actually works and produces in-range ids
        test_text = "This is a test sentence."
        token_ids = tokenizer.encode(test_text)
        assert len(token_ids) > 0
        assert all(0 <= tid < vocab_size for tid in token_ids)

    def test_train_tokenizer_handles_newlines(self, sample_jsonl_path, tmp_path):
        """
        Test that a freshly-trained tokenizer encodes newlines as a real piece.

        SentencePiece strips '\n' as the training-file line delimiter, so a
        newline piece never emerges from the corpus on its own. We inject it as a
        user-defined symbol unconditionally; this guards that the newline
        survives the full pipeline (spm training -> HF backend) and does not fall
        back to UNK.
        """
        vocab_size = 64
        output_dir = tmp_path / "tokenizer_newline"

        config = make_tokenizer_config(vocab_size=vocab_size)
        tokenizer = train_new_tokenizer(
            config=config,
            jsonl_path=str(sample_jsonl_path),
            output_path=str(output_dir),
        )

        # The newline must be a single dedicated piece, not the unknown token
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        assert tokenizer.unk_token_id not in newline_ids
        assert len(newline_ids) >= 1

        # A newline embedded between words must be preserved, not dropped
        embedded_ids = tokenizer.encode("alpha\nbeta", add_special_tokens=False)
        assert any(
            "\n" in tokenizer.convert_ids_to_tokens(tid) for tid in embedded_ids
        )
        assert tokenizer.unk_token_id not in embedded_ids

    def test_train_tokenizer_without_additional_tokens(self, sample_jsonl_path, tmp_path):
        """
        Test training a tokenizer without additional special tokens.

        This mode maximizes vocabulary space for the target language.
        """
        vocab_size = 64
        output_dir = tmp_path / "tokenizer_without_additional"

        config = make_tokenizer_config(
            vocab_size=vocab_size,
            inherit_additional_special_tokens=False
        )
        tokenizer = train_new_tokenizer(
            config=config,
            jsonl_path=str(sample_jsonl_path),
            output_path=str(output_dir)
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

        config = make_tokenizer_config(
            vocab_size=vocab_size,
            inherit_additional_special_tokens=True
        )

        # First call: actually trains
        tokenizer1 = train_new_tokenizer(
            config=config,
            jsonl_path=str(sample_jsonl_path),
            output_path=str(output_dir)
        )

        # Get timestamp of created file
        tokenizer_file = output_dir / "tokenizer.json"
        mtime_before = tokenizer_file.stat().st_mtime

        # Second call: should load from cache
        tokenizer2 = train_new_tokenizer(
            config=config,
            jsonl_path=str(sample_jsonl_path),
            output_path=str(output_dir)
        )

        # Verify file wasn't modified (indicates it was loaded, not retrained)
        mtime_after = tokenizer_file.stat().st_mtime
        assert mtime_after == mtime_before

        # Both tokenizers should have same vocab size
        assert len(tokenizer1) == len(tokenizer2) == vocab_size

    def test_seed_tokenizer_caching(self, sample_jsonl_path, tmp_path):
        """
        Test that seed tokenizer is cached separately and reused across lambda values.

        Strategy:
        1. Train tokenizer with seeded vocab and lambda=0.5
        2. Train another with lambda=0.7 (same vocab_size, num_samples, multiplier)
        3. Verify seed tokenizer is in separate directory
        4. Verify seed tokenizer was NOT retrained for second lambda
        """
        # Use small vocab sizes that work with our 10-sentence test corpus
        # Note: Test corpus has 38 unique characters + 11 special tokens = 49 minimum
        vocab_size = 50
        num_samples = 10
        multiplier = 2.0  # Results in 100 tokens for seed tokenizer

        # Create directory structure like production:
        # tmp_path/tokenizers/test_lang/
        tokenizers_dir = tmp_path / "tokenizers" / "test_lang"
        tokenizers_dir.mkdir(parents=True)

        # First tokenizer with lambda=0.5
        config1 = make_tokenizer_config(
            vocab_size=vocab_size,
            num_samples=num_samples,
            use_seed_vocabulary=True,
            seed_lambda=0.5,
            seed_vocab_multiplier=multiplier,
            character_coverage=0.9995  # Reduce coverage to allow smaller vocab
        )
        output_dir_1 = tokenizers_dir / "xglm564m_focus-v50-s10_seeded-2.0x-lambda0.5"
        tokenizer1 = train_new_tokenizer(
            config=config1,
            jsonl_path=str(sample_jsonl_path),
            output_path=str(output_dir_1)
        )

        # Verify seed tokenizer directory exists at sibling level
        seed_dir = tokenizers_dir / "xglm564m_focus-v50-s10_seed-2.0x"
        assert seed_dir.exists(), f"Seed tokenizer should exist at {seed_dir}"
        assert (seed_dir / "spm.model").exists()

        # Get timestamp of seed tokenizer
        seed_model_file = seed_dir / "spm.model"
        seed_mtime_before = seed_model_file.stat().st_mtime

        # Second tokenizer with different lambda (should reuse seed tokenizer)
        config2 = make_tokenizer_config(
            vocab_size=vocab_size,
            num_samples=num_samples,
            use_seed_vocabulary=True,
            seed_lambda=0.7,  # Different lambda
            seed_vocab_multiplier=multiplier,
            character_coverage=0.9995
        )
        output_dir_2 = tokenizers_dir / "xglm564m_focus-v50-s10_seeded-2.0x-lambda0.7"
        tokenizer2 = train_new_tokenizer(
            config=config2,
            jsonl_path=str(sample_jsonl_path),
            output_path=str(output_dir_2)
        )

        # Verify seed tokenizer was NOT retrained (timestamp unchanged)
        seed_mtime_after = seed_model_file.stat().st_mtime
        assert seed_mtime_after == seed_mtime_before, \
            "Seed tokenizer should be reused, not retrained"

        # Verify both final tokenizers were created in separate directories
        assert (output_dir_1 / "tokenizer.json").exists()
        assert (output_dir_2 / "tokenizer.json").exists()

        # Verify both have correct vocab size
        assert len(tokenizer1) == vocab_size
        assert len(tokenizer2) == vocab_size


import os
from types import SimpleNamespace

import torch

from tokenizer_utils import (
    apply_focus_initialization,
    resolve_cached_embedding_paths,
    _sidecar_paths,
    FOCUS_EMBS_SUBDIR,
    LEGACY_INPUT_NAME,
    LEGACY_OUTPUT_NAME,
)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'')


class TestResolveCachedEmbeddingPaths:
    """Tests for the embedding sidecar resolver under default and 'any' policies."""

    def test_empty_dir_default_policy(self, tmp_path):
        assert resolve_cached_embedding_paths(str(tmp_path), 'abcd1234', None) is None

    def test_empty_dir_any_policy(self, tmp_path):
        assert resolve_cached_embedding_paths(str(tmp_path), 'abcd1234', 'any') is None

    def test_default_policy_hash_match(self, tmp_path):
        inp, outp, _meta = _sidecar_paths(str(tmp_path), 'abcd1234')
        _touch(inp)
        _touch(outp)
        result = resolve_cached_embedding_paths(str(tmp_path), 'abcd1234', None)
        assert result == (inp, outp)

    def test_default_policy_hash_match_no_output_sidecar(self, tmp_path):
        inp, outp, _meta = _sidecar_paths(str(tmp_path), 'abcd1234')
        _touch(inp)
        result = resolve_cached_embedding_paths(str(tmp_path), 'abcd1234', None)
        assert result == (inp, None)

    def test_default_policy_hash_miss(self, tmp_path):
        # A different hash's sidecar exists; default policy must NOT load it.
        other_inp, _, _ = _sidecar_paths(str(tmp_path), 'deadbeef')
        _touch(other_inp)
        assert resolve_cached_embedding_paths(str(tmp_path), 'abcd1234', None) is None

    def test_any_policy_loads_lone_new_sidecar(self, tmp_path):
        inp, outp, _ = _sidecar_paths(str(tmp_path), 'deadbeef')
        _touch(inp)
        _touch(outp)
        result = resolve_cached_embedding_paths(str(tmp_path), 'abcd1234', 'any')
        assert result == (inp, outp)

    def test_any_policy_loads_lone_legacy_file(self, tmp_path):
        legacy_in = tmp_path / LEGACY_INPUT_NAME
        legacy_out = tmp_path / LEGACY_OUTPUT_NAME
        _touch(str(legacy_in))
        _touch(str(legacy_out))
        result = resolve_cached_embedding_paths(str(tmp_path), 'abcd1234', 'any')
        assert result == (str(legacy_in), str(legacy_out))

    def test_any_policy_legacy_input_only(self, tmp_path):
        legacy_in = tmp_path / LEGACY_INPUT_NAME
        _touch(str(legacy_in))
        result = resolve_cached_embedding_paths(str(tmp_path), 'abcd1234', 'any')
        assert result == (str(legacy_in), None)

    def test_any_policy_ambiguous_new_plus_legacy(self, tmp_path):
        inp, _, _ = _sidecar_paths(str(tmp_path), 'deadbeef')
        _touch(inp)
        _touch(str(tmp_path / LEGACY_INPUT_NAME))
        with pytest.raises(ValueError, match='2 cached embedding sets'):
            resolve_cached_embedding_paths(str(tmp_path), 'abcd1234', 'any')

    def test_any_policy_ambiguous_two_hashed(self, tmp_path):
        a, _, _ = _sidecar_paths(str(tmp_path), 'aaaaaaaa')
        b, _, _ = _sidecar_paths(str(tmp_path), 'bbbbbbbb')
        _touch(a)
        _touch(b)
        with pytest.raises(ValueError, match='2 cached embedding sets'):
            resolve_cached_embedding_paths(str(tmp_path), 'aaaaaaaa', 'any')

    def test_explicit_hash_policy_present(self, tmp_path):
        inp, outp, _ = _sidecar_paths(str(tmp_path), 'feedface')
        _touch(inp)
        _touch(outp)
        # Request a specific hash; the active emb_hash arg is ignored when a
        # specific reuse_policy is set.
        result = resolve_cached_embedding_paths(str(tmp_path), 'unrelated', 'feedface')
        assert result == (inp, outp)

    def test_explicit_hash_policy_absent(self, tmp_path):
        with pytest.raises(ValueError, match="reuse_embeddings='feedface'"):
            resolve_cached_embedding_paths(str(tmp_path), 'unrelated', 'feedface')

    def test_cache_dir_none(self):
        assert resolve_cached_embedding_paths(None, 'abcd1234', None) is None
        assert resolve_cached_embedding_paths(None, 'abcd1234', 'any') is None


class _StubModel:
    """Minimal source-model stub for apply_focus_initialization.

    Provides config + get_input_embeddings/get_output_embeddings.
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 4, tied: bool = True):
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self._inp = torch.nn.Embedding(vocab_size, hidden_dim)
        if not tied:
            self._out = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self._inp

    def get_output_embeddings(self):
        return self._out  # only reached when not tied


class _StubTokenizer:
    """Minimal HF-tokenizer-like object exposing only the methods that
    apply_focus_initialization touches: __len__, convert_ids_to_tokens,
    convert_tokens_to_ids. Used to construct synthetic target vocabularies
    that are either strict subsets of a real source tokenizer (no novel
    tokens) or include explicit novel strings."""

    def __init__(self, tokens):
        self._tokens = list(tokens)
        self._id_to_tok = dict(enumerate(self._tokens))
        self._tok_to_id = {t: i for i, t in enumerate(self._tokens)}

    def __len__(self):
        return len(self._tokens)

    def convert_ids_to_tokens(self, i):
        return self._id_to_tok[i]

    def convert_tokens_to_ids(self, t):
        return self._tok_to_id.get(t, 0)


def _pruned_target_from_source(source_tokenizer, n: int = 64):
    """Build a target tokenizer whose vocab is a strict subset of source's.

    Skips ID 0 (typically a special token whose exact string differs across
    fast/slow XGLM tokenizer variants) and takes the next n IDs to keep the
    test independent of XGLM's special-token layout.
    """
    tokens = [source_tokenizer.convert_ids_to_tokens(i) for i in range(1, 1 + n)]
    return _StubTokenizer(tokens)


class TestApplyFocusInitializationCache:
    """Sidecar write + cache-hit / cache-miss behavior of apply_focus_initialization.

    Uses real XGLM as source and either a pruned XGLM subset (no novel tokens)
    or a stub with synthetic novel strings (novel tokens) as the target.
    """

    def test_sidecar_write_layout(self, tmp_path, base_tokenizer):
        target = _pruned_target_from_source(base_tokenizer, n=32)
        cache_dir = str(tmp_path / "tok")
        os.makedirs(cache_dir, exist_ok=True)
        emb_hash = 'cafebabe'
        meta = {'embedding_hash': emb_hash, 'num_samples': 100}

        model = _StubModel(vocab_size=len(base_tokenizer))
        new_in, new_out = apply_focus_initialization(
            source_model=model,
            source_tokenizer=base_tokenizer,
            target_tokenizer=target,
            training_data_path=None,
            cache_dir=cache_dir,
            embedding_hash=emb_hash,
            embedding_meta=meta,
        )
        assert new_in.shape == (len(target), 4)
        assert new_out is None  # tied

        inp, outp, meta_path = _sidecar_paths(cache_dir, emb_hash)
        assert os.path.exists(inp), f"missing {inp}"
        assert not os.path.exists(outp), "no output sidecar for tied model"
        assert os.path.exists(meta_path)

        import yaml as _yaml
        with open(meta_path) as f:
            loaded_meta = _yaml.safe_load(f)
        assert loaded_meta == meta

    def test_cache_hit_skips_compute_with_none_training_data(self, tmp_path, base_tokenizer):
        # Use a target with synthetic novel tokens to prove that on a cache hit
        # the function returns *without* triggering the novel-token validation
        # (i.e. caching genuinely short-circuits the entire downstream path).
        target = _StubTokenizer(['<<<NOVEL_TOK_A>>>', '<<<NOVEL_TOK_B>>>'])
        cache_dir = str(tmp_path / "tok")
        emb_hash = 'cafe1111'
        inp, _outp, _meta = _sidecar_paths(cache_dir, emb_hash)
        os.makedirs(os.path.dirname(inp), exist_ok=True)

        sentinel = torch.full((len(target), 4), 7.0)
        torch.save(sentinel, inp)

        model = _StubModel(vocab_size=len(base_tokenizer))
        new_in, new_out = apply_focus_initialization(
            source_model=model,
            source_tokenizer=base_tokenizer,
            target_tokenizer=target,
            training_data_path=None,
            cache_dir=cache_dir,
            embedding_hash=emb_hash,
        )
        assert torch.equal(new_in, sentinel)
        assert new_out is None

    def test_cache_miss_no_novel_tokens_runs_without_training_data(
        self, tmp_path, base_tokenizer
    ):
        """No cache hit + target ⊂ source → direct-copy path succeeds without
        a JSONL. Build_vocab_adapted_model relies on this, and so does the
        JSONL gate in _initialize_focus_model."""
        target = _pruned_target_from_source(base_tokenizer, n=64)
        cache_dir = str(tmp_path / "tok")
        os.makedirs(cache_dir, exist_ok=True)

        # Sanity: every target token really is in source.
        source_toks = {
            base_tokenizer.convert_ids_to_tokens(i)
            for i in range(len(base_tokenizer))
        }
        assert all(target.convert_ids_to_tokens(i) in source_toks
                   for i in range(len(target)))

        model = _StubModel(vocab_size=len(base_tokenizer))
        new_in, _new_out = apply_focus_initialization(
            source_model=model,
            source_tokenizer=base_tokenizer,
            target_tokenizer=target,
            training_data_path=None,
            cache_dir=cache_dir,
            embedding_hash='nope0000',
            embedding_meta={'embedding_hash': 'nope0000'},
        )
        assert new_in.shape == (len(target), 4)
        inp, _outp, _meta = _sidecar_paths(cache_dir, 'nope0000')
        assert os.path.exists(inp)

    def test_cache_miss_novel_tokens_requires_training_data(
        self, tmp_path, base_tokenizer
    ):
        """Novel tokens + cache miss + training_data_path=None → ValueError,
        raised before the FOCUS / fastText path runs."""
        # Synthetic tokens guaranteed not to appear in XGLM's vocab.
        target = _StubTokenizer(['<<<NOVEL_TOK_A>>>', '<<<NOVEL_TOK_B>>>'])
        cache_dir = str(tmp_path / "tok")
        os.makedirs(cache_dir, exist_ok=True)

        model = _StubModel(vocab_size=len(base_tokenizer))
        with pytest.raises(ValueError, match='training_data_path is required'):
            apply_focus_initialization(
                source_model=model,
                source_tokenizer=base_tokenizer,
                target_tokenizer=target,
                training_data_path=None,
                cache_dir=cache_dir,
                embedding_hash='nope0001',
            )
