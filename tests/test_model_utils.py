"""Tests for model_utils module."""

import pytest
from omegaconf import OmegaConf

from lapt.__main__ import _validate_init_model_id
from lapt.model_utils import (
    format_number,
    get_init_model_identifier,
    get_model_shortname,
    get_tokenized_path,
    is_local_model_path,
)


class TestFormatNumber:
    """
    Test suite for the format_number() utility function.
    """

    def test_format_thousands(self):
        """
        Test that numbers >= 1000 are formatted with 'k' suffix.
        """
        assert format_number(50000) == "50k"
        assert format_number(1000) == "1k"
        assert format_number(1500) == "1k"  # Integer division

    def test_format_millions(self):
        """Test that numbers >= 1,000,000 are formatted with 'm' suffix."""
        assert format_number(1000000) == "1m"
        assert format_number(5000000) == "5m"
        assert format_number(1500000) == "1m"  # Integer division

    def test_format_small_numbers(self):
        """Test that numbers < 1000 are returned as strings unchanged."""
        assert format_number(999) == "999"
        assert format_number(100) == "100"
        assert format_number(1) == "1"
        assert format_number(0) == "0"


class TestGetModelShortname:
    """
    Test suite for get_model_shortname() utility function.

    This function extracts a compact identifier from HuggingFace model names
    for use in directory paths.
    """

    def test_extract_from_org_path(self):
        """Test extracting model name from org/model format."""
        assert get_model_shortname("facebook/xglm-564M") == "xglm564m"
        assert get_model_shortname("meta-llama/Llama-2-7b") == "llama27b"

    def test_simple_model_name(self):
        """Test model names without organization prefix."""
        assert get_model_shortname("gpt2") == "gpt2"
        assert get_model_shortname("bert-base-uncased") == "bertbaseuncased"

    def test_removes_special_characters(self):
        """Test that dashes and dots are removed."""
        assert get_model_shortname("model-name-v2.1") == "modelnamev21"
        assert get_model_shortname("test.model.v1-beta") == "testmodelv1beta"

    def test_lowercases_output(self):
        """Test that output is lowercased."""
        assert get_model_shortname("GPT-2-XL") == "gpt2xl"
        assert get_model_shortname("BERT") == "bert"

    def test_complex_path(self):
        """Test with complex model path."""
        assert get_model_shortname("organization/sub-org/model-v1.2.3") == "modelv123"


class TestIsLocalModelPath:
    """Test detection of local model paths vs HuggingFace model names."""

    def test_hf_model_name(self):
        assert not is_local_model_path("facebook/xglm-564M")

    def test_absolute_path(self):
        assert is_local_model_path("/scratch/user/models/checkpoint")

    def test_relative_path(self):
        assert is_local_model_path("./models/checkpoint")

    def test_home_path(self):
        assert is_local_model_path("~/models/checkpoint")


class TestGetInitModelIdentifier:
    """Test model identifier resolution with and without init_model_id."""

    def test_uses_init_model_id_when_provided(self):
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-1.7B',
            'init_model_id': 'xglm1b',
        })
        assert get_init_model_identifier(args) == "xglm1b"

    def test_derives_from_hf_model_when_no_init_model_id(self):
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
        })
        assert get_init_model_identifier(args) == "xglm564m"


class TestValidateInitModelId:
    """Test validation that init_model_id is required for local model paths."""

    def test_passes_with_hf_model_name(self):
        """No error for standard HuggingFace model names."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
        })
        _validate_init_model_id(args)

    def test_passes_with_local_path_and_init_model_id(self):
        """No error when init_model_id is provided for a local path."""
        args = OmegaConf.create({
            'hf_model': '/scratch/user/models/v81/best-checkpoint',
            'init_model_id': 'v81',
        })
        _validate_init_model_id(args)

    def test_raises_for_local_path_without_init_model_id(self):
        """Error when hf_model is a local path but init_model_id is missing."""
        args = OmegaConf.create({
            'hf_model': '/scratch/user/models/checkpoint',
        })
        with pytest.raises(ValueError, match="init_model_id is required"):
            _validate_init_model_id(args)


class TestGetTokenizedPath:
    """Test tokenized dataset path generation."""

    def test_no_focus(self):
        """Without FOCUS, path includes model shortname."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {'enabled': False},
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert path == "data/test/tokenized_xglm564m"

    def test_no_focus_with_init_model_id(self):
        """Without FOCUS but with init_model_id, uses init_model_id."""
        args = OmegaConf.create({
            'hf_model': '/local/checkpoint',
            'init_model_id': 'v81',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {'enabled': False},
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert path == "data/test/tokenized_v81"

    def test_with_focus(self):
        """With FOCUS, path includes full focus suffix."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 100000,
            },
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert path == "data/test/tokenized_xglm564m_focus-v16k-s100k"

    def test_with_focus_and_init_model_id(self):
        """With FOCUS and init_model_id, uses init_model_id in path."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-1.7B',
            'init_model_id': 'xglm1b',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 32768,
                'num_samples': 5000000,
            },
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert path == "data/test/tokenized_xglm1b_focus-v32k-s5m"

    def test_no_redundant_focus_prefix(self):
        """Path should not contain 'tokenized_focus_' (old redundant format)."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 100000,
            },
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert "tokenized_focus_" not in path


import os
from types import SimpleNamespace

import torch

import lapt.model_utils as model_utils_mod
from lapt.artifact_configs import TokenizerConfig, focus_embedding_hash
from lapt.tokenizer_utils import _sidecar_paths


def _focus_model_args(cache_dir_base: str, language: str = "hy"):
    """Minimal args for _initialize_focus_model in a tmp working dir."""
    return OmegaConf.create({
        'hf_model': 'facebook/xglm-564M',
        'dataset': {
            'type': 'oscar',
            'language': language,
            'cache_dir': cache_dir_base,
            'dev_size': 0.1,
        },
        'training': {'max_length': 512},
        'focus': {
            'enabled': True,
            'tokenizer_path': None,
            'vocab_size': 16384,
            'num_samples': 100_000,
            'inherit_additional_special_tokens': True,
            'character_coverage': 1.0,
            'fasttext_model_min_count': 4,
            'reuse_embeddings': None,
        },
        'seed': 1,
    })


class _FakeEmbed(torch.nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(vocab_size, hidden))
        self.padding_idx = 0


class _FakeModel:
    def __init__(self, vocab_size=10, hidden=4):
        self.config = SimpleNamespace(
            tie_word_embeddings=True,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, vocab_size=vocab_size,
        )
        # The generation config is a separate object that the FOCUS init syncs
        # alongside model.config; without it the sync raises AttributeError.
        self.generation_config = SimpleNamespace(
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        self._embed = _FakeEmbed(vocab_size, hidden)

    def get_input_embeddings(self): return self._embed
    def get_output_embeddings(self): return self._embed
    def resize_token_embeddings(self, n): pass
    def tie_weights(self): pass


class _FakeTokenizer:
    def __init__(self, n=10):
        self._n = n
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def __len__(self): return self._n


class TestInitializeFocusModelJsonlGating:
    """Verify _initialize_focus_model only materializes the FOCUS JSONL when
    needed (i.e., not when both tokenizer and per-mix embeddings are cached)."""

    def _common_patches(self, monkeypatch, jsonl_calls, focus_calls):
        monkeypatch.setattr(
            model_utils_mod, 'prepare_focus_training_data',
            lambda **kw: (jsonl_calls.append(kw) or '/tmp/fake.jsonl'),
        )

        class _FakeTokenizerArtifact:
            """Stands in for TokenizerArtifact: same exists()/path/resolve()
            surface _initialize_focus_model relies on, but exists() checks for
            tokenizer.json specifically (matching what these tests set up)
            rather than just the cache directory, and resolve() never trains."""

            def __init__(self, language, tokenizer_config, jsonl_path=None, root="tokenizers"):
                self.path = tokenizer_config.cache_dir(language)
                self.jsonl_path = jsonl_path

            def exists(self):
                return os.path.exists(os.path.join(self.path, "tokenizer.json"))

            def resolve(self):
                return _FakeTokenizer()

        monkeypatch.setattr(model_utils_mod, 'TokenizerArtifact', _FakeTokenizerArtifact)
        def _apply(**kw):
            focus_calls.append(kw)
            n = len(kw['target_tokenizer'])
            return torch.zeros(n, 4), None
        monkeypatch.setattr(model_utils_mod, 'apply_focus_initialization', _apply)

        class _CfgStub:
            @staticmethod
            def from_pretrained(name): return SimpleNamespace()
        class _ModelStub:
            @staticmethod
            def from_pretrained(name, config=None): return _FakeModel()
        class _TokStub:
            @staticmethod
            def from_pretrained(name, **kw): return _FakeTokenizer()
        monkeypatch.setattr(model_utils_mod, 'AutoConfig', _CfgStub)
        monkeypatch.setattr(model_utils_mod, 'AutoModelForCausalLM', _ModelStub)
        monkeypatch.setattr(model_utils_mod, 'AutoTokenizer', _TokStub)

    def test_skips_jsonl_when_both_caches_hit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = _focus_model_args(cache_dir_base=str(tmp_path / 'data'))

        tok_cfg = TokenizerConfig.from_args(args)
        tok_dir = tok_cfg.cache_dir(args.dataset.language)
        os.makedirs(tok_dir, exist_ok=True)
        with open(os.path.join(tok_dir, 'tokenizer.json'), 'w') as f:
            f.write('{}')

        emb_hash = focus_embedding_hash(args)
        inp, _outp, _meta = _sidecar_paths(tok_dir, emb_hash)
        os.makedirs(os.path.dirname(inp), exist_ok=True)
        torch.save(torch.zeros(10, 4), inp)

        jsonl_calls, focus_calls = [], []
        self._common_patches(monkeypatch, jsonl_calls, focus_calls)

        model_utils_mod._initialize_focus_model(args)
        assert jsonl_calls == [], (
            "prepare_focus_training_data should NOT be called when both "
            f"tokenizer and embeddings are cached; got {len(jsonl_calls)} call(s)."
        )
        assert len(focus_calls) == 1
        assert focus_calls[0]['training_data_path'] is None

    def test_calls_jsonl_when_embeddings_miss(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = _focus_model_args(cache_dir_base=str(tmp_path / 'data'))

        tok_cfg = TokenizerConfig.from_args(args)
        tok_dir = tok_cfg.cache_dir(args.dataset.language)
        os.makedirs(tok_dir, exist_ok=True)
        with open(os.path.join(tok_dir, 'tokenizer.json'), 'w') as f:
            f.write('{}')

        jsonl_calls, focus_calls = [], []
        self._common_patches(monkeypatch, jsonl_calls, focus_calls)

        model_utils_mod._initialize_focus_model(args)
        assert len(jsonl_calls) == 1, (
            f"prepare_focus_training_data should be called exactly once when "
            f"embeddings are missing; got {len(jsonl_calls)} call(s)."
        )
        assert focus_calls[0]['training_data_path'] == '/tmp/fake.jsonl'
