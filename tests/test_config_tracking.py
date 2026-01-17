"""
Unit tests for configuration tracking functionality.

Tests that config mismatches are properly detected and that configs
are correctly saved/loaded alongside artifacts.
"""

import os
import tempfile
import pytest
from omegaconf import OmegaConf

from src.config_utils import (
    extract_dataset_config,
    extract_tokenized_config,
    save_config,
    load_config,
    check_config_match,
    _dict_diff
)
from src.artifact_configs import TokenizerConfig


class TestDictDiff:
    """Test the recursive dictionary difference function."""

    def test_identical_dicts(self):
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'a': 1, 'b': 2}
        diffs = _dict_diff(dict1, dict2)
        assert len(diffs) == 0

    def test_different_values(self):
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'a': 1, 'b': 3}
        diffs = _dict_diff(dict1, dict2)
        assert len(diffs) == 1
        assert 'b: 2 (cached) != 3 (current)' in diffs[0]

    def test_missing_keys_in_dict2(self):
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'a': 1}
        diffs = _dict_diff(dict1, dict2)
        assert len(diffs) == 1
        assert 'b' in diffs[0]

    def test_extra_keys_in_dict2(self):
        dict1 = {'a': 1}
        dict2 = {'a': 1, 'b': 2}
        diffs = _dict_diff(dict1, dict2)
        assert len(diffs) == 1
        assert 'b' in diffs[0]

    def test_nested_dicts(self):
        dict1 = {'a': {'x': 1, 'y': 2}}
        dict2 = {'a': {'x': 1, 'y': 3}}
        diffs = _dict_diff(dict1, dict2)
        assert len(diffs) == 1
        assert 'a.y: 2 (cached) != 3 (current)' in diffs[0]


class TestConfigSaveLoad:
    """Test saving and loading configuration files."""

    def test_save_and_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'key1': 'value1', 'key2': 42}
            config_path = os.path.join(tmpdir, 'test_config.yaml')

            save_config(config, config_path)
            assert os.path.exists(config_path)

            loaded = load_config(config_path)
            assert loaded == config

    def test_load_nonexistent_config(self):
        result = load_config('/nonexistent/path/config.yaml')
        assert result is None

    def test_save_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'test': 'data'}
            config_path = os.path.join(tmpdir, 'nested', 'dir', 'config.yaml')

            save_config(config, config_path)
            assert os.path.exists(config_path)


class TestConfigMatch:
    """Test configuration matching and error handling."""

    def test_match_with_no_cached_config(self):
        """No cached config should return True (no error)."""
        current_config = {'a': 1}
        result = check_config_match(None, current_config, "Test Artifact")
        assert result is True

    def test_match_with_identical_configs(self):
        """Identical configs should return True."""
        config = {'a': 1, 'b': 2}
        result = check_config_match(config, config, "Test Artifact")
        assert result is True

    def test_mismatch_raises_error(self):
        """Different configs should raise ValueError by default."""
        cached = {'a': 1}
        current = {'a': 2}

        with pytest.raises(ValueError) as exc_info:
            check_config_match(cached, current, "Test Artifact")

        assert "CONFIG MISMATCH" in str(exc_info.value)
        assert "Test Artifact" in str(exc_info.value)
        assert "a: 1 (cached) != 2 (current)" in str(exc_info.value)

    def test_mismatch_warning_mode(self):
        """With error_on_mismatch=False, should return False instead of error."""
        cached = {'a': 1}
        current = {'a': 2}

        result = check_config_match(cached, current, "Test", error_on_mismatch=False)
        assert result is False


class TestExtractDatasetConfig:
    """Test extraction of dataset-relevant config parameters."""

    def test_extract_oscar_config(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy_oscar'
            },
            'seed': 42
        })

        config = extract_dataset_config(args)
        assert config['type'] == 'oscar'
        assert config['language'] == 'hy'
        assert config['seed'] == 42

    def test_extract_plaintext_config(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'plaintext',
                'path': '/path/to/data.txt',
                'cache_dir': 'data/my_corpus'
            },
            'seed': 42
        })

        config = extract_dataset_config(args)
        assert config['type'] == 'plaintext'
        assert config['path'] == '/path/to/data.txt'
        assert config['seed'] == 42

    def test_extract_multinomial_config(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'multinomial',
                'sources': [{'type': 'oscar', 'language': 'hy'}],
                'alpha': 0.7,
                'total_samples': 1000000,
                'cache_dir': 'data/multi'
            },
            'training': {
                'dev_size': 0.1
            },
            'seed': 42
        })

        config = extract_dataset_config(args)
        assert config['type'] == 'multinomial'
        assert config['alpha'] == 0.7
        assert config['total_samples'] == 1000000
        assert config['dev_size'] == 0.1
        assert config['seed'] == 42


class TestTokenizerConfig:
    """Test TokenizerConfig extraction and dict conversion."""

    def test_from_args_with_focus_disabled(self):
        args = OmegaConf.create({
            'focus': {'enabled': False}
        })

        tok_config = TokenizerConfig.from_args(args)
        assert tok_config is None

    def test_from_args_basic_focus_config(self):
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 100000
            },
            'seed': 42
        })

        tok_config = TokenizerConfig.from_args(args)
        assert tok_config is not None
        assert tok_config.vocab_size == 16384
        assert tok_config.num_samples == 100000
        assert tok_config.hf_model == 'facebook/xglm-564M'
        assert tok_config.seed == 42
        assert tok_config.train_dataset_cache == 'data/test'

    def test_from_args_focus_with_separate_dataset(self):
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 100000,
                'dataset': {
                    'type': 'plaintext',
                    'path': '/other/data.txt'
                }
            },
            'seed': 42
        })

        tok_config = TokenizerConfig.from_args(args)
        assert tok_config is not None
        assert tok_config.focus_dataset is not None
        assert tok_config.train_dataset_cache is None
        assert tok_config.focus_dataset['type'] == 'plaintext'

    def test_to_dict_roundtrip(self):
        """Test that to_dict produces expected keys for config tracking."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 100000
            },
            'seed': 42
        })

        tok_config = TokenizerConfig.from_args(args)
        config_dict = tok_config.to_dict()
        assert config_dict['vocab_size'] == 16384
        assert config_dict['hf_model'] == 'facebook/xglm-564M'
        assert config_dict['seed'] == 42


class TestTokenizerConfigFocusSuffix:
    """
    Test suite for TokenizerConfig.focus_suffix() method.

    This method builds the path suffix that encodes FOCUS tokenizer parameters
    to avoid cache collisions when changing vocabulary settings.
    """

    def test_basic_suffix_no_optional_flags(self):
        """Test basic suffix with just model, vocab, and samples."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 50000,
                'num_samples': 100000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': False
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.focus_suffix() == "xglm564m_focus-v50k-s100k"

    def test_suffix_with_no_additional_flag(self):
        """Test suffix when not inheriting additional special tokens."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 32768,
                'num_samples': 1000000,
                'inherit_additional_special_tokens': False,
                'use_seed_vocabulary': False
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.focus_suffix() == "xglm564m_focus-v32k-s1m_no-additional"

    def test_suffix_with_seed_vocabulary_default_params(self):
        """Test suffix with seed vocabulary using default parameters."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 50000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': True,
                'seed_min_frequency': 1,
                'seed_lambda': 0.5,
                'seed_vocab_multiplier': 5.0
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.focus_suffix() == "xglm564m_focus-v16k-s50k_seeded-5.0x-lambda0.5"

    def test_suffix_with_seed_vocabulary_custom_min_frequency(self):
        """Test suffix with non-default seed min frequency."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 50000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': True,
                'seed_min_frequency': 5,
                'seed_lambda': 0.5,
                'seed_vocab_multiplier': 5.0
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.focus_suffix() == "xglm564m_focus-v16k-s50k_seeded-5.0x-lambda0.5-min5"

    def test_suffix_with_seed_lambda(self):
        """Test suffix with non-default seed lambda."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 50000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': True,
                'seed_min_frequency': 1,
                'seed_lambda': 0.7,
                'seed_vocab_multiplier': 5.0
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.focus_suffix() == "xglm564m_focus-v16k-s50k_seeded-5.0x-lambda0.7"

    def test_suffix_with_all_flags(self):
        """Test suffix with all optional flags enabled."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 32768,
                'num_samples': 1000000,
                'inherit_additional_special_tokens': False,
                'use_seed_vocabulary': True,
                'seed_min_frequency': 10,
                'seed_lambda': 0.7,
                'seed_vocab_multiplier': 5.0
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.focus_suffix() == "xglm564m_focus-v32k-s1m_no-additional_seeded-5.0x-lambda0.7-min10"

    def test_suffix_respects_number_formatting(self):
        """Test that vocab and sample sizes use format_number() correctly."""
        args = OmegaConf.create({
            'hf_model': 'gpt2',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 128000,
                'num_samples': 5000000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': False
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.focus_suffix() == "gpt2_focus-v128k-s5m"

    def test_suffix_with_different_model(self):
        """Test suffix generation with different base model."""
        args = OmegaConf.create({
            'hf_model': 'meta-llama/Llama-2-7b',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 50000,
                'num_samples': 100000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': False
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.focus_suffix() == "llama27b_focus-v50k-s100k"


class TestExtractTokenizedConfig:
    """Test extraction of tokenized dataset config parameters."""

    def test_extract_tokenized_config_no_focus(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy'
            },
            'training': {
                'max_length': 512,
                'dev_size': 0.1
            },
            'focus': {'enabled': False},
            'seed': 42
        })

        config = extract_tokenized_config(args)
        assert config['max_length'] == 512
        assert config['dev_size'] == 0.1
        assert 'dataset' in config
        assert config['dataset']['type'] == 'oscar'
        assert 'tokenizer' not in config

    def test_extract_tokenized_config_with_focus(self):
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy'
            },
            'training': {
                'max_length': 1024,
                'dev_size': 0.2
            },
            'focus': {
                'enabled': True,
                'vocab_size': 32768,
                'num_samples': 500000
            },
            'seed': 42
        })

        config = extract_tokenized_config(args)
        assert config['max_length'] == 1024
        assert config['dev_size'] == 0.2
        assert 'dataset' in config
        assert 'tokenizer' in config
        assert config['tokenizer']['vocab_size'] == 32768
