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
    extract_tokenizer_config,
    extract_tokenized_config,
    save_config,
    load_config,
    check_config_match,
    _dict_diff
)


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


class TestExtractTokenizerConfig:
    """Test extraction of tokenizer-relevant config parameters."""

    def test_extract_with_focus_disabled(self):
        args = OmegaConf.create({
            'focus': {'enabled': False}
        })

        config = extract_tokenizer_config(args)
        assert config is None

    def test_extract_basic_focus_config(self):
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

        config = extract_tokenizer_config(args)
        assert config is not None
        assert config['vocab_size'] == 16384
        assert config['num_samples'] == 100000
        assert config['hf_model'] == 'facebook/xglm-564M'
        assert config['seed'] == 42
        assert 'train_dataset_cache' in config

    def test_extract_focus_with_separate_dataset(self):
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

        config = extract_tokenizer_config(args)
        assert config is not None
        assert 'focus_dataset' in config
        assert 'train_dataset_cache' not in config
        assert config['focus_dataset']['type'] == 'plaintext'


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
