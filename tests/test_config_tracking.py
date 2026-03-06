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


class TestTokenizerConfigInitModelId:
    """Test that init_model_id overrides derived model shortname in all suffix methods."""

    def _make_args(self, init_model_id=None):
        args = {
            'hf_model': 'facebook/xglm-1.7B',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 32768,
                'num_samples': 5000000,
                'use_seed_vocabulary': True,
                'seed_vocab_multiplier': 2.0,
                'seed_lambda': 0.5,
            },
            'seed': 42,
        }
        if init_model_id is not None:
            args['init_model_id'] = init_model_id
        return OmegaConf.create(args)

    def test_focus_suffix_without_init_model_id(self):
        """Without init_model_id, derives shortname from hf_model."""
        tok_config = TokenizerConfig.from_args(self._make_args())
        assert tok_config.focus_suffix().startswith("xglm17b_")

    def test_focus_suffix_with_init_model_id(self):
        """With init_model_id, uses it instead of derived shortname."""
        tok_config = TokenizerConfig.from_args(self._make_args(init_model_id="xglm1b"))
        assert tok_config.focus_suffix().startswith("xglm1b_")
        assert "xglm17b" not in tok_config.focus_suffix()

    def test_seed_tokenizer_suffix_with_init_model_id(self):
        """seed_tokenizer_suffix should also use init_model_id."""
        tok_config = TokenizerConfig.from_args(self._make_args(init_model_id="xglm1b"))
        suffix = tok_config.seed_tokenizer_suffix()
        assert suffix.startswith("xglm1b_")
        assert "xglm17b" not in suffix

    def test_cache_dir_with_init_model_id(self):
        """cache_dir should use init_model_id in the directory name."""
        tok_config = TokenizerConfig.from_args(self._make_args(init_model_id="xglm1b"))
        path = tok_config.cache_dir("old_germanic")
        assert "xglm1b_" in path
        assert "xglm17b" not in path

    def test_all_suffixes_consistent(self):
        """focus_suffix, seed_tokenizer_suffix, and cache_dir should all use the same model id."""
        tok_config = TokenizerConfig.from_args(self._make_args(init_model_id="v81"))
        assert tok_config.focus_suffix().startswith("v81_")
        assert tok_config.seed_tokenizer_suffix().startswith("v81_")
        assert "/v81_" in tok_config.cache_dir("got")


class TestTokenizerConfigSeedScoreMode:
    """Test that seed_score_mode is reflected in focus_suffix."""

    def _make_args(self, score_mode="count"):
        return OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 50000,
                'use_seed_vocabulary': True,
                'seed_vocab_multiplier': 5.0,
                'seed_lambda': 0.5,
                'seed_score_mode': score_mode,
            },
            'seed': 42,
        })

    def test_default_count_mode_not_in_suffix(self):
        """Default 'count' mode should not appear in the suffix."""
        tok_config = TokenizerConfig.from_args(self._make_args("count"))
        assert "count" not in tok_config.focus_suffix()
        assert "charlength" not in tok_config.focus_suffix()

    def test_charlength_mode_in_suffix(self):
        """Non-default 'charlength' mode should appear in the suffix."""
        tok_config = TokenizerConfig.from_args(self._make_args("charlength"))
        assert tok_config.focus_suffix().endswith("-charlength")

    def test_score_mode_not_in_seed_tokenizer_suffix(self):
        """Seed tokenizer suffix should NOT include score_mode (shared across modes)."""
        tok_config = TokenizerConfig.from_args(self._make_args("charlength"))
        assert "charlength" not in tok_config.seed_tokenizer_suffix()


class TestSeedTokenizerSuffix:
    """Test seed_tokenizer_suffix method directly."""

    def test_basic_suffix(self):
        """Test basic seed tokenizer suffix format."""
        tok_config = TokenizerConfig(
            hf_model='facebook/xglm-564M',
            vocab_size=16384,
            num_samples=200000,
            character_coverage=1.0,
            inherit_additional_special_tokens=True,
            use_seed_vocabulary=True,
            seed_vocab_multiplier=5.0,
            seed_lambda=0.5,
            seed_min_frequency=1,
            seed_round_mode='round',
            seed_score_mode='count',
            fasttext_model_min_count=4,
            seed=42,
        )
        assert tok_config.seed_tokenizer_suffix() == "xglm564m_focus-v16k-s200k_seed-5.0x"

    def test_suffix_with_different_multiplier(self):
        """Test that multiplier is reflected in seed tokenizer suffix."""
        tok_config = TokenizerConfig(
            hf_model='facebook/xglm-564M',
            vocab_size=32768,
            num_samples=5000000,
            character_coverage=1.0,
            inherit_additional_special_tokens=True,
            use_seed_vocabulary=True,
            seed_vocab_multiplier=2.0,
            seed_lambda=0.5,
            seed_min_frequency=1,
            seed_round_mode='round',
            seed_score_mode='count',
            fasttext_model_min_count=4,
            seed=42,
        )
        assert tok_config.seed_tokenizer_suffix() == "xglm564m_focus-v32k-s5m_seed-2.0x"


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
            'hf_model': 'facebook/xglm-564M',
            'seed': 42
        })

        config = extract_tokenized_config(args)
        assert config['max_length'] == 512
        assert config['dev_size'] == 0.1
        assert 'dataset' in config
        assert config['dataset']['type'] == 'oscar'
        assert 'tokenizer' not in config
        assert config['hf_model'] == 'facebook/xglm-564M'

    def test_extract_tokenized_config_no_focus_with_init_model_id(self):
        """When FOCUS is disabled and init_model_id is set, both hf_model and init_model_id are tracked."""
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
            'hf_model': '/local/path/to/checkpoint',
            'init_model_id': 'v81',
            'seed': 42
        })

        config = extract_tokenized_config(args)
        assert config['hf_model'] == '/local/path/to/checkpoint'
        assert config['init_model_id'] == 'v81'
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
