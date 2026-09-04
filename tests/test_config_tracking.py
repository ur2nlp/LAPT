"""
Unit tests for configuration tracking functionality.

Tests that config mismatches are properly detected and that configs
are correctly saved/loaded alongside artifacts.
"""

import os
import tempfile

import pytest
from omegaconf import OmegaConf

from lapt.artifact_configs import (
    DatasetConfig,
    TokenizedDatasetConfig,
    TokenizerConfig,
    dict_diff,
    focus_embedding_hash,
    multinomial_mix_slug,
)


class TestDictDiff:
    """Test the recursive dictionary difference function."""

    def test_identical_dicts(self):
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'a': 1, 'b': 2}
        diffs = dict_diff(dict1, dict2)
        assert len(diffs) == 0

    def test_different_values(self):
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'a': 1, 'b': 3}
        diffs = dict_diff(dict1, dict2)
        assert len(diffs) == 1
        assert 'b: 2 (cached) != 3 (current)' in diffs[0]

    def test_missing_keys_in_dict2(self):
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'a': 1}
        diffs = dict_diff(dict1, dict2)
        assert len(diffs) == 1
        assert 'b' in diffs[0]

    def test_extra_keys_in_dict2(self):
        dict1 = {'a': 1}
        dict2 = {'a': 1, 'b': 2}
        diffs = dict_diff(dict1, dict2)
        assert len(diffs) == 1
        assert 'b' in diffs[0]

    def test_nested_dicts(self):
        dict1 = {'a': {'x': 1, 'y': 2}}
        dict2 = {'a': {'x': 1, 'y': 3}}
        diffs = dict_diff(dict1, dict2)
        assert len(diffs) == 1
        assert 'a.y: 2 (cached) != 3 (current)' in diffs[0]


class TestArtifactConfigSaveAndCheck:
    """Test ArtifactConfig save/check_cached via a concrete subclass (DatasetConfig)."""

    def test_save_and_check_roundtrip(self):
        args = OmegaConf.create({
            'dataset': {'type': 'oscar', 'language': 'hy', 'cache_dir': 'data/hy'},
            'seed': 42
        })
        config = DatasetConfig.from_args(args)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')
            config.save(config_path)
            assert os.path.exists(config_path)
            assert config.check_cached(config_path) is True

    def test_save_creates_directories(self):
        args = OmegaConf.create({
            'dataset': {'type': 'oscar', 'language': 'hy', 'cache_dir': 'data/hy'},
            'seed': 42
        })
        config = DatasetConfig.from_args(args)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'nested', 'dir', 'config.yaml')
            config.save(config_path)
            assert os.path.exists(config_path)

    def test_check_cached_no_file(self):
        """No cached config should return True (no error)."""
        args = OmegaConf.create({
            'dataset': {'type': 'oscar', 'language': 'hy', 'cache_dir': 'data/hy'},
            'seed': 42
        })
        result = DatasetConfig.from_args(args).check_cached('/nonexistent/config.yaml')
        assert result is True

    def test_check_cached_mismatch_raises(self):
        """Different configs should raise ValueError by default."""
        args1 = OmegaConf.create({
            'dataset': {'type': 'oscar', 'language': 'hy', 'cache_dir': 'data/hy'},
            'seed': 42
        })
        args2 = OmegaConf.create({
            'dataset': {'type': 'oscar', 'language': 'ka', 'cache_dir': 'data/ka'},
            'seed': 42
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')
            DatasetConfig.from_args(args1).save(config_path)

            with pytest.raises(ValueError) as exc_info:
                DatasetConfig.from_args(args2).check_cached(config_path)

            assert "CONFIG MISMATCH" in str(exc_info.value)
            assert "Untokenized Dataset" in str(exc_info.value)

    def test_check_cached_mismatch_warning_mode(self):
        """With error_on_mismatch=False, should return False instead of error."""
        args1 = OmegaConf.create({
            'dataset': {'type': 'oscar', 'language': 'hy', 'cache_dir': 'data/hy'},
            'seed': 42
        })
        args2 = OmegaConf.create({
            'dataset': {'type': 'oscar', 'language': 'ka', 'cache_dir': 'data/ka'},
            'seed': 42
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')
            DatasetConfig.from_args(args1).save(config_path)

            result = DatasetConfig.from_args(args2).check_cached(
                config_path, error_on_mismatch=False
            )
            assert result is False


class TestDatasetConfig:
    """Test DatasetConfig extraction and dict conversion."""

    def test_oscar_config(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy_oscar'
            },
            'seed': 42
        })

        config = DatasetConfig.from_args(args).to_dict()
        assert config['type'] == 'oscar'
        assert config['language'] == 'hy'
        assert config['seed'] == 42

    def test_plaintext_config(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'plaintext',
                'path': '/path/to/data.txt',
                'cache_dir': 'data/my_corpus'
            },
            'seed': 42
        })

        config = DatasetConfig.from_args(args).to_dict()
        assert config['type'] == 'plaintext'
        assert config['path'] == '/path/to/data.txt'
        assert config['seed'] == 42

    def test_multinomial_config(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'multinomial',
                'sources': [{'type': 'oscar', 'language': 'hy'}],
                'alpha': 0.7,
                'total_samples': 1000000,
                'cache_dir': 'data/multi',
                'dev_size': 0.1,
            },
            'seed': 42
        })

        config = DatasetConfig.from_args(args).to_dict()
        assert config['type'] == 'multinomial'
        assert config['alpha'] == 0.7
        assert config['total_samples'] == 1000000
        assert config['dev_size'] == 0.1
        assert config['seed'] == 42

    def test_multinomial_config_dev_size_fallback(self):
        """dev_size in training config should work with a deprecation warning."""
        args = OmegaConf.create({
            'dataset': {
                'type': 'multinomial',
                'sources': [{'type': 'oscar', 'language': 'hy'}],
                'alpha': 0.7,
                'total_samples': 1000000,
                'cache_dir': 'data/multi',
            },
            'training': {'dev_size': 0.2},
            'seed': 42
        })

        with pytest.warns(FutureWarning, match="should be under 'dataset'"):
            config = DatasetConfig.from_args(args).to_dict()
        assert config['dev_size'] == 0.2



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
    Test suite for TokenizerConfig.tokenizer_id() method.

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
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.tokenizer_id() == "xglm564m_focus-v50k-s100k"

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
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.tokenizer_id() == "xglm564m_focus-v32k-s1m_no-additional"

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
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.tokenizer_id() == "xglm564m_focus-v32k-s1m_no-additional"

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
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.tokenizer_id() == "gpt2_focus-v128k-s5m"

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
            },
            'seed': 42
        })
        tok_config = TokenizerConfig.from_args(args)
        assert tok_config.tokenizer_id() == "llama27b_focus-v50k-s100k"


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
            },
            'seed': 42,
        }
        if init_model_id is not None:
            args['init_model_id'] = init_model_id
        return OmegaConf.create(args)

    def test_tokenizer_id_without_init_model_id(self):
        """Without init_model_id, derives shortname from hf_model."""
        tok_config = TokenizerConfig.from_args(self._make_args())
        assert tok_config.tokenizer_id().startswith("xglm17b_")

    def test_tokenizer_id_with_init_model_id(self):
        """With init_model_id, uses it instead of derived shortname."""
        tok_config = TokenizerConfig.from_args(self._make_args(init_model_id="xglm1b"))
        assert tok_config.tokenizer_id().startswith("xglm1b_")
        assert "xglm17b" not in tok_config.tokenizer_id()

    def test_cache_dir_with_init_model_id(self):
        """cache_dir should use init_model_id in the directory name."""
        tok_config = TokenizerConfig.from_args(self._make_args(init_model_id="xglm1b"))
        path = tok_config.cache_dir("old_germanic")
        assert "xglm1b_" in path
        assert "xglm17b" not in path

    def test_all_suffixes_consistent(self):
        """tokenizer_id and cache_dir should both use the same model id."""
        tok_config = TokenizerConfig.from_args(self._make_args(init_model_id="v81"))
        assert tok_config.tokenizer_id().startswith("v81_")
        assert "/v81_" in tok_config.cache_dir("got")


class TestTokenizedDatasetConfig:
    """Test TokenizedDatasetConfig extraction, dict conversion, and cache paths."""

    def test_from_args_no_focus(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy',
                'dev_size': 0.1,
            },
            'training': {'max_length': 512},
            'focus': {'enabled': False},
            'hf_model': 'facebook/xglm-564M',
            'seed': 42
        })

        config = TokenizedDatasetConfig.from_args(args).to_dict()
        assert config['max_length'] == 512
        assert config['dev_size'] == 0.1
        assert 'dataset' in config
        assert config['dataset']['type'] == 'oscar'
        assert 'tokenizer' not in config
        assert config['hf_model'] == 'facebook/xglm-564M'

    def test_from_args_no_focus_with_init_model_id(self):
        """When FOCUS is disabled and init_model_id is set, both are tracked."""
        args = OmegaConf.create({
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy',
                'dev_size': 0.1,
            },
            'training': {'max_length': 512},
            'focus': {'enabled': False},
            'hf_model': '/local/path/to/checkpoint',
            'init_model_id': 'v81',
            'seed': 42
        })

        config = TokenizedDatasetConfig.from_args(args).to_dict()
        assert config['hf_model'] == '/local/path/to/checkpoint'
        assert config['init_model_id'] == 'v81'
        assert 'tokenizer' not in config

    def test_from_args_with_focus(self):
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy',
                'dev_size': 0.2,
            },
            'training': {'max_length': 1024},
            'focus': {
                'enabled': True,
                'vocab_size': 32768,
                'num_samples': 500000
            },
            'seed': 42
        })

        config = TokenizedDatasetConfig.from_args(args).to_dict()
        assert config['max_length'] == 1024
        assert config['dev_size'] == 0.2
        assert 'dataset' in config
        assert 'tokenizer' in config
        assert config['tokenizer']['vocab_size'] == 32768

    def test_cache_dir_no_focus(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy',
                'dev_size': 0.1,
            },
            'training': {'max_length': 512},
            'focus': {'enabled': False},
            'hf_model': 'facebook/xglm-564M',
            'seed': 42
        })

        tok_dataset = TokenizedDatasetConfig.from_args(args)
        assert tok_dataset.cache_dir('data/hy') == 'data/hy/tokenized_xglm564m'

    def test_cache_dir_with_focus(self):
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy',
                'dev_size': 0.1,
            },
            'training': {'max_length': 512},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 100000,
            },
            'seed': 42
        })

        tok_dataset = TokenizedDatasetConfig.from_args(args)
        assert tok_dataset.cache_dir('data/hy') == (
            'data/hy/tokenized_xglm564m_focus-v16k-s100k'
        )

    def test_cache_dir_with_init_model_id(self):
        args = OmegaConf.create({
            'dataset': {
                'type': 'oscar',
                'language': 'hy',
                'cache_dir': 'data/hy',
                'dev_size': 0.1,
            },
            'training': {'max_length': 512},
            'focus': {'enabled': False},
            'hf_model': '/local/path/to/checkpoint',
            'init_model_id': 'v81',
            'seed': 42
        })

        tok_dataset = TokenizedDatasetConfig.from_args(args)
        assert tok_dataset.cache_dir('data/hy') == 'data/hy/tokenized_v81'


def _focus_args(**overrides):
    """Build a minimal args DictConfig for focus_embedding_hash tests."""
    args = OmegaConf.create({
        'hf_model': 'facebook/xglm-564M',
        'dataset': {
            'type': 'oscar',
            'language': 'hy',
            'cache_dir': 'data/hy',
            'dev_size': 0.1,
        },
        'training': {'max_length': 512},
        'focus': {
            'enabled': True,
            'vocab_size': 16384,
            'num_samples': 100_000,
            'fasttext_model_min_count': 4,
        },
        'seed': 1,
    })
    return OmegaConf.merge(args, OmegaConf.create(overrides))


def _multinomial_args(**source_overrides):
    """Multinomial dataset args with two sources."""
    sources = source_overrides.pop('sources', [
        {'id': 'enwiki', 'sampling_prob': 0.5, 'upsampling_factor': 1.0},
        {'id': 'ang', 'sampling_prob': 0.5, 'upsampling_factor': 2.0},
    ])
    args = OmegaConf.create({
        'hf_model': 'facebook/xglm-564M',
        'dataset': {
            'type': 'multinomial',
            'cache_dir': 'data/oldgerm',
            'dev_size': 0.1,
            'alpha': 0.5,
            'total_samples': 5_000_000,
            'sources': sources,
        },
        'training': {'max_length': 512},
        'focus': {
            'enabled': True,
            'vocab_size': 16384,
            'num_samples': 100_000,
            'fasttext_model_min_count': 4,
        },
        'seed': 1,
    })
    return OmegaConf.merge(args, OmegaConf.create(source_overrides))


class TestFocusEmbeddingHash:
    """Test that focus_embedding_hash captures every input that changes what
    fastText sees during FOCUS, and only those inputs."""

    def test_deterministic_and_short(self):
        args = _focus_args()
        h1 = focus_embedding_hash(args)
        h2 = focus_embedding_hash(args)
        assert h1 == h2
        assert len(h1) == 8
        assert all(c in '0123456789abcdef' for c in h1)

    def test_sensitive_to_num_samples(self):
        h_a = focus_embedding_hash(_focus_args(focus={'num_samples': 100_000}))
        h_b = focus_embedding_hash(_focus_args(focus={'num_samples': 200_000}))
        assert h_a != h_b

    def test_sensitive_to_seed(self):
        h_a = focus_embedding_hash(_focus_args(seed=1))
        h_b = focus_embedding_hash(_focus_args(seed=2))
        assert h_a != h_b

    def test_sensitive_to_fasttext_min_count(self):
        h_a = focus_embedding_hash(_focus_args(focus={'fasttext_model_min_count': 4}))
        h_b = focus_embedding_hash(_focus_args(focus={'fasttext_model_min_count': 8}))
        assert h_a != h_b

    def test_sensitive_to_dataset_language(self):
        h_a = focus_embedding_hash(_focus_args(dataset={'language': 'hy'}))
        h_b = focus_embedding_hash(_focus_args(dataset={'language': 'ka'}))
        assert h_a != h_b

    def test_insensitive_to_unrelated_knobs(self):
        h_a = focus_embedding_hash(_focus_args(focus={'vocab_size': 16384}))
        h_b = focus_embedding_hash(_focus_args(focus={'vocab_size': 32768}))
        assert h_a == h_b
        # training knobs don't enter the hash
        h_c = focus_embedding_hash(_focus_args(training={'max_length': 1024}))
        assert h_a == h_c

    def test_multinomial_alpha_changes_hash(self):
        h_a = focus_embedding_hash(_multinomial_args(dataset={'alpha': 0.5}))
        h_b = focus_embedding_hash(_multinomial_args(dataset={'alpha': 0.7}))
        assert h_a != h_b

    def test_multinomial_total_samples_changes_hash(self):
        h_a = focus_embedding_hash(_multinomial_args(dataset={'total_samples': 5_000_000}))
        h_b = focus_embedding_hash(_multinomial_args(dataset={'total_samples': 10_000_000}))
        assert h_a != h_b

    def test_multinomial_per_source_changes_hash(self):
        base_sources = [
            {'id': 'enwiki', 'sampling_prob': 0.5, 'upsampling_factor': 1.0},
            {'id': 'ang', 'sampling_prob': 0.5, 'upsampling_factor': 2.0},
        ]
        alt_sources = [
            {'id': 'enwiki', 'sampling_prob': 0.5, 'upsampling_factor': 1.0},
            {'id': 'ang', 'sampling_prob': 0.5, 'upsampling_factor': 3.0},
        ]
        h_a = focus_embedding_hash(_multinomial_args(sources=base_sources))
        h_b = focus_embedding_hash(_multinomial_args(sources=alt_sources))
        assert h_a != h_b

    def test_focus_dataset_overrides_training_dataset(self):
        # When focus.dataset is set, hash depends on focus.dataset and not on args.dataset.
        focus_ds = {'type': 'plaintext', 'path': '/data/got_only.txt', 'cache_dir': 'data/got'}
        a = _focus_args(focus={'dataset': focus_ds})
        b = _focus_args(focus={'dataset': focus_ds}, dataset={'language': 'ka'})
        assert focus_embedding_hash(a) == focus_embedding_hash(b)

        focus_ds_alt = {'type': 'plaintext', 'path': '/data/other.txt', 'cache_dir': 'data/other'}
        c = _focus_args(focus={'dataset': focus_ds_alt})
        assert focus_embedding_hash(a) != focus_embedding_hash(c)

    def test_seed_still_matters_with_focus_dataset(self):
        focus_ds = {'type': 'plaintext', 'path': '/data/got.txt', 'cache_dir': 'data/got'}
        h_a = focus_embedding_hash(_focus_args(focus={'dataset': focus_ds}, seed=1))
        h_b = focus_embedding_hash(_focus_args(focus={'dataset': focus_ds}, seed=2))
        assert h_a != h_b

    def test_focus_dataset_key_order_does_not_change_hash(self):
        """Reordering keys in the focus dataset spec must not change the hash.

        `focus.dataset` is a free-form mapping copied straight out of the user's
        YAML, so its key order is whatever the file happens to use. The hash is
        stable across that only because focus_embedding_hash passes
        sort_keys=True to json.dumps. Without it, editing a config purely for
        readability would change the hash, miss the embedding cache, and pay for
        a full fastText training run to arrive at identical embeddings.
        """
        spec = {'type': 'plaintext', 'path': '/data/got.txt', 'cache_dir': 'data/got'}
        reordered = {'cache_dir': 'data/got', 'path': '/data/got.txt', 'type': 'plaintext'}
        assert list(spec) != list(reordered)

        h_a = focus_embedding_hash(_focus_args(focus={'dataset': spec}))
        h_b = focus_embedding_hash(_focus_args(focus={'dataset': reordered}))
        assert h_a == h_b

    def test_num_samples_still_matters_with_focus_dataset(self):
        focus_ds = {'type': 'plaintext', 'path': '/data/got.txt', 'cache_dir': 'data/got'}
        h_a = focus_embedding_hash(_focus_args(
            focus={'dataset': focus_ds, 'num_samples': 100_000},
        ))
        h_b = focus_embedding_hash(_focus_args(
            focus={'dataset': focus_ds, 'num_samples': 200_000},
        ))
        assert h_a != h_b

    def test_fasttext_min_count_still_matters_with_focus_dataset(self):
        focus_ds = {'type': 'plaintext', 'path': '/data/got.txt', 'cache_dir': 'data/got'}
        h_a = focus_embedding_hash(_focus_args(
            focus={'dataset': focus_ds, 'fasttext_model_min_count': 4},
        ))
        h_b = focus_embedding_hash(_focus_args(
            focus={'dataset': focus_ds, 'fasttext_model_min_count': 8},
        ))
        assert h_a != h_b


class TestTokenizerConfigEmbeddingFieldStripping:
    """Tests for the migration / decoupling change in to_dict and check_cached."""

    def _make_args(self, **overrides):
        return _focus_args(**overrides)

    def test_to_dict_excludes_embedding_only_fields(self):
        args = self._make_args()
        d = TokenizerConfig.from_args(args).to_dict()
        for k in ('train_dataset_cache', 'focus_dataset', 'fasttext_model_min_count'):
            assert k not in d, f"to_dict should not emit {k}"

    def test_check_cached_tolerates_legacy_fields(self, tmp_path):
        # Simulate a tokenizer cache written by old code that included the
        # now-stripped fields. check_cached must NOT consider them a mismatch.
        cached = {
            'hf_model': 'facebook/xglm-564M',
            'vocab_size': 16384,
            'num_samples': 100000,
            'character_coverage': 1.0,
            'inherit_additional_special_tokens': True,
            'use_seed_vocabulary': False,
            'seed_vocab_multiplier': 5.0,
            'seed_lambda': 0.5,
            'seed_min_frequency': 1,
            'seed_round_mode': 'round',
            'seed_score_mode': 'count',
            'seed': 1,
            'init_model_id': None,
            'tokenizer_path': None,
            # Legacy fields that current to_dict no longer emits:
            'train_dataset_cache': 'data/hy_legacy',
            'focus_dataset': None,
            'fasttext_model_min_count': 4,
        }
        cfg_path = tmp_path / 'training_config.yaml'
        import yaml as _yaml
        with open(cfg_path, 'w') as f:
            _yaml.safe_dump(cached, f)

        args = self._make_args()
        # Should be a no-op return True, NOT raise.
        assert TokenizerConfig.from_args(args).check_cached(str(cfg_path)) is True

    def test_check_cached_still_raises_on_real_tokenizer_diff(self, tmp_path):
        args1 = self._make_args(focus={'vocab_size': 16384})
        args2 = self._make_args(focus={'vocab_size': 32768})
        cfg_path = tmp_path / 'training_config.yaml'
        TokenizerConfig.from_args(args1).save(str(cfg_path))
        with pytest.raises(ValueError, match='vocab_size'):
            TokenizerConfig.from_args(args2).check_cached(str(cfg_path))

    def test_mix_change_does_not_invalidate_tokenizer(self, tmp_path):
        # Two mixes (different cache_dir / language) but same tokenizer params.
        args1 = self._make_args(dataset={'cache_dir': 'data/mix_a', 'language': 'hy'})
        args2 = self._make_args(dataset={'cache_dir': 'data/mix_b', 'language': 'hy'})
        cfg_path = tmp_path / 'training_config.yaml'
        TokenizerConfig.from_args(args1).save(str(cfg_path))
        assert TokenizerConfig.from_args(args2).check_cached(str(cfg_path)) is True

    def test_focus_dataset_change_does_not_invalidate_tokenizer(self, tmp_path):
        args1 = self._make_args(focus={'dataset': {
            'type': 'plaintext', 'path': '/a.txt', 'cache_dir': 'data/a'
        }})
        args2 = self._make_args(focus={'dataset': {
            'type': 'plaintext', 'path': '/b.txt', 'cache_dir': 'data/b'
        }})
        cfg_path = tmp_path / 'training_config.yaml'
        TokenizerConfig.from_args(args1).save(str(cfg_path))
        assert TokenizerConfig.from_args(args2).check_cached(str(cfg_path)) is True


class TestMixSlugSeedKeying:
    """A non-default seed gets its own mix directory; the default changes nothing.

    Mixes that differ only by seed contain different samples, so they must not
    share a directory. The key is omitted at the default so that slugs written
    before seed-keying are unchanged -- 113 GiB of mix directories on the
    cluster depend on that, and every one of them was built at the default.
    """

    BASE = {
        'alpha': 0.5,
        'total_samples': 5000000,
        'dev_size': 0.1,
        'sources': [{'id': 'a'}, {'id': 'b'}],
    }

    def test_default_seed_matches_an_absent_seed(self):
        assert multinomial_mix_slug({**self.BASE, 'seed': 1}) == multinomial_mix_slug(self.BASE)

    def test_non_default_seed_gets_its_own_slug(self):
        assert multinomial_mix_slug({**self.BASE, 'seed': 2}) != multinomial_mix_slug(self.BASE)

    def test_distinct_seeds_stay_distinct(self):
        slugs = {multinomial_mix_slug({**self.BASE, 'seed': s}) for s in (1, 2, 3, 4)}
        assert len(slugs) == 4

    def test_the_readable_segments_are_unchanged_by_seed(self):
        """Only the digest moves, so a mix stays recognizable in a listing."""
        with_seed = multinomial_mix_slug({**self.BASE, 'seed': 7})
        assert with_seed.startswith("mix_a0.5_s5m_")
