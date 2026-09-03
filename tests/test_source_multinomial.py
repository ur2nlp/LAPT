"""Tests for `MultinomialDataset`: mix addressing, dev splits, and sampling."""

import os

import pytest
import yaml

from lapt.sources.multinomial import MultinomialDataset
from lapt_core.artifacts import ConfigMismatchError


@pytest.fixture
def corpora(tmp_path):
    """A large and a small plaintext source, so upsampling is observable."""
    big = tmp_path / "big.txt"
    big.write_text("\n".join(f"big {i}" for i in range(40)) + "\n", encoding='utf-8')
    small = tmp_path / "small.txt"
    small.write_text("\n".join(f"small {i}" for i in range(4)) + "\n", encoding='utf-8')
    return [
        {'type': 'plaintext', 'id': 'big', 'path': str(big)},
        {'type': 'plaintext', 'id': 'small', 'path': str(small)},
    ]


def mix(root, corpora, **overrides):
    """Build a mix with workable defaults."""
    settings = {'alpha': 0.5, 'total_samples': 40, 'dev_size': 0.25}
    settings.update(overrides)
    return MultinomialDataset(
        root, corpora, settings['alpha'], settings['total_samples'], settings['dev_size'],
        seed=settings.get('seed', 1),
    )


class TestPathAddressing:
    def test_caches_inside_a_mix_directory(self, tmp_path, corpora):
        source = mix(str(tmp_path / "c"), corpora)

        assert os.path.basename(source.path) == "untokenized"
        assert os.path.basename(source.mix_dir).startswith("mix_a0.5_s40_")

    def test_sources_stay_beside_the_mix_not_inside_it(self, tmp_path, corpora):
        """Per-source caches are shared across mixes, so they sit at the parent."""
        root = str(tmp_path / "c")
        mix(root, corpora).resolve()

        assert os.path.exists(os.path.join(root, "big", "untokenized"))
        assert not os.path.exists(os.path.join(root, "big", "untokenized", "mix"))

    def test_a_different_mix_gets_a_different_directory(self, tmp_path, corpora):
        root = str(tmp_path / "c")
        assert mix(root, corpora).mix_dir != mix(root, corpora, alpha=0.7).mix_dir

    def test_a_different_seed_gets_a_different_directory(self, tmp_path, corpora):
        root = str(tmp_path / "c")
        assert mix(root, corpora).mix_dir != mix(root, corpora, seed=2).mix_dir

    def test_two_mixes_share_their_source_caches(self, tmp_path, corpora):
        root = str(tmp_path / "c")
        first = mix(root, corpora)
        first.resolve()

        # the source file is gone, so the second mix can only succeed from cache
        os.remove(corpora[0]['path'])
        os.remove(corpora[1]['path'])
        second = mix(root, corpora, alpha=0.7)
        second.resolve()

        assert first.mix_dir != second.mix_dir
        assert len(second.resolve()['train']) == 40


class TestDevSplits:
    def test_each_source_gets_a_named_dev_split(self, tmp_path, corpora):
        result = mix(str(tmp_path / "c"), corpora).resolve()

        assert set(result) == {'train', 'big', 'small'}

    def test_dev_examples_do_not_appear_in_train(self, tmp_path, corpora):
        """Splitting before upsampling is what prevents leakage."""
        result = mix(str(tmp_path / "c"), corpora).resolve()

        train_texts = set(result['train']['text'])
        for split in ('big', 'small'):
            assert not train_texts & set(result[split]['text'])

    def test_global_skip_produces_no_dev_splits(self, tmp_path, corpora):
        result = mix(str(tmp_path / "c"), corpora, dev_size=-1).resolve()

        assert set(result) == {'train'}

    def test_a_source_can_opt_out_individually(self, tmp_path, corpora):
        corpora[0]['dev_size'] = -1
        result = mix(str(tmp_path / "c"), corpora).resolve()

        assert set(result) == {'train', 'small'}


class TestSampling:
    def test_train_split_has_exactly_total_samples(self, tmp_path, corpora):
        result = mix(str(tmp_path / "c"), corpora, total_samples=37).resolve()

        assert len(result['train']) == 37

    def test_a_small_source_is_upsampled_by_repetition(self, tmp_path, corpora):
        result = mix(str(tmp_path / "c"), corpora, total_samples=60).resolve()

        small_texts = [t for t in result['train']['text'] if t.startswith("small")]
        assert len(small_texts) > len(set(small_texts))


class TestValidation:
    @pytest.mark.parametrize("overrides,message", [
        ({'total_samples': 0}, "total_samples must be positive"),
        ({'alpha': -1}, "alpha must be positive"),
        ({'dev_size': 0}, "ambiguous"),
        ({'dev_size': 5}, "fractional dev_size"),
    ])
    def test_bad_settings_are_refused(self, tmp_path, corpora, overrides, message):
        with pytest.raises(ValueError, match=message):
            mix(str(tmp_path / "c"), corpora, **overrides)

    def test_empty_sources_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            MultinomialDataset(str(tmp_path / "c"), [], 0.5, 10, 0.1)


class TestCaching:
    def test_record_carries_the_seed_unconditionally(self, tmp_path, corpora):
        """Unlike the path, which omits it at the default."""
        source = mix(str(tmp_path / "c"), corpora)
        source.resolve()

        with open(source.config_path) as record:
            assert yaml.safe_load(record)['seed'] == 1

    def test_a_changed_source_invalidates_the_mix(self, tmp_path, corpora):
        root = str(tmp_path / "c")
        mix(root, corpora).resolve()

        elsewhere = tmp_path / "other.txt"
        elsewhere.write_text("delta\n", encoding='utf-8')
        corpora[1]['path'] = str(elsewhere)

        with pytest.raises(ConfigMismatchError):
            mix(root, corpora).resolve()
