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

    def test_a_different_seed_gets_different_data_not_just_a_directory(
        self, tmp_path, corpora
    ):
        """The companion the directory test needed.

        `build` used to hardcode seed=1 for both the dev split and the train
        shuffle while the slug and the config record carried the real seed. So
        a non-default seed produced a byte-identical mix in a fresh directory,
        with the record claiming otherwise -- and the directory test above
        passed throughout, because it only ever checked the addressing half.
        """
        root = str(tmp_path / "c")
        first = mix(root, corpora).resolve()
        second = mix(root, corpora, seed=2).resolve()

        assert first['train']['text'] != second['train']['text']
        assert first['big']['text'] != second['big']['text']

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


class TestCanonicalDevSplits:
    """A source that ships its own dev split keeps it instead of being re-carved.

    The reason the mix carves dev *before* upsampling is that a randomly held
    out example must not be duplicated into train. That argument is about
    random splitting: a corpus that arrives already split is disjoint from its
    own train upstream, before anything here runs, so no amount of upsampling
    can leak it -- and it is the split results are comparable on. Carving a
    fresh random slice and discarding the canonical one loses information and
    buys no safety.

    Stub children, because no source in this repository is pre-split.
    """

    @staticmethod
    def _stub_factory(split_map):
        from datasets import Dataset, DatasetDict

        from lapt_core.dataset_artifacts import DatasetArtifact

        class StubSource(DatasetArtifact):
            def __init__(self, cache_dir, splits):
                super().__init__(cache_dir)
                self.splits = splits

            def config(self):
                return {'type': 'stub', 'splits': sorted(self.splits)}

            def build(self, deps):
                return DatasetDict({
                    name: Dataset.from_dict({'text': values})
                    for name, values in self.splits.items()
                })

        def factory(cache_dir, source_config, seed=1):
            return StubSource(cache_dir, split_map[source_config['id']])

        return factory

    def _mix(self, tmp_path, split_map, sources=None, **overrides):
        from lapt_core.composites import MultinomialArtifact

        settings = {'alpha': 0.5, 'total_samples': 20, 'dev_size': 0.25}
        settings.update(overrides)
        return MultinomialArtifact(
            str(tmp_path / "mix"),
            sources if sources is not None else [{'id': name} for name in split_map],
            settings['alpha'], settings['total_samples'], settings['dev_size'],
            child_factory=self._stub_factory(split_map),
            seed=1,
        ).resolve()

    def test_a_canonical_dev_split_is_used_as_is(self, tmp_path):
        split_map = {'a': {
            'train': [f"t{i}" for i in range(8)],
            'validation': ['held-1', 'held-2'],
        }}
        result = self._mix(tmp_path, split_map)

        assert sorted(result['a']['text']) == ['held-1', 'held-2']

    def test_the_canonical_dev_examples_are_not_also_in_train(self, tmp_path):
        split_map = {'a': {
            'train': [f"t{i}" for i in range(8)],
            'validation': ['held-1', 'held-2'],
        }}
        result = self._mix(tmp_path, split_map, total_samples=40)

        assert not set(result['a']['text']) & set(result['train']['text'])

    def test_the_whole_train_split_stays_available_for_sampling(self, tmp_path):
        """Nothing is carved away, so every training example can be sampled."""
        split_map = {'a': {
            'train': [f"t{i}" for i in range(8)],
            'validation': ['held-1'],
        }}
        result = self._mix(tmp_path, split_map, total_samples=8)

        assert set(result['train']['text']) == {f"t{i}" for i in range(8)}

    def test_a_source_without_one_is_still_carved(self, tmp_path):
        split_map = {'a': {'train': [f"t{i}" for i in range(8)]}}
        result = self._mix(tmp_path, split_map)

        assert 'a' in result
        assert set(result['a']['text']) <= {f"t{i}" for i in range(8)}

    def test_a_per_source_dev_size_overrides_the_canonical_split(self, tmp_path):
        split_map = {'a': {
            'train': [f"t{i}" for i in range(8)],
            'validation': ['held-1', 'held-2'],
        }}
        result = self._mix(
            tmp_path, split_map, sources=[{'id': 'a', 'dev_size': 0.25}]
        )

        assert not set(result['a']['text']) & {'held-1', 'held-2'}

    def test_skipping_dev_still_skips(self, tmp_path):
        split_map = {'a': {'train': ['t0', 't1'], 'validation': ['held-1']}}
        result = self._mix(tmp_path, split_map, dev_size=-1)

        assert set(result) == {'train'}

    def test_validation_wins_over_other_dev_names(self, tmp_path):
        split_map = {'a': {
            'train': ['t0', 't1'],
            'validation': ['canonical'],
            'dev': ['other'],
        }}
        result = self._mix(tmp_path, split_map)

        assert result['a']['text'] == ['canonical']

    def test_a_test_split_is_not_mistaken_for_dev(self, tmp_path):
        split_map = {'a': {'train': [f"t{i}" for i in range(8)], 'test': ['unseen']}}
        result = self._mix(tmp_path, split_map)

        assert 'unseen' not in result['a']['text']

    def test_a_source_with_no_train_split_is_refused_clearly(self, tmp_path):
        split_map = {'a': {'validation': ['only']}}
        with pytest.raises(ValueError, match="no 'train' split"):
            self._mix(tmp_path, split_map)
