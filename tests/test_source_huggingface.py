"""Tests for `HuggingFaceDataset`, with the download stubbed out.

The conditional-seed behavior gets particular attention: it is the first source
whose tracked parameters depend on its other parameters, which a static field
list could not express.
"""

from unittest.mock import patch

import pytest
import yaml
from datasets import Dataset, DatasetDict, load_from_disk

from lapt.sources.huggingface import HuggingFaceDataset
from lapt_core.artifacts import ConfigMismatchError


def documents():
    """Two multi-line documents in a non-default text column."""
    return DatasetDict({
        'train': Dataset.from_dict({
            'content': ["one two three four\nfive six", "seven eight nine ten"],
        })
    })['train']


@pytest.fixture
def fake_download():
    with patch('lapt.sources.huggingface.load_dataset', return_value=documents()) as mock:
        yield mock


class TestConditionalSeed:
    def test_seed_is_absent_without_max_samples(self, tmp_path, fake_download):
        source = HuggingFaceDataset(str(tmp_path / "c"), "fake/ds", text_column='content')
        assert 'seed' not in source.config()

    def test_seed_is_present_with_max_samples(self, tmp_path, fake_download):
        source = HuggingFaceDataset(
            str(tmp_path / "c"), "fake/ds", text_column='content', max_samples=2, seed=7
        )
        assert source.config()['seed'] == 7

    def test_seed_change_does_not_invalidate_a_whole_split_cache(self, tmp_path, fake_download):
        """A cache nothing sampled must survive a seed change."""
        cache = str(tmp_path / "c")
        HuggingFaceDataset(cache, "fake/ds", text_column='content', seed=1).resolve()

        reused = HuggingFaceDataset(cache, "fake/ds", text_column='content', seed=99)
        assert reused.validate() is True

    def test_seed_change_builds_a_sibling_of_a_subsampled_cache(self, tmp_path, fake_download):
        """A new seed is a new sample beside the old one, not a conflict with it."""
        # split_into_lines=False to skip the estimation phase, which divides
        # max_samples by 10 and so cannot be exercised at test scale
        cache = str(tmp_path / "c")
        first = HuggingFaceDataset(
            cache, "fake/ds", text_column='content',
            max_samples=2, split_into_lines=False, seed=1,
        )
        first.resolve()
        second = HuggingFaceDataset(
            cache, "fake/ds", text_column='content',
            max_samples=2, split_into_lines=False, seed=99,
        )
        second.resolve()

        assert second.path == str(tmp_path / "c" / "untokenized_seed99")
        assert first.validate() is True
        assert second.validate() is True


class TestSeedKeyedPath:
    def test_default_seed_keeps_the_bare_name(self, tmp_path):
        """Every cache built before seed-keying was built at the default."""
        source = HuggingFaceDataset(str(tmp_path / "c"), "fake/ds", max_samples=2, seed=1)
        assert source.path == str(tmp_path / "c" / "untokenized")

    def test_uncapped_source_ignores_the_seed(self, tmp_path):
        source = HuggingFaceDataset(str(tmp_path / "c"), "fake/ds", seed=2)
        assert source.path == str(tmp_path / "c" / "untokenized")

    def test_a_mismatched_record_at_the_bare_name_still_raises(self, tmp_path, fake_download):
        """A legacy cache drawn at a non-default seed is not silently reused."""
        cache = str(tmp_path / "c")
        HuggingFaceDataset(
            cache, "fake/ds", text_column='content',
            max_samples=2, split_into_lines=False, seed=1,
        ).resolve()
        record = tmp_path / "c" / "untokenized" / "config.yaml"
        tracked = yaml.safe_load(record.read_text())
        tracked['seed'] = 3
        record.write_text(yaml.safe_dump(tracked))

        with pytest.raises(ConfigMismatchError):
            HuggingFaceDataset(
                cache, "fake/ds", text_column='content',
                max_samples=2, split_into_lines=False, seed=1,
            ).resolve()


class TestSampleSeedFromConfig:
    def test_global_seed_does_not_reach_the_sample(self, tmp_path):
        """A seed replicate reuses the default-seed sample unless asked not to."""
        source = HuggingFaceDataset.from_config(
            str(tmp_path / "c"), {'name': "fake/ds", 'max_samples': 2}, seed=2
        )
        assert source.seed == 1
        assert source.path == str(tmp_path / "c" / "untokenized")

    def test_sample_seed_is_honored(self, tmp_path):
        source = HuggingFaceDataset.from_config(
            str(tmp_path / "c"), {'name': "fake/ds", 'max_samples': 2, 'sample_seed': 2}, seed=1
        )
        assert source.config()['seed'] == 2
        assert source.path == str(tmp_path / "c" / "untokenized_seed2")


class TestBuild:
    def test_documents_are_split_into_lines_by_default(self, tmp_path, fake_download):
        source = HuggingFaceDataset(str(tmp_path / "c"), "fake/ds", text_column='content')
        source.resolve()

        assert load_from_disk(source.path)['train']['text'] == [
            "one two three four", "five six", "seven eight nine ten",
        ]

    def test_documents_are_kept_whole_when_not_splitting(self, tmp_path, fake_download):
        source = HuggingFaceDataset(
            str(tmp_path / "c"), "fake/ds", text_column='content', split_into_lines=False
        )
        source.resolve()

        assert load_from_disk(source.path)['train']['text'] == [
            "one two three four\nfive six", "seven eight nine ten",
        ]

    def test_short_examples_are_filtered(self, tmp_path, fake_download):
        source = HuggingFaceDataset(
            str(tmp_path / "c"), "fake/ds", text_column='content', min_words_per_line=3
        )
        source.resolve()

        assert "five six" not in load_from_disk(source.path)['train']['text']


class TestC4MigrationTarget:
    """Pins the record a migrated C4 cache must carry.

    Two large caches on the cluster predate this conversion and are expensive
    enough to rebuild that they will be migrated instead. Their existing record
    is this one minus `split_into_lines` and `seed`, so the migration adds
    exactly those two. Pinning the expected result here means the migration can
    be checked against the code rather than against anyone's recollection.
    """

    def test_expected_record_for_the_cluster_c4_caches(self, tmp_path):
        source = HuggingFaceDataset(
            str(tmp_path / "c"),
            "allenai/c4",
            config="en",
            split="train",
            text_column="text",
            max_samples=5020000,
            min_words_per_line=8,
            oversampling_factor=8,
            split_into_lines=True,
            seed=1,
        )

        assert source.config() == {
            'type': 'huggingface',
            'name': 'allenai/c4',
            'config': 'en',
            'split': 'train',
            'text_column': 'text',
            'max_samples': 5020000,
            'min_words_per_line': 8,
            'oversampling_factor': 8,
            'split_into_lines': True,
            'seed': 1,
        }

    def test_the_record_round_trips_through_yaml(self, tmp_path, fake_download):
        """The migration writes YAML by hand, so the record must survive it."""
        source = HuggingFaceDataset(
            str(tmp_path / "c"), "fake/ds", text_column='content',
            max_samples=2, split_into_lines=False,
        )
        source.resolve()

        with open(source.config_path) as record:
            assert yaml.safe_load(record) == source.config()


class TestMismatchMessage:
    def test_points_at_the_seed_and_away_from_fresh_dataset(self, tmp_path, fake_download):
        """The old message's advice, fresh_dataset, removes every source."""
        cache = str(tmp_path / "c")
        HuggingFaceDataset(
            cache, "fake/ds", text_column='content',
            max_samples=2, split_into_lines=False, seed=1,
        ).resolve()
        record = tmp_path / "c" / "untokenized" / "config.yaml"
        tracked = yaml.safe_load(record.read_text())
        tracked['seed'] = 3
        record.write_text(yaml.safe_dump(tracked))

        with pytest.raises(ConfigMismatchError) as caught:
            HuggingFaceDataset(
                cache, "fake/ds", text_column='content',
                max_samples=2, split_into_lines=False, seed=1,
            ).resolve()

        assert "untokenized_seed<n>" in str(caught.value)
        assert "Not fresh_dataset=true" in str(caught.value)
