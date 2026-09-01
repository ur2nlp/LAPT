"""Tests for `PlaintextDataset` and the pre-refactor cache guard on `SourceDataset`."""

import os

import pytest
import yaml
from datasets import load_from_disk

from lapt.core.artifacts import ConfigMismatchError
from lapt.sources.base import LEGACY_CONFIG_FILENAME
from lapt.sources.plaintext import PlaintextDataset


@pytest.fixture
def corpus(tmp_path):
    """Write a small plaintext corpus with blank lines and padding to strip."""
    path = tmp_path / "corpus.txt"
    path.write_text("first line\n\n  second line  \n\n\nthird line\n", encoding='utf-8')
    return str(path)


class TestBuild:
    def test_reads_non_empty_stripped_lines(self, tmp_path, corpus):
        source = PlaintextDataset(str(tmp_path / "cache"), corpus)
        source.resolve()

        dataset = load_from_disk(source.path)
        assert dataset['train']['text'] == ["first line", "second line", "third line"]

    def test_caches_under_untokenized(self, tmp_path, corpus):
        source = PlaintextDataset(str(tmp_path / "cache"), corpus)
        source.resolve()

        assert source.path == os.path.join(str(tmp_path / "cache"), "untokenized")
        assert os.path.exists(os.path.join(source.path, "config.yaml"))

    def test_missing_file_raises(self, tmp_path):
        source = PlaintextDataset(str(tmp_path / "cache"), str(tmp_path / "absent.txt"))

        with pytest.raises(FileNotFoundError):
            source.resolve()

    def test_file_without_content_raises(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("\n\n  \n", encoding='utf-8')
        source = PlaintextDataset(str(tmp_path / "cache"), str(empty))

        with pytest.raises(ValueError, match="no non-empty lines"):
            source.resolve()


class TestCaching:
    def test_second_resolve_does_not_rebuild(self, tmp_path, corpus):
        cache_dir = str(tmp_path / "cache")
        PlaintextDataset(cache_dir, corpus).resolve()

        # a cache hit must not consult the source file
        os.remove(corpus)
        dataset = PlaintextDataset(cache_dir, corpus).resolve()

        assert dataset['train']['text'][0] == "first line"

    def test_different_path_is_a_mismatch(self, tmp_path, corpus):
        cache_dir = str(tmp_path / "cache")
        PlaintextDataset(cache_dir, corpus).resolve()

        other = tmp_path / "other.txt"
        other.write_text("elsewhere\n", encoding='utf-8')

        with pytest.raises(ConfigMismatchError):
            PlaintextDataset(cache_dir, str(other)).resolve()

    def test_config_record_omits_seed(self, tmp_path, corpus):
        """Nothing in this source is random, so the seed must not key its cache."""
        source = PlaintextDataset(str(tmp_path / "cache"), corpus)
        source.resolve()

        with open(source.config_path) as record:
            assert yaml.safe_load(record) == {'type': 'plaintext', 'path': corpus}


class TestPreRefactorCacheGuard:
    def test_legacy_record_is_refused_rather_than_accepted(self, tmp_path, corpus):
        cache_dir = str(tmp_path / "cache")
        source = PlaintextDataset(cache_dir, corpus)
        source.resolve()

        # simulate a cache written before sources became artifacts
        os.rename(
            source.config_path,
            os.path.join(source.path, LEGACY_CONFIG_FILENAME),
        )

        with pytest.raises(ConfigMismatchError, match="PRE-REFACTOR SOURCE CACHE"):
            PlaintextDataset(cache_dir, corpus).resolve()

    def test_legacy_record_alongside_a_current_one_is_ignored(self, tmp_path, corpus):
        """A leftover legacy file must not shadow a valid current record."""
        cache_dir = str(tmp_path / "cache")
        source = PlaintextDataset(cache_dir, corpus)
        source.resolve()

        legacy = os.path.join(source.path, LEGACY_CONFIG_FILENAME)
        with open(legacy, 'w') as handle:
            yaml.dump({'type': 'plaintext', 'path': '/stale'}, handle)

        assert PlaintextDataset(cache_dir, corpus).validate() is True
