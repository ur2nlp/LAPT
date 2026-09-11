"""Tests for `PlaintextDirDataset`."""

import os

import pytest
import yaml
from datasets import load_from_disk

from lapt.sources import SOURCE_TYPES
from lapt.sources.plaintext_dir import PlaintextDirDataset
from lapt_core.artifacts import ConfigMismatchError


@pytest.fixture
def corpus_dir(tmp_path):
    """A directory of shards, plus one file the default pattern excludes."""
    directory = tmp_path / "sagas"
    directory.mkdir()
    (directory / "a.on.txt").write_text("alpha one\nalpha two\n", encoding='utf-8')
    (directory / "b.on.txt").write_text("beta one\n\n  beta two  \n", encoding='utf-8')
    (directory / "notes.md").write_text("not a corpus file\n", encoding='utf-8')
    return directory


class TestRegistration:
    def test_the_type_is_registered(self):
        assert 'plaintext_dir' in SOURCE_TYPES.known_types()

    def test_from_config_reads_directory_and_pattern(self, tmp_path, corpus_dir):
        source = SOURCE_TYPES.get('plaintext_dir').from_config(
            str(tmp_path / "c"),
            {'type': 'plaintext_dir', 'directory': str(corpus_dir), 'pattern': '*.on.txt'},
        )
        assert source.pattern == '*.on.txt'

    def test_pattern_defaults_to_txt(self, tmp_path, corpus_dir):
        source = SOURCE_TYPES.get('plaintext_dir').from_config(
            str(tmp_path / "c"), {'type': 'plaintext_dir', 'directory': str(corpus_dir)}
        )
        assert source.pattern == '*.txt'


class TestBuild:
    def _resolve(self, tmp_path, corpus_dir, pattern='*.on.txt'):
        source = PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), pattern)
        return source, source.resolve()

    def test_reads_every_matching_file_in_name_order(self, tmp_path, corpus_dir):
        _, data = self._resolve(tmp_path, corpus_dir)
        assert data['train']['text'] == [
            "alpha one", "alpha two", "beta one", "beta two",
        ]

    def test_blank_lines_are_dropped_and_whitespace_stripped(self, tmp_path, corpus_dir):
        _, data = self._resolve(tmp_path, corpus_dir)
        assert "" not in data['train']['text']
        assert "  beta two  " not in data['train']['text']

    def test_the_pattern_selects_which_files_are_read(self, tmp_path, corpus_dir):
        _, data = self._resolve(tmp_path, corpus_dir, pattern='a.*.txt')
        assert data['train']['text'] == ["alpha one", "alpha two"]

    def test_caches_under_untokenized(self, tmp_path, corpus_dir):
        source, _ = self._resolve(tmp_path, corpus_dir)
        assert os.path.basename(source.path) == "untokenized"
        assert load_from_disk(source.path)['train'].num_rows == 4

    def test_missing_directory_raises(self, tmp_path):
        source = PlaintextDirDataset(str(tmp_path / "c"), str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            source.resolve()

    def test_a_file_given_instead_of_a_directory_raises(self, tmp_path, corpus_dir):
        source = PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir / "a.on.txt"))
        with pytest.raises(ValueError, match="not a directory"):
            source.resolve()

    def test_a_pattern_matching_nothing_raises(self, tmp_path, corpus_dir):
        source = PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), '*.nope')
        with pytest.raises(ValueError, match="No files found"):
            source.resolve()

    def test_files_without_content_raise(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        (directory / "a.txt").write_text("\n\n   \n", encoding='utf-8')

        source = PlaintextDirDataset(str(tmp_path / "c"), str(directory))
        with pytest.raises(ValueError, match="no non-empty lines"):
            source.resolve()


class TestCaching:
    def test_second_resolve_does_not_rebuild(self, tmp_path, corpus_dir):
        source = PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), '*.on.txt')
        source.resolve()
        mtime = os.path.getmtime(source.path)

        PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), '*.on.txt').resolve()
        assert os.path.getmtime(source.path) == mtime

    def test_a_changed_pattern_is_a_mismatch(self, tmp_path, corpus_dir):
        PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), '*.on.txt').resolve()

        with pytest.raises(ConfigMismatchError):
            PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), 'a.*.txt').resolve()

    def test_an_added_shard_is_a_mismatch(self, tmp_path, corpus_dir):
        """Directory and pattern are unchanged, but the corpus is not.

        Recording only the two configured strings would silently reuse a cache
        that predates the new file.
        """
        PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), '*.on.txt').resolve()
        (corpus_dir / "c.on.txt").write_text("gamma one\n", encoding='utf-8')

        with pytest.raises(ConfigMismatchError, match="files"):
            PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), '*.on.txt').resolve()

    def test_the_record_names_the_files_not_their_full_paths(self, tmp_path, corpus_dir):
        """Basenames keep the record readable and machine-independent."""
        source = PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), '*.on.txt')
        source.resolve()

        with open(source.config_path) as record:
            assert yaml.safe_load(record)['files'] == ["a.on.txt", "b.on.txt"]

    def test_config_omits_seed(self, tmp_path, corpus_dir):
        source = PlaintextDirDataset(str(tmp_path / "c"), str(corpus_dir), '*.on.txt')
        assert 'seed' not in source.config()
