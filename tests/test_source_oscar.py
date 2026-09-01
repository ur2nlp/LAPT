"""Tests for `OscarDataset`, with the gated download stubbed out."""

from unittest.mock import patch

import pytest
import yaml
from datasets import Dataset, DatasetDict, load_from_disk

from lapt.core.artifacts import ConfigMismatchError
from lapt.sources.oscar import OscarDataset


@pytest.fixture
def fake_oscar():
    """Stand in for the gated OSCAR download with two multi-line documents."""
    documents = DatasetDict({
        'train': Dataset.from_dict({
            'text': ["first doc line one\n\nfirst doc line two", "second doc"],
            'meta': [{}, {}],
        })
    })
    with patch('lapt.sources.oscar.load_dataset', return_value=documents) as mock:
        yield mock


class TestBuild:
    def test_documents_are_split_into_lines(self, tmp_path, fake_oscar):
        source = OscarDataset(str(tmp_path / "cache"), "en")
        source.resolve()

        dataset = load_from_disk(source.path)
        assert dataset['train']['text'] == [
            "first doc line one",
            "first doc line two",
            "second doc",
        ]

    def test_source_columns_are_dropped(self, tmp_path, fake_oscar):
        source = OscarDataset(str(tmp_path / "cache"), "en")
        source.resolve()

        assert load_from_disk(source.path)['train'].column_names == ['text']

    def test_language_is_passed_through(self, tmp_path, fake_oscar):
        OscarDataset(str(tmp_path / "cache"), "is").resolve()

        assert fake_oscar.call_args.kwargs['language'] == "is"


class TestCaching:
    def test_config_record_omits_seed(self, tmp_path, fake_oscar):
        """The whole language split is taken, so the seed must not key this."""
        source = OscarDataset(str(tmp_path / "cache"), "en")
        source.resolve()

        with open(source.config_path) as record:
            assert yaml.safe_load(record) == {'type': 'oscar', 'language': "en"}

    def test_second_resolve_does_not_download_again(self, tmp_path, fake_oscar):
        cache_dir = str(tmp_path / "cache")
        OscarDataset(cache_dir, "en").resolve()
        OscarDataset(cache_dir, "en").resolve()

        assert fake_oscar.call_count == 1

    def test_different_language_is_a_mismatch(self, tmp_path, fake_oscar):
        cache_dir = str(tmp_path / "cache")
        OscarDataset(cache_dir, "en").resolve()

        with pytest.raises(ConfigMismatchError):
            OscarDataset(cache_dir, "is").resolve()
