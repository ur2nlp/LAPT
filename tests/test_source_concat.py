"""Tests for `ConcatDataset` and the source factory it dispatches through."""

import os

import pytest
import yaml
from datasets import load_from_disk

from lapt.sources import SOURCE_TYPES, ConcatDataset
from lapt.sources.factory import build_source, source_type
from lapt.sources.plaintext import PlaintextDataset
from lapt_core.artifacts import ConfigMismatchError


@pytest.fixture
def corpora(tmp_path):
    """Two small plaintext corpora with their source configs."""
    first = tmp_path / "first.txt"
    first.write_text("alpha\nbeta\n", encoding='utf-8')
    second = tmp_path / "second.txt"
    second.write_text("gamma\n", encoding='utf-8')
    return [
        {'type': 'plaintext', 'id': 'first', 'path': str(first)},
        {'type': 'plaintext', 'id': 'second', 'path': str(second)},
    ]


class TestFactory:
    def test_builds_the_registered_class(self, tmp_path, corpora):
        source = build_source(str(tmp_path / "c"), corpora[0])
        assert isinstance(source, PlaintextDataset)

    def test_type_defaults_to_oscar_for_configs_predating_the_field(self):
        assert source_type({'language': 'en'}) == 'oscar'

    def test_unregistered_type_is_reported_with_the_known_ones(self, tmp_path):
        with pytest.raises(ValueError, match="Known types:"):
            build_source(str(tmp_path / "c"), {'type': 'nonesuch'})

    def test_every_registered_type_can_be_looked_up(self):
        for type_name in SOURCE_TYPES.known_types():
            assert SOURCE_TYPES.get(type_name).type_name == type_name


class TestBuild:
    def test_children_are_concatenated_in_order(self, tmp_path, corpora):
        source = ConcatDataset(str(tmp_path / "c"), corpora)
        result = source.resolve()

        assert result['train']['text'] == ["alpha", "beta", "gamma"]

    def test_empty_sources_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            ConcatDataset(str(tmp_path / "c"), [])

    def test_children_cache_under_their_own_ids(self, tmp_path, corpora):
        root = str(tmp_path / "c")
        ConcatDataset(root, corpora).resolve()

        assert os.path.exists(os.path.join(root, "first", "untokenized", "config.yaml"))
        assert os.path.exists(os.path.join(root, "second", "untokenized", "config.yaml"))

    def test_unnamed_children_fall_back_to_the_parent_id(self, tmp_path):
        path = tmp_path / "only.txt"
        path.write_text("solo\n", encoding='utf-8')
        root = str(tmp_path / "c")

        ConcatDataset(root, [{'type': 'plaintext', 'path': str(path)}], parent_id="mix").resolve()

        assert os.path.exists(os.path.join(root, "mix_0", "untokenized"))

    def test_unnamed_children_without_a_parent_id_are_positional(self, tmp_path):
        path = tmp_path / "only.txt"
        path.write_text("solo\n", encoding='utf-8')
        root = str(tmp_path / "c")

        ConcatDataset(root, [{'type': 'plaintext', 'path': str(path)}]).resolve()

        assert os.path.exists(os.path.join(root, "source_0", "untokenized"))


class TestCaching:
    def test_config_records_the_child_configs(self, tmp_path, corpora):
        source = ConcatDataset(str(tmp_path / "c"), corpora)
        source.resolve()

        with open(source.config_path) as record:
            cached = yaml.safe_load(record)
        assert cached['type'] == 'concat'
        assert [child['id'] for child in cached['sources']] == ['first', 'second']

    def test_config_omits_seed(self, tmp_path, corpora):
        """Concatenation preserves order and samples nothing."""
        assert 'seed' not in ConcatDataset(str(tmp_path / "c"), corpora).config()

    def test_a_changed_child_invalidates_the_parent(self, tmp_path, corpora):
        root = str(tmp_path / "c")
        ConcatDataset(root, corpora).resolve()

        elsewhere = tmp_path / "other.txt"
        elsewhere.write_text("delta\n", encoding='utf-8')
        changed = [corpora[0], {'type': 'plaintext', 'id': 'second', 'path': str(elsewhere)}]

        with pytest.raises(ConfigMismatchError):
            ConcatDataset(root, changed).resolve()

    def test_children_are_shared_across_parents(self, tmp_path, corpora):
        """Two composites over the same cache dir reuse one copy of a child."""
        root = str(tmp_path / "c")
        ConcatDataset(root, corpora).resolve()

        child = PlaintextDataset(os.path.join(root, "first"), corpora[0]['path'])
        assert child.exists()
        assert load_from_disk(child.path)['train']['text'] == ["alpha", "beta"]
