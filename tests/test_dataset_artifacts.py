"""Tests for the shared `DatasetArtifact` base and its type registry."""

import os

import pytest
from datasets import Dataset, DatasetDict

from lapt.core.artifacts import ConfigMismatchError
from lapt.core.dataset_artifacts import DatasetArtifact, DatasetRegistry


class LinesDataset(DatasetArtifact):
    """A minimal concrete dataset artifact: one example per supplied line."""

    type_name = "lines"

    def __init__(self, root, lines=("a", "b"), name=None):
        super().__init__(root)
        if name is not None:
            self.name = name
        self.lines = list(lines)
        self.build_count = 0

    def config(self):
        return {'type': 'lines', 'lines': self.lines}

    def build(self, deps):
        self.build_count += 1
        return DatasetDict({'train': Dataset.from_dict({'text': self.lines})})


class TestRoundTrip:
    def test_build_writes_a_loadable_dataset(self, tmp_path):
        artifact = LinesDataset(str(tmp_path), lines=["one", "two"])
        built = artifact.resolve()

        assert built['train']['text'] == ["one", "two"]
        assert artifact.build_count == 1
        assert os.path.exists(artifact.config_path)

    def test_second_resolve_reads_from_disk(self, tmp_path):
        LinesDataset(str(tmp_path), lines=["one", "two"]).resolve()

        second = LinesDataset(str(tmp_path), lines=["one", "two"])
        loaded = second.resolve()

        assert second.build_count == 0
        assert loaded['train']['text'] == ["one", "two"]

    def test_changed_config_is_rejected(self, tmp_path):
        LinesDataset(str(tmp_path), lines=["one"]).resolve()

        with pytest.raises(ConfigMismatchError):
            LinesDataset(str(tmp_path), lines=["two"]).resolve()

    def test_fresh_rebuilds_over_a_mismatched_cache(self, tmp_path):
        LinesDataset(str(tmp_path), lines=["one"]).resolve()

        rebuilt = LinesDataset(str(tmp_path), lines=["two"])
        assert rebuilt.resolve(fresh=True)['train']['text'] == ["two"]


class TestRegistry:
    def test_register_and_create(self, tmp_path):
        registry = DatasetRegistry()
        registry.register(LinesDataset)

        artifact = registry.create("lines", str(tmp_path), lines=["x"])

        assert isinstance(artifact, LinesDataset)
        assert registry.known_types() == ["lines"]
        assert "lines" in registry

    def test_register_returns_the_class_for_decorator_use(self):
        registry = DatasetRegistry()
        assert registry.register(LinesDataset) is LinesDataset

    def test_unknown_type_lists_the_known_ones(self, tmp_path):
        registry = DatasetRegistry()
        registry.register(LinesDataset)

        with pytest.raises(ValueError, match="Known types: lines"):
            registry.get("plaintext")

    def test_unknown_type_on_an_empty_registry(self):
        with pytest.raises(ValueError, match=r"\(none registered\)"):
            DatasetRegistry().get("lines")

    def test_missing_type_name_is_refused(self):
        class Unnamed(DatasetArtifact):
            pass

        with pytest.raises(ValueError, match="without a type_name"):
            DatasetRegistry().register(Unnamed)

    def test_duplicate_type_name_is_refused(self):
        class Rival(DatasetArtifact):
            type_name = "lines"

        registry = DatasetRegistry()
        registry.register(LinesDataset)

        with pytest.raises(ValueError, match="already registered to LinesDataset"):
            registry.register(Rival)

    def test_registries_are_independent(self):
        first = DatasetRegistry()
        first.register(LinesDataset)

        assert DatasetRegistry().known_types() == []
