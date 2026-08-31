"""Tests for lapt.core.artifacts: config digests, cache resolution, and the graph."""

import json
import os

import pytest
import yaml

from lapt.core.artifacts import (
    ArtifactGraph,
    CachedArtifact,
    ConfigMismatchError,
    config_digest,
    dict_diff,
    format_number,
)


class RecordingArtifact(CachedArtifact):
    """A trivial artifact that records how many times it was built.

    Persists a single string so the tests can distinguish a cache hit (the
    value came off disk) from a rebuild (`build_count` incremented).
    """

    name = "recording"

    def __init__(self, root, payload="v1", name=None):
        super().__init__(root)
        if name is not None:
            self.name = name
        self.payload = payload
        self.build_count = 0

    def config(self):
        return {'payload': self.payload}

    def build(self, deps):
        self.build_count += 1
        return f"{self.payload}:{'+'.join(sorted(deps))}" if deps else self.payload

    def write(self, value, path):
        with open(os.path.join(path, 'value.txt'), 'w') as handle:
            handle.write(value)

    def read(self, path):
        with open(os.path.join(path, 'value.txt')) as handle:
            return handle.read()


class DerivedArtifact(RecordingArtifact):
    """An artifact downstream of `recording`."""

    name = "derived"
    depends_on = ("recording",)


class TestConfigDigest:
    """The digest is the single source of truth for cache identity."""

    def test_is_stable_across_calls(self):
        payload = {'a': 1, 'b': [1, 2, 3]}
        assert config_digest(payload) == config_digest(payload)

    def test_ignores_key_order(self):
        # this is the property that lets a config be reordered without
        # orphaning every cache built from it
        assert config_digest({'a': 1, 'b': 2}) == config_digest({'b': 2, 'a': 1})

    def test_ignores_nested_key_order(self):
        first = {'outer': {'a': 1, 'b': 2}}
        second = {'outer': {'b': 2, 'a': 1}}
        assert config_digest(first) == config_digest(second)

    def test_distinguishes_different_values(self):
        assert config_digest({'a': 1}) != config_digest({'a': 2})

    def test_respects_length(self):
        assert len(config_digest({'a': 1})) == 8
        assert len(config_digest({'a': 1}, length=16)) == 16
        assert config_digest({'a': 1}, length=16).startswith(config_digest({'a': 1}))

    def test_falls_back_to_str_for_unserializable_values(self):
        # a stray Path or enum in a config should not raise
        digest = config_digest({'path': os.PathLike})
        assert len(digest) == 8

    def test_matches_the_inline_implementation_it_replaced(self):
        # pins the wire format: changing it would silently move every cache
        # path that embeds a digest
        import hashlib

        payload = {'alpha': 0.5, 'sources': [{'id': 'eng'}]}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        assert config_digest(payload) == expected


class TestDictDiff:
    """Diffs drive the operator-facing mismatch message."""

    def test_identical_dicts_have_no_diffs(self):
        assert dict_diff({'a': 1}, {'a': 1}) == []

    def test_reports_changed_value_with_both_sides(self):
        diffs = dict_diff({'a': 1}, {'a': 2})
        assert len(diffs) == 1
        assert '1 (cached)' in diffs[0] and '2 (current)' in diffs[0]

    def test_reports_keys_missing_from_each_side(self):
        diffs = dict_diff({'only_cached': 1}, {'only_current': 2})
        joined = '\n'.join(diffs)
        assert 'only_cached' in joined and 'only_current' in joined

    def test_recurses_into_nested_dicts_with_dotted_path(self):
        diffs = dict_diff({'outer': {'inner': 1}}, {'outer': {'inner': 2}})
        assert len(diffs) == 1
        assert 'outer.inner' in diffs[0]

    def test_output_is_deterministically_ordered(self):
        cached = {'b': 1, 'a': 1, 'c': 1}
        current = {'b': 2, 'a': 2, 'c': 2}
        assert dict_diff(cached, current) == dict_diff(cached, current)


class TestFormatNumber:
    """Truncating, never rounding, so a count always maps to one string."""

    def test_thousands_and_millions_truncate(self):
        assert format_number(50_000) == "50k"
        assert format_number(1_500) == "1k"
        assert format_number(1_500_000) == "1m"

    def test_small_numbers_are_unsuffixed(self):
        assert format_number(999) == "999"
        assert format_number(0) == "0"


class TestCachedArtifactResolve:
    """resolve() is the whole point: build once, then load."""

    def test_builds_when_absent(self, tmp_path):
        artifact = RecordingArtifact(str(tmp_path))
        assert artifact.resolve() == "v1"
        assert artifact.build_count == 1
        assert os.path.exists(artifact.path)

    def test_second_resolve_loads_from_cache(self, tmp_path):
        RecordingArtifact(str(tmp_path)).resolve()
        second = RecordingArtifact(str(tmp_path))
        assert second.resolve() == "v1"
        assert second.build_count == 0

    def test_saves_a_config_record_beside_the_artifact(self, tmp_path):
        artifact = RecordingArtifact(str(tmp_path))
        artifact.resolve()
        with open(artifact.config_path) as handle:
            assert yaml.safe_load(handle) == {'payload': 'v1'}

    def test_changed_config_raises_rather_than_reusing(self, tmp_path):
        RecordingArtifact(str(tmp_path), payload="v1").resolve()
        changed = RecordingArtifact(str(tmp_path), payload="v2")
        with pytest.raises(ConfigMismatchError) as exc_info:
            changed.resolve()
        assert "CONFIG MISMATCH" in str(exc_info.value)
        assert changed.build_count == 0

    def test_fresh_rebuilds_over_a_mismatch(self, tmp_path):
        RecordingArtifact(str(tmp_path), payload="v1").resolve()
        changed = RecordingArtifact(str(tmp_path), payload="v2")
        assert changed.resolve(fresh=True) == "v2"
        assert changed.build_count == 1

    def test_untracked_cache_is_tolerated_not_rejected(self, tmp_path, capsys):
        # an artifact directory predating config tracking must keep working
        artifact = RecordingArtifact(str(tmp_path))
        artifact.resolve()
        os.remove(artifact.config_path)

        reloaded = RecordingArtifact(str(tmp_path))
        assert reloaded.resolve() == "v1"
        assert "without config tracking" in capsys.readouterr().err

    def test_missing_dependency_is_reported(self, tmp_path):
        derived = DerivedArtifact(str(tmp_path))
        with pytest.raises(KeyError, match="recording"):
            derived.resolve()

    def test_clear_removes_the_cache_directory(self, tmp_path):
        artifact = RecordingArtifact(str(tmp_path))
        artifact.resolve()
        artifact.clear()
        assert not os.path.exists(artifact.path)


class TestDigestAddressedPaths:
    """path_includes_digest makes configurations coexist instead of colliding."""

    def test_different_configs_get_different_paths(self, tmp_path):
        class Addressed(RecordingArtifact):
            path_includes_digest = True

        first = Addressed(str(tmp_path), payload="v1")
        second = Addressed(str(tmp_path), payload="v2")
        assert first.path != second.path

    def test_both_configs_survive_and_neither_mismatches(self, tmp_path):
        class Addressed(RecordingArtifact):
            path_includes_digest = True

        assert Addressed(str(tmp_path), payload="v1").resolve() == "v1"
        assert Addressed(str(tmp_path), payload="v2").resolve() == "v2"
        # the first cache is untouched by the second build
        assert Addressed(str(tmp_path), payload="v1").resolve() == "v1"

    def test_path_digest_agrees_with_the_saved_config(self, tmp_path):
        # the property that makes drift impossible: both derive from config()
        class Addressed(RecordingArtifact):
            path_includes_digest = True

        artifact = Addressed(str(tmp_path), payload="v1")
        artifact.resolve()
        with open(artifact.config_path) as handle:
            saved = yaml.safe_load(handle)
        assert artifact.path.endswith(config_digest(saved))


class TestArtifactGraph:
    """Dependency wiring, memoization, and cascade invalidation."""

    def _graph(self, tmp_path):
        return ArtifactGraph(
            RecordingArtifact(str(tmp_path)),
            DerivedArtifact(str(tmp_path)),
        )

    def test_resolves_dependencies_first(self, tmp_path):
        graph = self._graph(tmp_path)
        assert graph.get("derived") == "v1:recording"

    def test_memoizes_within_a_graph(self, tmp_path):
        graph = self._graph(tmp_path)
        graph.get("derived")
        graph.get("recording")
        assert graph.artifacts["recording"].build_count == 1

    def test_rejects_duplicate_names(self, tmp_path):
        with pytest.raises(ValueError, match="Duplicate"):
            ArtifactGraph(
                RecordingArtifact(str(tmp_path)),
                RecordingArtifact(str(tmp_path)),
            )

    def test_unknown_name_lists_what_is_registered(self, tmp_path):
        graph = self._graph(tmp_path)
        with pytest.raises(KeyError, match="recording"):
            graph.get("nonexistent")

    def test_detects_cycles(self, tmp_path):
        class Looping(RecordingArtifact):
            name = "a"
            depends_on = ("b",)

        class AlsoLooping(RecordingArtifact):
            name = "b"
            depends_on = ("a",)

        graph = ArtifactGraph(Looping(str(tmp_path)), AlsoLooping(str(tmp_path)))
        with pytest.raises(ValueError, match="Cyclic"):
            graph.get("a")

    def test_dependents_includes_transitive_stages(self, tmp_path):
        class Final(RecordingArtifact):
            name = "final"
            depends_on = ("derived",)

        graph = ArtifactGraph(
            RecordingArtifact(str(tmp_path)),
            DerivedArtifact(str(tmp_path)),
            Final(str(tmp_path)),
        )
        assert graph.dependents("recording") == ["derived", "final"]

    def test_dependents_is_order_independent(self, tmp_path):
        # registering downstream stages before upstream ones must not hide them
        class Final(RecordingArtifact):
            name = "final"
            depends_on = ("derived",)

        graph = ArtifactGraph(
            Final(str(tmp_path)),
            DerivedArtifact(str(tmp_path)),
            RecordingArtifact(str(tmp_path)),
        )
        assert sorted(graph.dependents("recording")) == ["derived", "final"]

    def test_invalidate_clears_the_whole_subtree(self, tmp_path):
        graph = self._graph(tmp_path)
        graph.get("derived")
        recording = graph.artifacts["recording"]
        derived = graph.artifacts["derived"]
        assert os.path.exists(derived.path)

        cleared = graph.invalidate("recording")

        assert cleared == ["recording", "derived"]
        assert not os.path.exists(recording.path)
        assert not os.path.exists(derived.path)

    def test_invalidate_forces_a_rebuild_on_next_get(self, tmp_path):
        graph = self._graph(tmp_path)
        graph.get("derived")
        graph.invalidate("recording")
        graph.get("derived")
        assert graph.artifacts["recording"].build_count == 2

    def test_leaf_invalidation_leaves_upstream_alone(self, tmp_path):
        graph = self._graph(tmp_path)
        graph.get("derived")
        graph.invalidate("derived")
        assert os.path.exists(graph.artifacts["recording"].path)
