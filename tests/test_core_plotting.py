"""Tests for lapt_core.plotting — run grouping and YAML config handling."""

import argparse

import pandas as pd
import pytest

pytest.importorskip("plotnine")

from lapt_core.plotting import (  # noqa: E402
    aggregate_groups,
    load_config,
    normalize_groups,
    parse_group_specs,
    per_metric_limits_from_mapping,
)


def _runs(**curves):
    """Build a wide training log from {run label: [(step, loss), ...]}."""
    rows = []
    for label, points in curves.items():
        for step, loss in points:
            rows.append({"run": label, "step": step, "loss": loss})
    return pd.DataFrame(rows)


class TestAggregateGroups:
    def test_mean_replaces_members(self):
        data = _runs(
            a_s1=[(1, 1.0), (2, 2.0)],
            a_s2=[(1, 3.0), (2, 4.0)],
            baseline=[(1, 9.0), (2, 9.0)],
        )
        result = aggregate_groups(data, ["loss"], {"a": [r"^a_"]})

        assert set(result["run"]) == {"a (n=2)", "baseline"}
        group_rows = result[result["run"] == "a (n=2)"].sort_values("step")
        assert group_rows["loss"].tolist() == [2.0, 3.0]

    def test_drops_steps_not_logged_by_every_member(self, capsys):
        """A member that stopped early must not make the mean jump."""
        data = _runs(
            long=[(1, 1.0), (2, 1.0), (3, 1.0)],
            short=[(1, 3.0), (2, 3.0)],
        )
        result = aggregate_groups(data, ["loss"], {"g": ["long", "short"]})

        assert result["step"].tolist() == [1, 2]
        assert "dropped 1 of 3" in capsys.readouterr().err

    def test_minmax_band(self):
        data = _runs(x1=[(1, 1.0)], x2=[(1, 2.0)], x3=[(1, 6.0)])
        result = aggregate_groups(data, ["loss"], {"x": ["x"]}, band="minmax")

        assert result["loss__band_lower"].tolist() == [1.0]
        assert result["loss__band_upper"].tolist() == [6.0]

    def test_std_band_is_symmetric_sample_std(self):
        data = _runs(x1=[(1, 1.0)], x2=[(1, 3.0)])
        result = aggregate_groups(data, ["loss"], {"x": ["x"]}, band="std")

        # sample std of [1, 3] is sqrt(2)
        assert result["loss__band_lower"].iloc[0] == pytest.approx(2.0 - 2**0.5)
        assert result["loss__band_upper"].iloc[0] == pytest.approx(2.0 + 2**0.5)

    def test_resumed_run_duplicate_step_keeps_last(self):
        data = _runs(r1=[(1, 5.0), (1, 1.0)], r2=[(1, 3.0)])
        result = aggregate_groups(data, ["loss"], {"r": ["r"]})

        assert result["loss"].tolist() == [2.0]

    def test_unmatched_group_warns_and_leaves_data(self, capsys):
        data = _runs(a=[(1, 1.0)])
        result = aggregate_groups(data, ["loss"], {"nope": ["zzz"]})

        assert result["run"].tolist() == ["a"]
        assert "matched no runs" in capsys.readouterr().err

    def test_rejects_unknown_band(self):
        with pytest.raises(ValueError):
            aggregate_groups(_runs(a=[(1, 1.0)]), ["loss"], {"a": ["a"]}, band="iqr")


class TestGroupSpecs:
    def test_repeated_name_accumulates_regexes(self):
        groups = parse_group_specs(["a=x", "a=y", "b=z=w"])
        assert groups == {"a": ["x", "y"], "b": ["z=w"]}

    @pytest.mark.parametrize("spec", ["noequals", "=regex", "name="])
    def test_malformed(self, spec):
        with pytest.raises(ValueError):
            parse_group_specs([spec])

    def test_normalize_accepts_string_or_list(self):
        assert normalize_groups({"a": "x", "b": ["y", "z"]}) == {"a": ["x"], "b": ["y", "z"]}

    @pytest.mark.parametrize("bad", [["x"], {"a": []}, {"a": 3}])
    def test_normalize_rejects(self, bad):
        with pytest.raises(ValueError):
            normalize_groups(bad)


class TestConfig:
    @staticmethod
    def _parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config")
        parser.add_argument("--state-pattern")
        parser.add_argument("--ylim", nargs="+", type=float)
        parser.add_argument("--group", action="append")
        return parser

    def test_dashes_and_underscores_map_to_dest(self, tmp_path):
        path = tmp_path / "plot.yaml"
        path.write_text("state-pattern: foo\nylim: 0\ngroups:\n  a: x\n")

        config = load_config(path, self._parser())

        assert config == {"state_pattern": "foo", "ylim": [0], "groups": {"a": ["x"]}}

    @pytest.mark.parametrize("key", ["bogus", "config", "group"])
    def test_rejects_unknown_keys(self, tmp_path, key):
        path = tmp_path / "plot.yaml"
        path.write_text(f"{key}: 1\n")
        with pytest.raises(ValueError, match="unknown key"):
            load_config(path, self._parser())

    def test_ylims_mapping(self):
        limits = per_metric_limits_from_mapping({"loss": [None, 3], "bpc": [1]})
        assert limits == {"loss": (None, 3.0), "bpc": (1.0, None)}
