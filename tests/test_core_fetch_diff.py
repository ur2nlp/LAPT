"""Tests for lapt_core.fetch_diff — id normalization and the fetch decision."""

import json

import pytest

from lapt_core.fetch_diff import get_local_status, main, normalize_exp_id


class TestNormalizeExpId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("v8L", "v08L"),
            ("v08L", "v08L"),
            ("v1", "v01"),
            ("v10", "v10"),
            ("v138", "v138"),
            ("v74L-i100", "v74L-i100"),
            ("whisper5", "whisper5"),
            ("pilot", "pilot"),
        ],
    )
    def test_pads_only_v_prefixed_numbers(self, raw, expected):
        assert normalize_exp_id(raw) == expected

    def test_is_idempotent(self):
        """Applied twice it must not pad again, or ids would drift per run."""
        assert normalize_exp_id(normalize_exp_id("v8L")) == normalize_exp_id("v8L")


class TestLocalStatus:
    @staticmethod
    def _state(tmp_path, **fields):
        path = tmp_path / "trainer_state.json"
        path.write_text(json.dumps(fields))
        return path

    def test_complete_when_finished_at_max_steps(self, tmp_path):
        path = self._state(
            tmp_path, global_step=20000, max_steps=20000,
            log_history=[{"train_runtime": 1234.0}],
        )
        assert get_local_status(path) == "complete"

    def test_early_stopped_when_finished_short(self, tmp_path):
        path = self._state(
            tmp_path, global_step=15000, max_steps=20000,
            log_history=[{"train_runtime": 900.0}],
        )
        assert get_local_status(path) == "early_stopped"

    def test_training_while_no_runtime_recorded(self, tmp_path):
        """`train_runtime` appears only in the entry Trainer writes at the end."""
        path = self._state(
            tmp_path, global_step=5000, max_steps=20000,
            log_history=[{"loss": 1.2}],
        )
        assert get_local_status(path) == "training"


class TestNormalizationIsOptIn:
    """The flag decides whether a remote `v8L` matches a local `v08L.json`."""

    @staticmethod
    def _outputs(tmp_path):
        states = tmp_path / "trainer_states"
        states.mkdir()
        (states / "v08L.json").write_text(json.dumps({
            "global_step": 20000, "max_steps": 20000,
            "log_history": [{"train_runtime": 10.0}],
        }))
        (tmp_path / "configs").mkdir()
        return tmp_path

    def _run(self, tmp_path, monkeypatch, capsys, argv):
        """Run main over one remote line; it exits only when nothing is to fetch."""
        import io

        monkeypatch.setattr(
            "sys.stdin", io.StringIO("v8L\tdir\t/remote/ts.json\t/remote/cfg.yaml\n")
        )
        monkeypatch.setattr("sys.argv", ["fetch_diff", "--remote", "host",
                                         "--outputs-dir", str(tmp_path), *argv])
        try:
            main(env_prefix="TEST")
        except SystemExit:
            pass
        return capsys.readouterr()

    def test_on_matches_the_padded_local_record(self, tmp_path, monkeypatch, capsys):
        out = self._run(self._outputs(tmp_path), monkeypatch, capsys, ["--normalize-ids"])

        assert "1 skipped" in out.err
        assert "Nothing to fetch" in out.err
        assert "scp" not in out.out

    def test_off_treats_them_as_different_runs(self, tmp_path, monkeypatch, capsys):
        """Without padding, remote v8L does not find local v08L and is re-fetched."""
        out = self._run(self._outputs(tmp_path), monkeypatch, capsys, ["--no-normalize-ids"])

        assert "1 to fetch" in out.err
        assert "trainer_states/v8L.json" in out.out
