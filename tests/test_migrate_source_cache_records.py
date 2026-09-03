"""Tests for the source-cache record migration tool.

This script writes into a live cache tree, so its decisions are worth pinning:
which directories it touches, what it adds, and what it refuses.
"""

import importlib.util
import os
import pathlib

import yaml

from lapt.sources.huggingface import HuggingFaceDataset

_MODULE_PATH = pathlib.Path(__file__).parent.parent / "tools" / "migrate_source_cache_records.py"
_spec = importlib.util.spec_from_file_location("migrate_source_cache_records", _MODULE_PATH)
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)


def write_legacy(root, relative: str, record: dict) -> str:
    """Create a cache directory holding a legacy record.

    Args:
        root: Base directory.
        relative: Path of the cache directory under `root`.
        record: The legacy record to write.

    Returns:
        The cache directory's path.
    """
    directory = os.path.join(str(root), relative)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, migrate.LEGACY_CONFIG_FILENAME), 'w') as handle:
        yaml.dump(record, handle, sort_keys=False)
    return directory


C4_LEGACY = {
    'type': 'huggingface',
    'name': 'allenai/c4',
    'config': 'en',
    'split': 'train',
    'text_column': 'text',
    'max_samples': 5020000,
    'min_words_per_line': 8,
    'oversampling_factor': 8,
}


class TestDiscovery:
    def test_finds_only_directories_with_a_legacy_record(self, tmp_path):
        write_legacy(tmp_path, "lang/got/untokenized", {'type': 'plaintext', 'path': 'a.txt'})
        os.makedirs(tmp_path / "lang" / "unrelated", exist_ok=True)

        found = migrate.find_legacy_records(str(tmp_path))

        assert found == [str(tmp_path / "lang" / "got" / "untokenized")]


class TestPlan:
    def test_a_complete_record_is_copied_unchanged(self, tmp_path):
        record = {'type': 'plaintext', 'path': 'data/got.txt'}
        directory = write_legacy(tmp_path, "lang/got/untokenized", record)

        plan = migrate.plan_for(directory, seed=1)

        assert plan['status'] == 'write'
        assert plan['record'] == record
        assert plan['additions'] == {}

    def test_a_c4_record_gains_exactly_the_two_missing_fields(self, tmp_path):
        directory = write_legacy(tmp_path, "lang/eng/untokenized", C4_LEGACY)

        plan = migrate.plan_for(directory, seed=1)

        assert plan['additions'] == {'split_into_lines': True, 'seed': 1}

    def test_the_c4_target_is_what_the_source_class_would_write(self, tmp_path):
        """Ties the migration to the class rather than to a transcribed literal."""
        directory = write_legacy(tmp_path, "lang/eng/untokenized", C4_LEGACY)

        plan = migrate.plan_for(directory, seed=1)
        expected = HuggingFaceDataset(
            str(tmp_path / "lang" / "eng"), **{k: v for k, v in C4_LEGACY.items() if k != 'type'},
            seed=1,
        ).config()

        assert plan['record'] == expected

    def test_the_recorded_seed_follows_the_flag(self, tmp_path):
        directory = write_legacy(tmp_path, "lang/eng/untokenized", C4_LEGACY)

        assert migrate.plan_for(directory, seed=7)['record']['seed'] == 7

    def test_an_uncapped_source_gains_no_seed(self, tmp_path):
        """The seed is only recorded where it changed the result."""
        directory = write_legacy(tmp_path, "lang/chat/untokenized", {
            'type': 'instruction_hf', 'name': 'x', 'config': None, 'split': 'train',
            'messages_column': 'messages', 'prompt_template': '{user}',
            'response_template': '{assistant}', 'max_samples': None,
        })

        assert 'seed' not in migrate.plan_for(directory, seed=7)['additions']

    def test_a_substituted_record_is_passed_through(self, tmp_path):
        """Its type is not in the registry, since it wraps a source."""
        record = {
            'type': 'substituted',
            'base': 'untokenized',
            'substitutions': [{'pattern': r'\n+', 'replacement': ' '}],
        }
        directory = write_legacy(tmp_path, "lang/chat/untokenized_sub_abc12345", record)

        plan = migrate.plan_for(directory, seed=1)

        assert plan['status'] == 'write'
        assert plan['record'] == record

    def test_an_unknown_type_is_reported_not_migrated(self, tmp_path):
        directory = write_legacy(tmp_path, "lang/x/untokenized", {'type': 'nonesuch'})

        assert migrate.plan_for(directory, seed=1)['status'] == 'unknown-type'

    def test_an_identical_current_record_is_already_migrated(self, tmp_path):
        record = {'type': 'plaintext', 'path': 'data/got.txt'}
        directory = write_legacy(tmp_path, "lang/got/untokenized", record)
        with open(os.path.join(directory, migrate.CONFIG_FILENAME), 'w') as handle:
            yaml.dump(record, handle)

        assert migrate.plan_for(directory, seed=1)['status'] == 'already-migrated'

    def test_a_disagreeing_current_record_is_left_alone(self, tmp_path):
        directory = write_legacy(tmp_path, "lang/got/untokenized", {
            'type': 'plaintext', 'path': 'data/got.txt',
        })
        with open(os.path.join(directory, migrate.CONFIG_FILENAME), 'w') as handle:
            yaml.dump({'type': 'plaintext', 'path': 'somewhere/else.txt'}, handle)

        assert migrate.plan_for(directory, seed=1)['status'] == 'differs'


class TestApply:
    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        directory = write_legacy(tmp_path, "lang/got/untokenized", {
            'type': 'plaintext', 'path': 'a.txt',
        })
        monkeypatch.setattr("sys.argv", ["migrate", str(tmp_path)])

        migrate.main()

        assert not os.path.exists(os.path.join(directory, migrate.CONFIG_FILENAME))

    def test_apply_writes_the_record_and_keeps_the_legacy_one(self, tmp_path, monkeypatch):
        """Keeping both is what lets the older code path keep validating."""
        record = {'type': 'plaintext', 'path': 'a.txt'}
        directory = write_legacy(tmp_path, "lang/got/untokenized", record)
        monkeypatch.setattr("sys.argv", ["migrate", str(tmp_path), "--apply"])

        migrate.main()

        with open(os.path.join(directory, migrate.CONFIG_FILENAME)) as handle:
            assert yaml.safe_load(handle) == record
        assert os.path.exists(os.path.join(directory, migrate.LEGACY_CONFIG_FILENAME))

    def test_apply_is_idempotent(self, tmp_path, monkeypatch):
        write_legacy(tmp_path, "lang/got/untokenized", {'type': 'plaintext', 'path': 'a.txt'})
        monkeypatch.setattr("sys.argv", ["migrate", str(tmp_path), "--apply"])

        assert migrate.main() == 0
        assert migrate.main() == 0

    def test_attention_cases_set_a_nonzero_status(self, tmp_path, monkeypatch):
        write_legacy(tmp_path, "lang/x/untokenized", {'type': 'nonesuch'})
        monkeypatch.setattr("sys.argv", ["migrate", str(tmp_path), "--apply"])

        assert migrate.main() == 1

    def test_a_missing_root_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["migrate", str(tmp_path / "absent")])

        assert migrate.main() == 2

    def test_an_empty_tree_is_a_clean_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["migrate", str(tmp_path), "--apply"])

        assert migrate.main() == 0
