"""Tests for `InstructionJsonlDataset`."""

import json
import os

import pytest
import yaml
from datasets import load_from_disk

from lapt.core.artifacts import ConfigMismatchError
from lapt.sources.instruction_jsonl import InstructionJsonlDataset


@pytest.fixture
def jsonl_file(tmp_path):
    """Write a small instruction file, with a blank line to be skipped."""
    path = tmp_path / "instructions.jsonl"
    lines = [
        json.dumps({'prompt': "Translate: hello\nResponse:", 'response': " hails"}),
        "",
        json.dumps({'prompt': "Translate: word\nResponse:", 'response': " waurd"}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding='utf-8')
    return str(path)


class TestBuild:
    def test_produces_prompt_and_response_columns(self, tmp_path, jsonl_file):
        source = InstructionJsonlDataset(str(tmp_path / "cache"), jsonl_file)
        source.resolve()

        dataset = load_from_disk(source.path)
        assert dataset['train'].column_names == ['prompt', 'response']
        assert dataset['train']['response'] == [" hails", " waurd"]

    def test_missing_file_raises(self, tmp_path):
        source = InstructionJsonlDataset(
            str(tmp_path / "cache"), str(tmp_path / "absent.jsonl")
        )
        with pytest.raises(FileNotFoundError):
            source.resolve()

    def test_missing_response_field_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({'prompt': "only a prompt"}) + "\n", encoding='utf-8')

        source = InstructionJsonlDataset(str(tmp_path / "cache"), str(path))
        with pytest.raises(ValueError, match="missing 'prompt' or 'response'"):
            source.resolve()

    def test_malformed_json_names_the_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"prompt": "a", "response": "b"}\nnot json\n', encoding='utf-8')

        source = InstructionJsonlDataset(str(tmp_path / "cache"), str(path))
        with pytest.raises(ValueError, match="line 2"):
            source.resolve()


class TestCaching:
    def test_config_record_omits_seed(self, tmp_path, jsonl_file):
        """Every example is kept in file order, so the seed must not key this."""
        source = InstructionJsonlDataset(str(tmp_path / "cache"), jsonl_file)
        source.resolve()

        with open(source.config_path) as record:
            assert yaml.safe_load(record) == {
                'type': 'instruction_jsonl',
                'path': jsonl_file,
            }

    def test_second_resolve_does_not_reread_the_file(self, tmp_path, jsonl_file):
        cache_dir = str(tmp_path / "cache")
        InstructionJsonlDataset(cache_dir, jsonl_file).resolve()

        os.remove(jsonl_file)
        dataset = InstructionJsonlDataset(cache_dir, jsonl_file).resolve()

        assert dataset['train']['response'] == [" hails", " waurd"]

    def test_different_path_is_a_mismatch(self, tmp_path, jsonl_file):
        cache_dir = str(tmp_path / "cache")
        InstructionJsonlDataset(cache_dir, jsonl_file).resolve()

        with pytest.raises(ConfigMismatchError):
            InstructionJsonlDataset(cache_dir, str(tmp_path / "other.jsonl")).resolve()
