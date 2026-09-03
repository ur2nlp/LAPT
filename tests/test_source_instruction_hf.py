"""Tests for `InstructionHFDataset`, with the download stubbed out."""

from unittest.mock import patch

import pytest
from datasets import Dataset, load_from_disk

from lapt.sources.instruction_hf import InstructionHFDataset
from lapt_core.artifacts import ConfigMismatchError


def chat_dataset():
    """Three examples: two usable pairs and one multi-turn to be dropped."""
    return Dataset.from_dict({
        'messages': [
            [
                {'role': 'user', 'content': "translate hello"},
                {'role': 'assistant', 'content': "hails"},
            ],
            [
                {'role': 'user', 'content': "translate word"},
                {'role': 'assistant', 'content': "waurd"},
            ],
            [
                {'role': 'system', 'content': "be helpful"},
                {'role': 'user', 'content': "hi"},
                {'role': 'assistant', 'content': "hello"},
            ],
        ]
    })


@pytest.fixture
def fake_download():
    with patch(
        'lapt.sources.instruction_hf.load_dataset', side_effect=lambda *a, **k: chat_dataset()
    ) as mock:
        yield mock


class TestBuild:
    def test_multi_turn_examples_are_dropped(self, tmp_path, fake_download):
        source = InstructionHFDataset(str(tmp_path / "c"), "fake/chat")
        source.resolve()

        assert len(load_from_disk(source.path)['train']) == 2

    def test_templates_are_applied(self, tmp_path, fake_download):
        source = InstructionHFDataset(
            str(tmp_path / "c"), "fake/chat",
            prompt_template='Q: {user}\nA:', response_template=' {assistant}',
        )
        source.resolve()

        dataset = load_from_disk(source.path)['train']
        assert dataset['prompt'][0] == "Q: translate hello\nA:"
        assert dataset['response'][0] == " hails"

    def test_message_columns_are_replaced(self, tmp_path, fake_download):
        source = InstructionHFDataset(str(tmp_path / "c"), "fake/chat")
        source.resolve()

        assert load_from_disk(source.path)['train'].column_names == ['prompt', 'response']

    def test_no_surviving_examples_raises(self, tmp_path):
        only_multi_turn = Dataset.from_dict({
            'messages': [[{'role': 'system', 'content': "x"}]],
        })
        with patch('lapt.sources.instruction_hf.load_dataset', return_value=only_multi_turn):
            source = InstructionHFDataset(str(tmp_path / "c"), "fake/chat")
            with pytest.raises(ValueError, match="No examples remained"):
                source.resolve()


class TestConditionalSeed:
    def test_seed_is_absent_without_max_samples(self, tmp_path):
        source = InstructionHFDataset(str(tmp_path / "c"), "fake/chat")
        assert 'seed' not in source.config()

    def test_seed_is_present_with_max_samples(self, tmp_path):
        source = InstructionHFDataset(str(tmp_path / "c"), "fake/chat", max_samples=1, seed=5)
        assert source.config()['seed'] == 5

    def test_seed_change_does_not_invalidate_an_uncapped_cache(self, tmp_path, fake_download):
        cache = str(tmp_path / "c")
        InstructionHFDataset(cache, "fake/chat", seed=1).resolve()

        assert InstructionHFDataset(cache, "fake/chat", seed=99).validate() is True

    def test_seed_change_does_invalidate_a_capped_cache(self, tmp_path, fake_download):
        cache = str(tmp_path / "c")
        InstructionHFDataset(cache, "fake/chat", max_samples=1, seed=1).resolve()

        with pytest.raises(ConfigMismatchError):
            InstructionHFDataset(cache, "fake/chat", max_samples=1, seed=99).resolve()


class TestCaching:
    def test_second_resolve_does_not_download_again(self, tmp_path, fake_download):
        cache = str(tmp_path / "c")
        InstructionHFDataset(cache, "fake/chat").resolve()
        InstructionHFDataset(cache, "fake/chat").resolve()

        assert fake_download.call_count == 1

    def test_template_change_is_a_mismatch(self, tmp_path, fake_download):
        cache = str(tmp_path / "c")
        InstructionHFDataset(cache, "fake/chat").resolve()

        with pytest.raises(ConfigMismatchError):
            InstructionHFDataset(cache, "fake/chat", prompt_template='{user} ->').resolve()
