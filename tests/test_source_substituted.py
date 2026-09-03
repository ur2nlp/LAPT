"""Tests for `SubstitutedDataset` and the substitution config parser."""

import os
import re

import pytest
import yaml
from datasets import load_from_disk

from lapt.sources.factory import build_source
from lapt.sources.plaintext import PlaintextDataset
from lapt.sources.substituted import SubstitutedDataset, parse_substitutions
from lapt_core.artifacts import ConfigMismatchError


@pytest.fixture
def base_source(tmp_path):
    """A resolved plaintext source whose text carries newline runs."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha\nbeta\ngamma\n", encoding='utf-8')
    return PlaintextDataset(str(tmp_path / "cache"), str(corpus))


class TestParseSubstitutions:
    def test_absent_config_is_empty(self):
        assert parse_substitutions(None) == []
        assert parse_substitutions([]) == []

    def test_replacement_defaults_to_deletion(self):
        assert parse_substitutions([{'pattern': r'\d+'}]) == [(r'\d+', '')]

    def test_declaration_order_is_preserved(self):
        parsed = parse_substitutions([
            {'pattern': 'a', 'replacement': 'b'},
            {'pattern': 'b', 'replacement': 'c'},
        ])
        assert parsed == [('a', 'b'), ('b', 'c')]

    def test_bad_regex_fails_at_parse_time(self):
        """Fail on the config, not part-way through mapping a large dataset."""
        with pytest.raises(re.error):
            parse_substitutions([{'pattern': '([unclosed'}])

    def test_missing_pattern_is_reported(self):
        with pytest.raises(ValueError, match="must specify a 'pattern'"):
            parse_substitutions([{'replacement': ' '}])


class TestBuild:
    def test_substitutions_are_applied(self, base_source):
        source = SubstitutedDataset(base_source, [(r'\n+', ' ')])
        result = source.resolve()

        assert result['train']['text'] == ["alpha", "beta", "gamma"]

    def test_substitutions_apply_in_order(self, base_source):
        source = SubstitutedDataset(base_source, [('alpha', 'beta'), ('beta', 'gamma')])
        result = source.resolve()

        assert result['train']['text'][0] == "gamma"

    def test_the_base_cache_is_left_untouched(self, base_source):
        SubstitutedDataset(base_source, [('alpha', 'ALPHA')]).resolve()

        assert load_from_disk(base_source.path)['train']['text'][0] == "alpha"

    def test_every_string_column_is_transformed(self, tmp_path):
        """Substitution is type-agnostic: instruction sources have two columns."""
        import json

        from lapt.sources.instruction_jsonl import InstructionJsonlDataset

        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({'prompt': "ask X", 'response': "say X"}) + "\n", encoding='utf-8'
        )
        base = InstructionJsonlDataset(str(tmp_path / "cache"), str(path))

        result = SubstitutedDataset(base, [('X', 'Y')]).resolve()

        assert result['train']['prompt'] == ["ask Y"]
        assert result['train']['response'] == ["say Y"]

    def test_an_empty_substitution_list_is_refused(self, base_source):
        with pytest.raises(ValueError, match="at least one substitution"):
            SubstitutedDataset(base_source, [])


class TestPathAddressing:
    def test_cache_is_a_sibling_of_the_base(self, base_source):
        source = SubstitutedDataset(base_source, [(r'\n+', ' ')])

        assert os.path.dirname(source.path) == os.path.dirname(base_source.path)
        assert os.path.basename(source.path).startswith("untokenized_sub_")

    def test_the_digest_is_pinned(self, base_source):
        """The suffix names directories that already exist on disk.

        It is also embedded in the names of the tokenized caches derived from
        them, so a change to how it is derived orphans both: the next run
        misses, rebuilds, and the old directories linger, and nothing about
        that looks like a failure. Pinning the literal makes it a decision.

        The realistic trigger is *adding* a field to the hashed payload -- regex
        flags, a match limit, a per-column scope -- not renaming one. Tests
        comparing two digests cannot see that, since both properties they check
        survive it.
        """
        source = SubstitutedDataset(base_source, parse_substitutions(
            [{'pattern': r'\n+', 'replacement': ' '}]
        ))

        assert os.path.basename(source.path) == "untokenized_sub_9ec26528"

    def test_distinct_patterns_get_distinct_paths(self, base_source):
        first = SubstitutedDataset(base_source, [(r'\n', ' ')])
        second = SubstitutedDataset(base_source, [('x', 'z')])

        assert first.path != second.path

    def test_the_record_cannot_disagree_with_the_path(self, base_source):
        """The two untracked-by-the-digest fields are fixed by the path itself.

        `base` is the path prefix and `type` is constant, so keying the digest
        on the substitutions alone still cannot address a cache whose record
        describes something else.
        """
        source = SubstitutedDataset(base_source, [(r'\n+', ' ')])
        source.resolve()

        with open(source.config_path) as record:
            cached = yaml.safe_load(record)
        assert cached['base'] == os.path.basename(base_source.path)
        assert cached['type'] == 'substituted'


class TestCaching:
    def test_second_resolve_does_not_rebuild(self, base_source, tmp_path):
        source = SubstitutedDataset(base_source, [(r'\n+', ' ')])
        source.resolve()

        # a cache hit must not consult the underlying corpus
        os.remove(base_source.file_path)
        assert SubstitutedDataset(base_source, [(r'\n+', ' ')]).resolve()['train']['text'] == [
            "alpha", "beta", "gamma",
        ]

    def test_config_omits_seed(self, base_source):
        """Substitution is deterministic."""
        assert 'seed' not in SubstitutedDataset(base_source, [('a', 'b')]).config()

    def test_a_tampered_record_is_caught(self, base_source):
        source = SubstitutedDataset(base_source, [(r'\n+', ' ')])
        source.resolve()

        with open(source.config_path, 'w') as record:
            yaml.dump({'type': 'substituted', 'base': 'elsewhere', 'substitutions': []}, record)

        with pytest.raises(ConfigMismatchError):
            SubstitutedDataset(base_source, [(r'\n+', ' ')]).validate()


class TestFactoryIntegration:
    def test_a_config_with_substitutions_is_wrapped(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("alpha\nbeta\n", encoding='utf-8')
        config = {
            'type': 'plaintext',
            'path': str(corpus),
            'substitutions': [{'pattern': 'alpha', 'replacement': 'ALPHA'}],
        }

        source = build_source(str(tmp_path / "cache"), config)

        assert isinstance(source, SubstitutedDataset)
        assert source.resolve()['train']['text'] == ["ALPHA", "beta"]

    def test_a_config_without_substitutions_is_not_wrapped(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("alpha\n", encoding='utf-8')

        source = build_source(str(tmp_path / "cache"), {'type': 'plaintext', 'path': str(corpus)})

        assert isinstance(source, PlaintextDataset)

    def test_substitutions_reach_a_mix_child(self, tmp_path):
        """A mix's children carry their own substitutions, as the configs use."""
        from lapt.sources.multinomial import MultinomialDataset

        first = tmp_path / "first.txt"
        first.write_text("\n".join(f"keep {i}" for i in range(20)) + "\n", encoding='utf-8')
        second = tmp_path / "second.txt"
        second.write_text("\n".join(f"MARK {i}" for i in range(20)) + "\n", encoding='utf-8')

        sources = [
            {'type': 'plaintext', 'id': 'plain', 'path': str(first)},
            {
                'type': 'plaintext',
                'id': 'subbed',
                'path': str(second),
                'substitutions': [{'pattern': 'MARK', 'replacement': 'SWAPPED'}],
            },
        ]
        result = MultinomialDataset(
            str(tmp_path / "cache"), sources, 0.5, 40, -1,
        ).resolve()

        texts = " ".join(result['train']['text'])
        assert "SWAPPED" in texts
        assert "MARK" not in texts
