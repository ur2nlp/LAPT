"""Behavioral tests for `build_untokenized_source`, the pipeline's entry point.

The per-type tests build source classes directly; these are the only ones that
go from a full Hydra config through to a resolved cache, so they cover the
argument extraction and cache-directory rules that sit between the two.

They run against on-disk plaintext corpora rather than mocks, since the point is
that path resolution, config tracking, and building stay consistent.
"""

import os

import pytest
import yaml
from datasets import load_from_disk
from omegaconf import OmegaConf

from lapt.core.artifacts import ConfigMismatchError, MissingConfigRecordError
from lapt.dataset_utils import build_untokenized_source


@pytest.fixture
def corpus(tmp_path):
    """Write a small plaintext corpus and return its path."""
    path = tmp_path / "corpus.txt"
    path.write_text("the first line\nthe second line\nthe third line\n")
    return str(path)


def plaintext_args(tmp_path, corpus_path, **overrides):
    """Build a minimal Hydra config for a plaintext dataset."""
    dataset = {
        'type': 'plaintext',
        'path': corpus_path,
        'cache_dir': str(tmp_path / "cache"),
        'dev_size': 0.5,
    }
    dataset.update(overrides)
    return OmegaConf.create({'dataset': dataset, 'seed': 1})


class TestResolve:
    """The build-or-load decision the entry point performs."""

    def test_builds_a_cache_at_the_reported_path(self, tmp_path, corpus):
        source = build_untokenized_source(plaintext_args(tmp_path, corpus))
        source.resolve()

        assert os.path.exists(source.path)

    def test_writes_the_config_record_beside_the_cache(self, tmp_path, corpus):
        source = build_untokenized_source(plaintext_args(tmp_path, corpus))
        source.resolve()

        with open(source.config_path) as handle:
            saved = yaml.safe_load(handle)
        assert saved == {'type': 'plaintext', 'path': corpus}

    def test_second_resolve_serves_the_cache(self, tmp_path, corpus):
        args = plaintext_args(tmp_path, corpus)
        first = build_untokenized_source(args)
        first.resolve()

        # a cache hit must not consult the corpus file
        os.remove(corpus)
        second = build_untokenized_source(args)

        assert second.resolve()['train']['text'] == first.resolve()['train']['text']

    def test_the_resolved_dataset_is_readable_from_the_path(self, tmp_path, corpus):
        source = build_untokenized_source(plaintext_args(tmp_path, corpus))
        source.resolve()

        assert len(load_from_disk(source.path)['train']) == 3


class TestConfigTracking:
    """A changed config must not silently reuse the old corpus."""

    def test_changed_source_raises(self, tmp_path, corpus):
        build_untokenized_source(plaintext_args(tmp_path, corpus)).resolve()

        other = tmp_path / "other.txt"
        other.write_text("entirely different text\n")
        # keep the same cache_dir so the two configs collide on one path
        changed = build_untokenized_source(plaintext_args(tmp_path, str(other)))

        with pytest.raises(ConfigMismatchError):
            changed.resolve()

    def test_changed_seed_does_not_invalidate_a_deterministic_source(self, tmp_path, corpus):
        """Reading a file start to finish cannot depend on the seed.

        The retired wrapper recorded `seed` for every dataset type, so bumping
        it invalidated caches nothing random had produced. Sources now record
        the seed only where it changes the result.
        """
        build_untokenized_source(plaintext_args(tmp_path, corpus)).resolve()

        args = plaintext_args(tmp_path, corpus)
        args.seed = 2

        assert build_untokenized_source(args).validate() is True

    def test_fresh_rebuilds_over_a_mismatch(self, tmp_path, corpus):
        build_untokenized_source(plaintext_args(tmp_path, corpus)).resolve()

        other = tmp_path / "other.txt"
        other.write_text("entirely different text\n")
        changed = build_untokenized_source(plaintext_args(tmp_path, str(other)))
        changed.resolve(fresh=True)

        with open(changed.config_path) as handle:
            assert yaml.safe_load(handle)['path'] == str(other)

    def test_cache_without_a_record_is_refused(self, tmp_path, corpus):
        source = build_untokenized_source(plaintext_args(tmp_path, corpus))
        source.resolve()
        os.remove(source.config_path)

        with pytest.raises(MissingConfigRecordError):
            build_untokenized_source(plaintext_args(tmp_path, corpus)).resolve()


class TestPathResolution:
    """The cache-directory rules the old __main__ glue encoded by hand."""

    def test_plain_dataset_lives_directly_under_cache_dir(self, tmp_path, corpus):
        source = build_untokenized_source(plaintext_args(tmp_path, corpus))

        assert source.path == os.path.join(str(tmp_path / "cache"), "untokenized")

    def _mix_args(self, tmp_path, corpus, alpha):
        return OmegaConf.create({
            'dataset': {
                'type': 'multinomial',
                'cache_dir': str(tmp_path / "cache"),
                'total_samples': 100,
                'alpha': alpha,
                'dev_size': 0.1,
                'sources': [
                    {'type': 'plaintext', 'id': 'a', 'path': corpus, 'sampling_prob': 0.6},
                    {'type': 'plaintext', 'id': 'b', 'path': corpus, 'sampling_prob': 0.4},
                ],
            },
            'seed': 1,
        })

    def test_multinomial_routes_into_a_mix_subfolder(self, tmp_path, corpus):
        source = build_untokenized_source(self._mix_args(tmp_path, corpus, 0.5))
        parent = str(tmp_path / "cache")

        assert source.path.startswith(os.path.join(parent, "mix_"))
        assert source.path.endswith("untokenized")

    def test_different_mixes_do_not_collide(self, tmp_path, corpus):
        first = build_untokenized_source(self._mix_args(tmp_path, corpus, 0.3))
        second = build_untokenized_source(self._mix_args(tmp_path, corpus, 0.7))

        assert first.path != second.path

    def test_dev_size_reaches_a_mix_from_the_deprecated_location(self, tmp_path, corpus):
        """`training.dev_size` still resolves, via the entry point's fallback."""
        args = self._mix_args(tmp_path, corpus, 0.5)
        del args.dataset.dev_size
        args.training = {'dev_size': 0.1}

        with pytest.warns(FutureWarning):
            source = build_untokenized_source(args)

        assert source.dev_size == 0.1


class TestSubstitutions:
    """A `substitutions` field relocates the source to a sibling directory."""

    def _substituted_args(self, tmp_path, corpus):
        return plaintext_args(
            tmp_path, corpus, substitutions=[{'pattern': 'the', 'replacement': 'THE'}],
        )

    def test_path_is_a_sibling_of_the_base_cache(self, tmp_path, corpus):
        source = build_untokenized_source(self._substituted_args(tmp_path, corpus))
        base = os.path.join(str(tmp_path / "cache"), "untokenized")

        assert source.path.startswith(f"{base}_sub_")

    def test_substitutions_are_actually_applied(self, tmp_path, corpus):
        source = build_untokenized_source(self._substituted_args(tmp_path, corpus))
        source.resolve()

        dataset = load_from_disk(source.path)
        assert all('the' not in text for text in dataset['train']['text'])

    def test_the_base_corpus_keeps_its_own_record(self, tmp_path, corpus):
        """Both caches exist and each describes itself."""
        source = build_untokenized_source(self._substituted_args(tmp_path, corpus))
        source.resolve()

        base = os.path.join(str(tmp_path / "cache"), "untokenized")
        with open(os.path.join(base, 'config.yaml')) as handle:
            assert 'substitutions' not in yaml.safe_load(handle)
        with open(source.config_path) as handle:
            assert yaml.safe_load(handle)['type'] == 'substituted'


class TestSeedPropagation:
    """`args.seed` must reach the sources that record it.

    The subsampling sources key their cache on the seed, so a seed that is only
    set in the global RNG would produce records claiming a value the run did
    not use.
    """

    def test_configured_seed_reaches_a_subsampling_source(self, tmp_path, corpus):
        from unittest.mock import patch

        from datasets import Dataset

        args = plaintext_args(tmp_path, corpus)
        args.seed = 77
        args.dataset.type = 'huggingface'
        args.dataset.name = 'fake/ds'
        args.dataset.max_samples = 2
        args.dataset.split_into_lines = False

        documents = Dataset.from_dict({'text': ["one two", "three four"]})
        with patch('lapt.sources.huggingface.load_dataset', return_value=documents):
            artifact = build_untokenized_source(args)
            artifact.resolve()

        with open(os.path.join(artifact.path, 'config.yaml')) as record:
            assert yaml.safe_load(record)['seed'] == 77
