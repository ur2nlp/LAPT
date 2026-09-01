"""Behavioral tests for UntokenizedDataset, the first stage ported to CachedArtifact.

These exercise the real dispatcher against on-disk plaintext corpora rather than
mocking it, since the point of the port is that path resolution, config
tracking, and building stay consistent with each other.
"""

import os

import pytest
import yaml
from omegaconf import OmegaConf

from lapt.core.artifacts import ConfigMismatchError, MissingConfigRecordError
from lapt.dataset_utils import UntokenizedDataset


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
    """The build-or-load decision the port moved out of __main__."""

    def test_builds_and_returns_a_real_path(self, tmp_path, corpus):
        artifact = UntokenizedDataset(plaintext_args(tmp_path, corpus))
        path = artifact.resolve()
        assert os.path.exists(path)
        assert path == artifact.path

    def test_writes_the_config_record_beside_the_cache(self, tmp_path, corpus):
        artifact = UntokenizedDataset(plaintext_args(tmp_path, corpus))
        artifact.resolve()
        with open(artifact.config_path) as handle:
            saved = yaml.safe_load(handle)
        assert saved['type'] == 'plaintext'
        assert saved['path'] == corpus
        assert saved['seed'] == 1

    def test_second_resolve_returns_the_same_path(self, tmp_path, corpus):
        args = plaintext_args(tmp_path, corpus)
        first = UntokenizedDataset(args).resolve()
        second = UntokenizedDataset(args).resolve()
        assert first == second

    def test_cached_dataset_is_readable(self, tmp_path, corpus):
        from datasets import load_from_disk

        path = UntokenizedDataset(plaintext_args(tmp_path, corpus)).resolve()
        dataset = load_from_disk(path)
        assert len(dataset['train']) > 0


class TestConfigTracking:
    """A changed config must not silently reuse the old corpus."""

    def test_changed_source_raises(self, tmp_path, corpus):
        UntokenizedDataset(plaintext_args(tmp_path, corpus)).resolve()

        other = tmp_path / "other.txt"
        other.write_text("entirely different text\n")
        # keep the same cache_dir so the two configs collide on one path
        changed = UntokenizedDataset(plaintext_args(tmp_path, str(other)))

        with pytest.raises(ConfigMismatchError) as exc_info:
            changed.resolve()
        assert "Untokenized Dataset" in str(exc_info.value)

    def test_changed_seed_raises(self, tmp_path, corpus):
        UntokenizedDataset(plaintext_args(tmp_path, corpus)).resolve()
        args = plaintext_args(tmp_path, corpus)
        args.seed = 2
        with pytest.raises(ConfigMismatchError):
            UntokenizedDataset(args).resolve()

    def test_fresh_rebuilds_over_a_mismatch(self, tmp_path, corpus):
        UntokenizedDataset(plaintext_args(tmp_path, corpus)).resolve()
        args = plaintext_args(tmp_path, corpus)
        args.seed = 2
        path = UntokenizedDataset(args).resolve(fresh=True)
        assert os.path.exists(path)
        with open(UntokenizedDataset(args).config_path) as handle:
            assert yaml.safe_load(handle)['seed'] == 2

    def test_cache_without_a_record_is_refused(self, tmp_path, corpus):
        # an unverifiable cache is refused rather than reused; the leaf source
        # holds its own record, so removing this one leaves the wrapper blind
        artifact = UntokenizedDataset(plaintext_args(tmp_path, corpus))
        artifact.resolve()
        os.remove(artifact.config_path)

        with pytest.raises(MissingConfigRecordError):
            UntokenizedDataset(plaintext_args(tmp_path, corpus)).resolve()


class TestPathResolution:
    """The layout rules the old __main__ glue encoded by hand."""

    def test_plain_dataset_lives_directly_under_cache_dir(self, tmp_path, corpus):
        args = plaintext_args(tmp_path, corpus)
        artifact = UntokenizedDataset(args)
        assert artifact.path == os.path.join(str(tmp_path / "cache"), "untokenized")

    def test_multinomial_routes_into_a_mix_subfolder(self, tmp_path, corpus):
        args = OmegaConf.create({
            'dataset': {
                'type': 'multinomial',
                'cache_dir': str(tmp_path / "cache"),
                'total_samples': 100,
                'alpha': 0.5,
                'dev_size': 0.1,
                'sources': [
                    {'type': 'plaintext', 'id': 'a', 'path': corpus, 'sampling_prob': 0.6},
                    {'type': 'plaintext', 'id': 'b', 'path': corpus, 'sampling_prob': 0.4},
                ],
            },
            'seed': 1,
        })
        artifact = UntokenizedDataset(args)
        parent = str(tmp_path / "cache")
        assert artifact.path != os.path.join(parent, "untokenized")
        assert artifact.path.startswith(os.path.join(parent, "mix_"))
        assert artifact.path.endswith("untokenized")

    def test_different_mixes_do_not_collide(self, tmp_path, corpus):
        def mix_args(alpha):
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

        assert UntokenizedDataset(mix_args(0.3)).path != UntokenizedDataset(mix_args(0.7)).path


class TestSubstitutions:
    """Substitutions relocate the returned path to a sibling of the cache dir."""

    def test_returned_path_is_the_substituted_sibling(self, tmp_path, corpus):
        args = plaintext_args(
            tmp_path,
            corpus,
            substitutions=[{'pattern': 'the', 'replacement': 'THE'}],
        )
        artifact = UntokenizedDataset(args)
        path = artifact.resolve()

        assert path != artifact.path
        assert path.startswith(f"{artifact.path}_sub_")
        assert os.path.exists(path)

    def test_substitutions_are_actually_applied(self, tmp_path, corpus):
        from datasets import load_from_disk

        args = plaintext_args(
            tmp_path,
            corpus,
            substitutions=[{'pattern': 'the', 'replacement': 'THE'}],
        )
        dataset = load_from_disk(UntokenizedDataset(args).resolve())
        assert all('the' not in row['text'] for row in dataset['train'])

    def test_config_record_still_describes_the_base_corpus(self, tmp_path, corpus):
        # DatasetConfig deliberately does not track substitutions; the raw cache
        # is what it describes, and the _sub_ copy carries its own tracking
        args = plaintext_args(
            tmp_path,
            corpus,
            substitutions=[{'pattern': 'the', 'replacement': 'THE'}],
        )
        artifact = UntokenizedDataset(args)
        artifact.resolve()
        with open(artifact.config_path) as handle:
            assert 'substitutions' not in yaml.safe_load(handle)
