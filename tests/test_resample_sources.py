"""Tests for `resample_sources`, which opts subsampling sources into the global seed."""

from omegaconf import OmegaConf

from lapt.sources.factory import resample_sources


def mix_config(**c4_overrides):
    """A mix with one capped source and one uncapped source."""
    return OmegaConf.create({
        'type': 'multinomial',
        'sources': [
            {'type': 'instruction_jsonl', 'id': 'got', 'path': 'x.jsonl'},
            {'type': 'huggingface', 'id': 'eng', 'name': 'allenai/c4', 'max_samples': 10,
             **c4_overrides},
        ],
    })


class TestResampleSources:
    def test_capped_source_is_reseeded(self):
        dataset = mix_config()
        assert resample_sources(dataset, 2) == ['eng']
        assert dataset.sources[1].sample_seed == 2

    def test_uncapped_source_is_left_alone(self):
        """Nothing about it is random, so a key would only perturb cache records."""
        dataset = mix_config()
        resample_sources(dataset, 2)
        assert 'sample_seed' not in dataset.sources[0]

    def test_explicit_sample_seed_wins(self):
        dataset = mix_config(sample_seed=5)
        assert resample_sources(dataset, 2) == []
        assert dataset.sources[1].sample_seed == 5

    def test_default_seed_changes_nothing(self):
        dataset = mix_config()
        assert resample_sources(dataset, 1) == []
        assert 'sample_seed' not in dataset.sources[1]

    def test_nested_sources_are_reached(self):
        dataset = OmegaConf.create({'type': 'concat', 'sources': [mix_config()]})
        resample_sources(dataset, 3)
        assert dataset.sources[0].sources[1].sample_seed == 3

    def test_a_top_level_capped_source_is_reseeded(self):
        dataset = OmegaConf.create({'type': 'huggingface', 'name': 'c4', 'max_samples': 10})
        resample_sources(dataset, 2)
        assert dataset.sample_seed == 2
