"""LAPT's binding of the shared multinomial mix to its source registry."""

from lapt.sources.base import SOURCE_TYPES
from lapt.sources.factory import make_source, normalize_sources
from lapt_core.composites import MultinomialArtifact
from lapt_core.mixing import field


class MultinomialDataset(MultinomialArtifact):
    """Several sources sampled into one mix, with per-source dev splits.

    The sampling, the mix-keyed cache directory, and the split-before-upsample
    discipline are domain-neutral and live in `lapt_core.composites`. This class
    supplies LAPT's registry key and `make_source` as the child factory.
    """

    def __init__(self, cache_dir, sources, *args, child_factory=make_source, **kwargs):
        """Adapt LAPT's construction conventions to the shared composite.

        Two things happen here so that every construction route is safe, not
        just `from_config`: the child factory defaults to LAPT's, and `sources`
        are normalized to plain dicts. The latter matters because the core
        class records `dict(source)` in `config()` -- a shallow copy, which
        would leave a nested `ListConfig` in place if handed a `DictConfig`.
        `config_digest` serializes with `default=str`, so that would not raise;
        it would quietly produce a different digest, and a different cache path.
        """
        super().__init__(
            cache_dir, normalize_sources(sources), *args,
            child_factory=child_factory, **kwargs,
        )

    type_name = "multinomial"

    @classmethod
    def from_config(
        cls,
        cache_dir: str,
        source_config,
        seed: int = 1,
        dev_size: float | None = None,
    ) -> 'MultinomialDataset':
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Parent directory for the mix and the source caches.
            source_config: Entry carrying `sources`, `total_samples`, and
                `dev_size`, optionally `alpha`.
            seed: Global random seed.
            dev_size: Resolved default from the caller, used when the entry
                names none. This is how the deprecated `training.dev_size`
                fallback still reaches a mix.

        Returns:
            The configured mix.
        """
        return cls(
            cache_dir,
            field(source_config, 'sources'),
            field(source_config, 'alpha'),
            field(source_config, 'total_samples'),
            field(source_config, 'dev_size', dev_size),
            seed=seed,
        )


SOURCE_TYPES.register(MultinomialDataset)
