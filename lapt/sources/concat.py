"""LAPT's binding of the shared concat composite to its source registry."""

from lapt.sources.base import SOURCE_TYPES
from lapt.sources.factory import make_source, normalize_sources
from lapt_core.composites import ConcatArtifact
from lapt_core.mixing import field, source_id


class ConcatDataset(ConcatArtifact):
    """Several sources concatenated into one training split.

    The concatenation itself is domain-neutral and lives in
    `lapt_core.composites`. This class supplies the two things that are LAPT's:
    the registry key, and `make_source` as the factory turning a configuration
    entry into a child -- which is what applies LAPT's `substitutions` wrapper
    and reaches its own source types, so a child may itself be composite.
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

    type_name = "concat"

    @classmethod
    def from_config(
        cls,
        cache_dir: str,
        source_config,
        seed: int = 1,
        dev_size: float | None = None,
    ) -> 'ConcatDataset':
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            source_config: Entry carrying `sources`.
            seed: Global random seed, passed to children that subsample.
            dev_size: Unused; only a mix holds out a dev split.

        Returns:
            The configured composite.
        """
        return cls(
            cache_dir,
            field(source_config, 'sources'),
            parent_id=source_id(source_config),
            seed=seed,
        )


SOURCE_TYPES.register(ConcatDataset)
