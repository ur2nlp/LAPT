"""Construction of source artifacts from dataset configuration entries.

Named `make_source` rather than `build_source` because `build()` on an artifact
means something else entirely: this returns an unresolved *object*, while
`build()` produces the *dataset contents*. Nothing here touches disk.

This replaces the `if/elif` chain that dispatched on a config's `type` field.
Each source class knows how to read its own parameters, via `from_config`, and
the registry maps the type name to the class — so adding a source type touches
one new file instead of a branch in a shared function.
"""

from typing import Any

from omegaconf import DictConfig, OmegaConf

from lapt.sources.base import SOURCE_TYPES
from lapt.sources.substituted import SubstitutedDataset, parse_substitutions
from lapt_core.dataset_artifacts import DatasetArtifact
from lapt_core.mixing import field

DEFAULT_DATASET_TYPE = 'oscar'


def normalize_sources(sources) -> list[dict]:
    """Unwrap a composite's `sources` list into plain dicts.

    The single omegaconf boundary for composites. `lapt_core.composites` takes
    plain dicts so it need not depend on Hydra; converting here means it happens
    once, at construction, rather than every time a `config()` record is built.

    Args:
        sources: Entries from a configuration, as `DictConfig`, `ListConfig`,
            or plain dicts.

    Returns:
        The same entries as plain dicts, with interpolations resolved.
    """
    return [
        OmegaConf.to_container(DictConfig(source), resolve=True)
        for source in (sources or [])
    ]


def source_type(source_config: Any) -> str:
    """Return a configuration's dataset type, defaulting for older configs.

    Args:
        source_config: The configuration entry.

    Returns:
        The `type` field, or `oscar` when absent, which is what configs
        predating the type field meant.
    """
    return field(source_config, 'type', DEFAULT_DATASET_TYPE)


def make_source(
    cache_dir: str,
    source_config: Any,
    seed: int = 1,
    dev_size: float | None = None,
) -> DatasetArtifact:
    """Construct the source artifact a configuration entry describes.

    A `substitutions` field wraps the result in a `SubstitutedDataset`, so the
    transformation applies to any source type and reaches nested sources
    identically -- a mix's children carry their own substitutions, and this is
    the single place that is honored.

    Args:
        cache_dir: Directory the source's `untokenized` subdirectory goes in.
        source_config: The configuration entry, carrying at least `type`.
        seed: Global random seed, passed to sources that subsample.
        dev_size: Resolved dev-split default, used by a mix that names none.

    Returns:
        An unresolved source artifact.

    Raises:
        ValueError: If no source type is registered under the config's `type`,
            or if a substitution entry is malformed.
    """
    base = SOURCE_TYPES.get(source_type(source_config)).from_config(
        cache_dir, source_config, seed, dev_size
    )

    substitutions = parse_substitutions(field(source_config, 'substitutions'))
    if substitutions:
        return SubstitutedDataset(base, substitutions)
    return base
