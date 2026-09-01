"""Construction of source artifacts from dataset configuration entries.

This replaces the `if/elif` chain that dispatched on a config's `type` field.
Each source class knows how to read its own parameters, via `from_config`, and
the registry maps the type name to the class — so adding a source type touches
one new file instead of a branch in a shared function.
"""

from typing import Any

from lapt.sources.base import SOURCE_TYPES, SourceDataset

DEFAULT_DATASET_TYPE = 'oscar'


def field(source_config: Any, name: str, default: Any = None) -> Any:
    """Read one field from a source configuration.

    Accepts both `DictConfig` and plain dicts, since sources arrive as either
    depending on whether they came from Hydra or from a parent's `sources` list.

    Args:
        source_config: The configuration entry.
        name: Field to read.
        default: Value to return when the field is absent.

    Returns:
        The field's value, or `default`.
    """
    if isinstance(source_config, dict):
        value = source_config.get(name, default)
    else:
        value = getattr(source_config, name, default)
    return default if value is None and default is not None else value


def source_type(source_config: Any) -> str:
    """Return a configuration's dataset type, defaulting for older configs.

    Args:
        source_config: The configuration entry.

    Returns:
        The `type` field, or `oscar` when absent, which is what configs
        predating the type field meant.
    """
    return field(source_config, 'type', DEFAULT_DATASET_TYPE)


def build_source(cache_dir: str, source_config: Any, seed: int = 1) -> SourceDataset:
    """Construct the source artifact a configuration entry describes.

    Args:
        cache_dir: Directory the source's `untokenized` subdirectory goes in.
        source_config: The configuration entry, carrying at least `type`.
        seed: Global random seed, passed to sources that subsample.

    Returns:
        An unresolved source artifact.

    Raises:
        ValueError: If no source type is registered under the config's `type`.
    """
    return SOURCE_TYPES.get(source_type(source_config)).from_config(
        cache_dir, source_config, seed
    )
