"""Construction of source artifacts from dataset configuration entries.

Named `make_source` rather than `build_source` because `build()` on an artifact
means something else entirely: this returns an unresolved *object*, while
`build()` produces the *dataset contents*. Nothing here touches disk.

This replaces the `if/elif` chain that dispatched on a config's `type` field.
Each source class knows how to read its own parameters, via `from_config`, and
the registry maps the type name to the class — so adding a source type touches
one new file instead of a branch in a shared function.
"""

import os
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from lapt.sources.base import SOURCE_TYPES
from lapt.sources.substituted import SubstitutedDataset, parse_substitutions
from lapt_core.dataset_artifacts import DatasetArtifact
from lapt_core.mixing import DEFAULT_SEED, field

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


def sample_seed(source_config: Any) -> int:
    """Return the seed a subsampling source draws its sample with.

    Deliberately not the global `seed`. A seed replicate varies how a model
    trains -- the mix, the plan shuffle, dropout -- and should not also re-draw
    a multi-million-line corpus sample unless asked to, since that is both a
    different experiment and an expensive download. A source opts in with its
    own `sample_seed`, which `resample_sources=true` fills in from the global
    seed.

    Args:
        source_config: The configuration entry.

    Returns:
        The entry's `sample_seed`, or the default seed when absent.
    """
    return field(source_config, 'sample_seed', DEFAULT_SEED)


def seed_keyed_path(artifact: DatasetArtifact, max_samples: int | None, seed: int) -> str:
    """Return a subsampling source's cache directory, keyed on its seed.

    A sample drawn at a non-default seed caches beside the default one, as
    `untokenized_seed<n>`, so switching seeds builds a sibling rather than
    colliding with -- or requiring the deletion of -- the existing sample. The
    default seed keeps the bare name, which is what keeps every cache built
    before seed-keying addressed. A source without `max_samples` draws nothing
    at random, so its path never carries a seed.

    The tokenized caches derived from this directory pick the suffix up from
    its name, so they are keyed on the seed without further work.

    Args:
        artifact: The source, supplying `root` and `name`.
        max_samples: The source's subsample cap, or None if it keeps everything.
        seed: The seed the subsample is drawn with.

    Returns:
        The cache directory.
    """
    if max_samples is None or seed == DEFAULT_SEED:
        return os.path.join(artifact.root, artifact.name)
    return os.path.join(artifact.root, f"{artifact.name}_seed{seed}")


def resample_sources(dataset_config: DictConfig, seed: int) -> list[str]:
    """Draw every subsampling source at the global seed, unless it pins its own.

    Implements `resample_sources=true`. Sets `sample_seed` to `seed` on each
    entry, at any depth, that has `max_samples` and no explicit `sample_seed`.
    Writing it into the configuration, rather than threading a flag through to
    the sources, means the run's saved config records which sample it trained
    on, and every cache keyed on the dataset spec sees the difference.

    At the default seed this changes nothing: the sources already draw at that
    seed, and leaving the entries untouched keeps every cache record and mix
    slug built without the key addressed.

    Args:
        dataset_config: The `dataset` node of the Hydra config, edited in place.
        seed: The global seed.

    Returns:
        Identifiers of the entries that were reseeded, for logging.
    """
    if seed == DEFAULT_SEED:
        return []

    reseeded = []

    def visit(node: DictConfig) -> None:
        if node.get('max_samples') is not None and node.get('sample_seed') is None:
            with open_dict(node):
                node.sample_seed = seed
            reseeded.append(node.get('id') or node.get('language') or node.get('name'))
        for child in node.get('sources') or []:
            visit(child)

    visit(dataset_config)
    return reseeded


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
