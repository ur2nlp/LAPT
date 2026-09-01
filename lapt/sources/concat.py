"""Composite source that concatenates several other sources end to end."""

import os
import sys

from datasets import DatasetDict, concatenate_datasets
from omegaconf import DictConfig, OmegaConf

from lapt.sources.base import SOURCE_TYPES, SourceDataset
from lapt.sources.factory import build_source, field


def source_id(source_config, fallback: str | None = None) -> str:
    """Return a source's cache identifier.

    Checks `id`, then the deprecated `language`, then the fallback.

    Args:
        source_config: The configuration entry.
        fallback: Identifier to use when the entry names none.

    Returns:
        The identifier.
    """
    identifier = field(source_config, 'id')
    if not identifier:
        identifier = field(source_config, 'language')
        if identifier:
            print(
                f"Warning: 'language' field for source identification is deprecated, "
                f"use 'id' instead (found language='{identifier}')",
                file=sys.stderr,
            )
    return identifier or fallback


class ConcatDataset(SourceDataset):
    """Several sources concatenated into one training split.

    Children are resolved through the same registry as any other source, so a
    child may itself be composite. Each child caches under its own
    subdirectory of this source's cache directory, keyed by its id — which is
    what lets two mixes referencing the same source id share one copy of it.

    The child set is read from the configuration at build time rather than
    declared statically, so these dependencies are deliberately resolved inside
    `build` rather than through an `ArtifactGraph`.
    """

    type_name = "concat"

    def __init__(self, cache_dir: str, sources: list, parent_id: str | None = None, seed: int = 1):
        """Initialize the composite.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in, and
                the parent of each child's own cache directory.
            sources: Configuration entries for the sources to concatenate.
            parent_id: This source's own id, used to name unnamed children.
            seed: Global random seed, passed to children that subsample.

        Raises:
            ValueError: If `sources` is empty.
        """
        super().__init__(cache_dir)
        if not sources:
            raise ValueError("Cannot concatenate datasets: sources list is empty")
        self.sources = sources
        self.parent_id = parent_id
        self.seed = seed

    @classmethod
    def from_config(cls, cache_dir: str, source_config, seed: int = 1) -> 'ConcatDataset':
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            source_config: Entry carrying `sources`.
            seed: Global random seed, passed to children.

        Returns:
            The configured composite.
        """
        return cls(
            cache_dir,
            field(source_config, 'sources'),
            parent_id=source_id(source_config),
            seed=seed,
        )

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        The children's full configurations are recorded, so a change anywhere
        in the tree invalidates this cache. The seed is deliberately absent:
        concatenation preserves order and samples nothing, and any child that
        does sample records the seed itself.
        """
        return {
            'type': 'concat',
            'sources': [
                OmegaConf.to_container(DictConfig(source), resolve=True)
                for source in self.sources
            ],
        }

    def children(self) -> list[tuple[str, SourceDataset]]:
        """Build the child sources, unresolved, paired with their ids.

        Returns:
            A list of `(source_id, source)` in configuration order.
        """
        built = []
        for index, source_config in enumerate(self.sources):
            default_id = f"{self.parent_id}_{index}" if self.parent_id else f"source_{index}"
            child_id = source_id(source_config, fallback=default_id)
            child = build_source(
                os.path.join(self.root, child_id), source_config, self.seed
            )
            built.append((child_id, child))
        return built

    def build(self, deps) -> DatasetDict:
        """Resolve each child and concatenate their training splits.

        Args:
            deps: Unused; children are resolved here rather than injected,
                since the child set is only known from the configuration.

        Returns:
            A `DatasetDict` with a single `train` split.
        """
        print(f"Concatenating {len(self.sources)} dataset sources", file=sys.stderr)

        to_concat = []
        for index, (child_id, child) in enumerate(self.children()):
            child_data = child.resolve()
            to_concat.append(child_data['train'])
            print(
                f"  Source {index} ({child_id}): {len(child_data['train'])} examples",
                file=sys.stderr,
            )

        concatenated = concatenate_datasets(to_concat)
        print(f"  Concatenated to {len(concatenated)} total examples", file=sys.stderr)
        return DatasetDict({'train': concatenated})


SOURCE_TYPES.register(ConcatDataset)
