"""LAPT's base class and registry for untokenized corpus sources.

A *source* is one entry resolvable from a dataset configuration's `type` field:
a plaintext file, a HuggingFace dataset, or a composite of other sources. Each
is a `DatasetArtifact`, so the cache-or-build decision, the config record, and
the round trip to disk come from `lapt.core`; a concrete source supplies only
the parameters it is keyed on and the code that produces the dataset.

`SourceDataset` adds one LAPT-specific concern: detecting caches written before
sources became artifacts. Those carry a differently named config record, which
would otherwise read as "no record at all" and be accepted silently.
"""

import os

from lapt.core.artifacts import ConfigMismatchError
from lapt.core.dataset_artifacts import DatasetArtifact, DatasetRegistry

LEGACY_CONFIG_FILENAME = "source_config.yaml"

SOURCE_TYPES = DatasetRegistry()


class SourceDataset(DatasetArtifact):
    """An untokenized corpus source, cached under `<cache_dir>/untokenized`.

    Subclasses implement `config()` — the parameters the cache is keyed on,
    which double as the record validated against on reuse — and `build()`,
    which returns the dataset.

    Attributes:
        name: Directory name, fixed at `untokenized` so that a source's cache
            layout is independent of its type.
    """

    name = "untokenized"

    def __init__(self, cache_dir: str):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory is created in.
                For a composite's children this is a per-child subdirectory, so
                sources with the same id are shared across mixes.
        """
        super().__init__(cache_dir)

    def validate(self, error_on_mismatch: bool = True) -> bool:
        """Validate a cached source, refusing pre-artifact caches explicitly.

        A cache written before this refactor carries `source_config.yaml`
        instead of `config.yaml`. The base implementation would find no record
        under the name it expects, treat the cache as predating config tracking,
        and accept it — turning a stale corpus into a warning rather than an
        error. Detect that case and say what to do about it instead.

        Args:
            error_on_mismatch: Raise on mismatch when True.

        Returns:
            True when the cached config matches or the artifact does not exist.

        Raises:
            ConfigMismatchError: On a parameter mismatch, or on a cache
                predating the artifact refactor.
        """
        legacy_path = os.path.join(self.path, LEGACY_CONFIG_FILENAME)
        if os.path.exists(legacy_path) and not os.path.exists(self.config_path):
            raise ConfigMismatchError(
                f"\n{'=' * 70}\n"
                f"PRE-REFACTOR SOURCE CACHE: {self.path}\n"
                f"{'=' * 70}\n"
                f"This cache carries {LEGACY_CONFIG_FILENAME}, written before\n"
                f"sources became tracked artifacts. Its parameters are recorded\n"
                f"under different keys than the current ones, so it cannot be\n"
                f"validated and must not be reused silently.\n\n"
                f"Regenerate it by passing fresh_dataset=true, or delete the\n"
                f"directory above.\n"
                f"{'=' * 70}\n"
            )

        return super().validate(error_on_mismatch=error_on_mismatch)
