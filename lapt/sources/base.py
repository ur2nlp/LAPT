"""LAPT's base class and registry for untokenized corpus sources.

A *source* is one entry resolvable from a dataset configuration's `type` field:
a plaintext file, a HuggingFace dataset, or a composite of other sources. Each
is a `DatasetArtifact`, so the cache-or-build decision, the config record, and
the round trip to disk come from `lapt_core`; a concrete source supplies only
the parameters it is keyed on and the code that produces the dataset.

`SourceDataset` adds one LAPT-specific concern: detecting caches written before
sources became artifacts. Those carry a differently named config record, which
would otherwise read as "no record at all" and be accepted silently.

That concern is temporary, and so is this class. It is a tripwire, not a
compatibility shim — it refuses pre-refactor caches rather than reading them, so
nothing is lost by removing it once none remain. Delete it when every source
type has been converted and the cache tree has been regenerated, at which point
`SourceDataset` has no content left: `name` duplicates the default it inherits
and `__init__` only renames a parameter. Concrete types then subclass
`DatasetArtifact` directly and this module keeps only `SOURCE_TYPES`.
"""

import os

from lapt_core.artifacts import ConfigMismatchError
from lapt_core.dataset_artifacts import DatasetArtifact, DatasetRegistry

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

        Sources tracked their parameters in `source_config.yaml` before they
        became artifacts; they now use `config.yaml`. The base implementation
        would find no record under the name it expects, treat the cache as
        predating config tracking altogether, and accept it — turning a stale
        corpus into a warning rather than an error. Detect that case instead.

        Such a cache is not untracked and not unusable: its record is usually
        the current one minus fields added since. Refusing it here is a
        deliberate choice to keep that reconciliation in a one-off migration
        rather than a tolerance mechanism carried in the code forever.

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
                f"This cache carries {LEGACY_CONFIG_FILENAME}, the record used\n"
                f"before sources became artifacts. Its parameters are tracked,\n"
                f"but under the previous filename and without any field added\n"
                f"since it was written, so it cannot be validated as-is and\n"
                f"must not be reused silently.\n\n"
                f"Either regenerate it by passing fresh_dataset=true, or -- if\n"
                f"rebuilding is expensive -- migrate the record: rename it to\n"
                f"config.yaml and add the fields this source now tracks.\n"
                f"{'=' * 70}\n"
            )

        return super().validate(error_on_mismatch=error_on_mismatch)
