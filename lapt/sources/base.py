"""LAPT's registry of untokenized corpus sources.

A *source* is one entry resolvable from a dataset configuration's `type` field:
a plaintext file, a HuggingFace dataset, or a composite of other sources. Each
is a `DatasetArtifact`, so the cache-or-build decision, the config record, and
the round trip to disk come from `lapt_core`; a concrete source supplies only
the parameters it is keyed on and the code that produces the dataset.

Concrete types subclass `DatasetArtifact` directly and register themselves
here. There is deliberately no LAPT-specific base class in between: the one
that used to sit here, `SourceDataset`, existed solely to refuse caches written
before sources became artifacts, and was removed once the cache tree carried
the current record everywhere.
"""

from lapt_core.dataset_artifacts import DatasetRegistry

SOURCE_TYPES = DatasetRegistry()
