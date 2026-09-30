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

# What a config-mismatch message offers first for each kind of source. Written
# out per kind because the obvious flag is the wrong one: `fresh_dataset`
# removes the whole cache tree, so it rebuilds every source to fix one.
SOURCE_REBUILD_HINT = (
    "Rebuild this source alone by removing its directory (the one\n"
    "     holding the cached config above). Not fresh_dataset=true, which\n"
    "     removes every source under dataset.cache_dir"
)
SAMPLED_SOURCE_REBUILD_HINT = (
    "If only `seed` differs, this cache predates seed-keyed paths:\n"
    "     rename its directory to untokenized_seed<n>, n the cached seed, to\n"
    "     keep it as that seed's sample. Otherwise, rebuild this source alone\n"
    "     by removing its directory. Not fresh_dataset=true, which removes\n"
    "     every source under dataset.cache_dir"
)
MIX_REBUILD_HINT = (
    "Resample the mix with fresh_mix=true, which keeps the per-source\n"
    "     caches it draws on"
)
