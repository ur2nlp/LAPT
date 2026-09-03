"""Framework-general building blocks shared across LAPT-style training pipelines.

Nothing here is specific to language-adaptive pre-training, to text, or to any
one corpus. It exists so that sibling projects can depend on one implementation
of the caching and config-tracking machinery instead of maintaining a copy that
drifts.

Deliberately a *top-level* package rather than `lapt.core`, so that it can be
distributed on its own as `lapt-core` without `lapt` having to become a
namespace package, and so that a consumer installs two dependencies rather than
LAPT's full training stack. A project depending on this should not inherit a
pin on torch or transformers.

The module re-exports nothing: importing a submodule here would pull the heavier
dependencies of whichever one happened to be listed first, and only
`dataset_artifacts` needs them. Import what you need directly:
`from lapt_core.artifacts import CachedArtifact`.
"""
