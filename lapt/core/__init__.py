"""Framework-general building blocks shared across LAPT-style training pipelines.

Nothing in this subpackage is specific to language-adaptive pre-training, to
text, or to any one corpus. It exists so that sibling projects (currently the
ASR adaptation repo) can depend on one implementation of the caching and
config-tracking machinery instead of maintaining a copy that drifts.

As with `lapt` itself, this module deliberately re-exports nothing: importing a
submodule here would pull the heavier dependencies of whichever module happened
to be listed first. Import what you need directly:
`from lapt.core.artifacts import CachedArtifact`.
"""
