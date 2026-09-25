# `lapt_core.registry`

Extract, annotate, compare and audit experiment run records.

Everything project-specific is a `RegistrySchema`: the environment-variable
prefix and the three tables saying which config sections to flatten, which keys
to drop, and how to order what is left. A project supplies one and calls
`main()`; see `tools/registry.py` in this repository for a worked example.

Needs no extra — standard library plus PyYAML, same as `lapt_core.artifacts`.

::: lapt_core.registry
