# `lapt-core`

Content-tracked caching and dataset-mixing primitives for multi-stage training
pipelines. Extracted from [LAPT](https://github.com/ur2nlp/LAPT) so sibling
projects depend on one implementation instead of a copy that drifts.

## Install

```
pip install "lapt-core @ git+https://github.com/ur2nlp/LAPT.git@main#subdirectory=packages/lapt-core"
```

Pin a tag rather than `main` for anything reproducible. Add the `datasets` extra
if you use the dataset layer:

```
pip install "lapt-core[datasets] @ git+https://github.com/ur2nlp/LAPT.git@<tag>#subdirectory=packages/lapt-core"
```

No PyPI account is involved, it works from a fork, and it is pinnable. It is
deliberately *not* an editable install of a local checkout: that bakes an
absolute machine path into the environment, so a student forking a consumer repo
on their own machine gets nothing.

## Modules

| Module | Depends on | What it holds |
|---|---|---|
| `artifacts` | stdlib, PyYAML | `CachedArtifact`, `ArtifactGraph`, `ArtifactConfig`, `config_digest`, `dict_diff` |
| `mixing` | stdlib, PyYAML | domain-neutral sampling/mixing vocabulary |
| `dataset_artifacts` | + `datasets` | `DatasetArtifact`, `DatasetRegistry` |
| `composites` | + `datasets` | concat/multinomial composites over `DatasetArtifact` |

The dependency split is per *module*, not per package: `__init__.py`
deliberately re-exports nothing, precisely so a consumer importing
`lapt_core.artifacts` never pays for `datasets`.

## Development

Working inside the LAPT repo, install this package before the root project so
pip resolves `lapt-core` locally rather than looking for it on an index:

```
pip install -e packages/lapt-core
pip install -e ".[dev]"
```

The test suite needs neither: `pythonpath` in the root `pyproject.toml` covers a
bare checkout.
