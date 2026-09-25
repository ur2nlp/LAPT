# LAPT: Language-Adaptive Pretraining Framework

[![CI](https://github.com/ur2nlp/LAPT/actions/workflows/ci.yml/badge.svg)](https://github.com/ur2nlp/LAPT/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs](https://img.shields.io/badge/docs-ur2nlp.io%2FLAPT-blue.svg)](https://ur2nlp.io/LAPT/)

A modular framework for continued pre-training of multilingual language models with flexible data loading and optional vocabulary specialization via [FOCUS](https://github.com/konstantinjdobler/focus).

## Features

- **Flexible dataset loading**: OSCAR corpus, local plaintext files, directory-based loading, concatenation, temperature-scaled multinomial sampling, and instruction-tuning data with prompt-token loss masking
- **FOCUS integration**: Optional vocabulary specialization with new tokenizer training using SentencePiece
- **Hydra configuration**: Composable YAML configs for easy experimentation
- **Per-language evaluation**: Automatic per-language dev set tracking for multilingual training
- **Tracked, resumable caches**: Every pipeline stage records the configuration that produced it and refuses to reuse a cache that does not match — see [Caching](#caching)

## Installation

Create a conda environment from the provided configuration:

```bash
conda env create -f environment.yml
conda activate lapt
```

To install the framework itself as a package, install the vendored `lapt-core`
distribution first so pip resolves it from the working tree instead of looking
for it on an index:

```bash
pip install -e packages/lapt-core
pip install -e ".[dev]"
```

Neither step is needed to run the test suite, which reaches both packages
through `pythonpath`.

### Using `lapt-core` elsewhere

`lapt_core` is a separate distribution so sibling projects can share the caching
layer without inheriting this project's `transformers` pin:

```bash
pip install "lapt-core[datasets] @ https://github.com/ur2nlp/LAPT/archive/refs/tags/<tag>.tar.gz#subdirectory=packages/lapt-core"
```

A release tarball rather than `git+https://`, because pip needs the `git`
binary to clone a VCS URL and a cluster compute node may not have one. See
`packages/lapt-core/README.md` for the git form, which is what a private
repository needs.

## Usage

### Basic Training

Train on OSCAR corpus for a single language:

```bash
python -m lapt dataset.type=oscar dataset.language=hy
```

### FOCUS Training

Train with vocabulary specialization:

```bash
python -m lapt focus.enabled=true focus.vocab_size=32768 focus.num_samples=1000000
```

### Multinomial Sampling

Train on multiple languages with alpha-scaled sampling (supports OSCAR, plaintext, or mixed sources):

```yaml
# configs/dataset/multilingual.yaml
type: multinomial
alpha: 0.7
total_samples: 1000000
sources:
  - type: oscar
    language: hu
  - type: plaintext
    path: /path/to/fi_corpus.txt
  - type: oscar
    language: et
cache_dir: data/uralic_mix
```

```bash
python -m lapt dataset=multilingual
```

### Local Data

Train on your own plaintext files:

```bash
python -m lapt dataset.type=plaintext dataset.path=/path/to/data.txt
```

## Caching

Each expensive stage — the untokenized corpus, the tokenizer, the tokenized
dataset — is an *artifact*: it owns its cache directory, records the
configuration that produced it in a YAML file beside the data, and decides for
itself whether to load or rebuild.

Two consequences worth knowing before you run anything twice:

- **Most config changes need no flags.** Cache paths encode the parameters that
  distinguish one result from another, so changing the vocabulary size (say)
  builds a new tokenizer in its own directory and leaves the old one and the
  corpus untouched. Configurations coexist rather than overwrite.
- **A changed parameter that *isn't* in the path is an error, not a silent
  reuse.** You will be told what differs and pointed at the flag that rebuilds
  that stage.

Selective rebuilds, each clearing everything downstream of it:

| flag | clears |
|---|---|
| `fresh_dataset=true` | the dataset cache tree outright, sources included — use when you do not trust what is on disk |
| `fresh_mix=true` | a multinomial or concat mix, *keeping* the per-source caches it draws on |
| `fresh_tokenizer=true` | the tokenizer, the tokenized data, and the model |
| `fresh_model=true` | model checkpoints only |

## Experiment Tracking

Runs are tracked by **experiment id** — the `experiment_id` you pass at launch,
which also names the output directory. Everything below keys on it.

```bash
python -m lapt experiment_id=<your_run_id> training.learning_rate=4e-5
```

Three files per run live under `outputs/`:

| path | holds |
|---|---|
| `outputs/configs/{id}.yaml` | the config the run was launched with |
| `outputs/trainer_states/{id}.json` | the metric history HuggingFace wrote |
| `outputs/registry.yaml` | one row per run: extracted params plus your notes |

### Pulling runs off a cluster

`fetch_results.sh` inventories the remote, works out what is missing or stale,
and copies only that. No host or path is baked into the repository, so set them
in your shell:

```bash
export LAPT_REMOTE=<ssh_host_or_alias>        # e.g. a Host entry in ~/.ssh/config
export LAPT_MODEL_DIRS=<remote_models_dir>    # colon-separated for several roots
```

```bash
bash scripts/fetch_results.sh
```

Finished runs are never re-fetched, and an in-progress run has only its
trainer state refreshed — a config cannot change mid-run. To see what it would
do without moving anything:

```bash
INV="ssh $LAPT_REMOTE 'bash -s' < tools/remote_inventory.sh -- -b $LAPT_MODEL_DIRS"
eval "$INV" | python tools/fetch_diff.py --dry-run
```

### Registering and annotating runs

`extract` reads configs and upserts a row per run. It is safe to re-run; it
updates rather than duplicates.

```bash
python tools/registry.py extract outputs/configs/<your_run_id>.yaml

# --pattern takes a regex over paths; this takes every id starting with 'lr'
python tools/registry.py extract --pattern 'outputs/configs/lr.*\.yaml'
```

The parameters come from the config automatically. What only you can supply is
why the run existed and what it showed:

```bash
python tools/registry.py annotate <your_run_id> \
    --note "lr 4e-5, 32k adapted vocabulary, effective batch 60" \
    --observation "best held-out bpc in this sweep; larger model plateaus above it" \
    --era <your_era> --group <your_group>
```

`--status manually_closed` retires a run, which also stops `fetch_results.sh`
re-fetching it.

### Reading the registry

```bash
python tools/registry.py show                      # everything
python tools/registry.py show --era <your_era> --group <your_group>
python tools/registry.py diff lr2e-5 lr4e-5        # only what differs (ids are examples)
python tools/registry.py verify                    # rows still match outputs/configs/
python tools/registry.py debt                      # runs on disk with no row, rows with no note
```

`diff` is the one to reach for when comparing a sweep: it prints only the
parameters that vary across the runs you name and lists the rest as constant, so
a forty-field config collapses to the three things you actually changed.

`debt --strict` exits non-zero, which makes it usable as a pre-commit or CI check
that no run went un-annotated.

### Plotting

`training_plot.py` reads trainer states directly — no registry required.

```bash
# one run, several metrics
python tools/training_plot.py --metrics loss eval_loss \
    --state-file outputs/trainer_states/<your_run_id>.json

# compare runs; --state-pattern is a regex over paths (ids here are examples)
python tools/training_plot.py --metric "eval_.*_bpc" \
    --state-pattern "outputs/trainer_states/lr(2|4)e-5\.json"

# discover what a run actually logged
python tools/training_plot.py --list-metrics --state-file outputs/trainer_states/<your_run_id>.json
```

Metric names are regexes, so `--metric "eval_.*_bpc"` draws every per-language
bpc series on one panel and `--metrics loss "eval_.*"` gives one panel per match.

Useful when the defaults fight you:

| flag | does |
|---|---|
| `--output plot.png` | save instead of opening a window |
| `--ylim 0 5` | shared y-limits across panels |
| `--ylims eval_loss:1:3` | per-metric limits; repeatable, wins over `--ylim` |
| `--run-names baseline adapted` | legend labels instead of file paths |
| `--exclude-pattern` | drop runs the state pattern swept up |
| `--x-axis epoch` | plot against epochs rather than steps |
| `--dark` | light-on-dark, for slides |

## Project Structure

- `lapt/` - Framework source (installable package)
  - `__main__.py` - Training orchestration
  - `sources/` - One module per dataset type, registered by its `type` field
  - `tokenized_data.py` - Text-to-token transformations and the cached
    tokenized stages that run them
  - `artifact_configs.py` - Per-stage configuration records and cache paths
  - `model.py` - Model loading and vocabulary adaptation
  - `tokenizer.py` - Tokenizer training and special-token placement
  - `focus.py` - FOCUS embedding initialization and its sidecar cache
  - `evaluation.py` - Eval sets, metrics, generation, and evaluation callbacks
- `packages/lapt-core/lapt_core/` - Domain-neutral caching layer, packaged
  separately so sibling projects can depend on it without inheriting this one's
  pins
  - `artifacts.py` - `CachedArtifact`, config validation, `ArtifactGraph`
  - `mixing.py` - Source sampling arithmetic and mix cache naming
  - `composites.py` - Concatenation and multinomial mixing
- `configs/` - Hydra configuration files
- `tools/` - Analysis and plotting scripts
- `tests/` - Unit tests

### Adding a dataset type

One new module in `lapt/sources/`: subclass `DatasetArtifact`, set `type_name`,
implement `config()` (the parameters the cache is keyed on), `build()`, and
`from_config()`, then call `SOURCE_TYPES.register(...)` and import it in
`lapt/sources/__init__.py`. There is no dispatcher to edit.

## Citation

If you use this framework, please cite this repository and the related work:

```bibtex
@misc{lapt2025,
  author = {Downey, C.M.},
  title = {LAPT: Language-Adaptive Pretraining Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ur2nlp/LAPT}
}

@inproceedings{downey-etal-2024-targeted,
  title = "Targeted Multilingual Adaptation for Low-resource Language Families",
  author = "Downey, C. M. and Blevins, Terra and Serai, Dhwani and Parikh, Dwija and Steinert-Threlkeld, Shane",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
  year = "2024",
  address = "Miami, Florida, USA",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2024.findings-emnlp.918",
  pages = "15647--15663",
}
```

If you use FOCUS vocabulary specialization, please also cite:

```bibtex
@inproceedings{dobler-de-melo-2023-focus,
  title = "{FOCUS}: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models",
  author = "Dobler, Konstantin and de Melo, Gerard",
  editor = "Bouamor, Houda and Pino, Juan and Bali, Kalika",
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  month = dec,
  year = "2023",
  address = "Singapore",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.emnlp-main.829",
  doi = "10.18653/v1/2023.emnlp-main.829",
  pages = "13440--13454",
}
```

## Related Work

This framework extends the approach from:
- [CLMBRs/targeted-xlms](https://github.com/CLMBRs/targeted-xlms) - Targeted cross-lingual pretraining
- [konstantinjdobler/focus](https://github.com/konstantinjdobler/focus) - FOCUS embedding initialization
