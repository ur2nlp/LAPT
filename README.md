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

## Project Structure

- `lapt/` - Framework source (installable package)
  - `__main__.py` - Training orchestration
  - `sources/` - One module per dataset type, registered by its `type` field
  - `dataset_utils.py` - Tokenized-dataset stages, eval sets, data collation
  - `tokenization.py` - Stateless text-to-token helpers
  - `artifact_configs.py` - Per-stage configuration records and cache paths
  - `model_utils.py` - Model and tokenizer initialization
  - `tokenizer_utils.py` - Tokenizer training and FOCUS operations
  - `eval_utils.py` - Metrics, generation, and evaluation callbacks
- `lapt_core/` - Domain-neutral caching layer, kept free of ML dependencies so
  sibling projects can depend on it without inheriting this one's pins
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
