# LAPT: Language-Adaptive Pretraining Framework

A modular framework for continued pre-training of multilingual language models with flexible data loading and optional vocabulary specialization via [FOCUS](https://github.com/konstantinjdobler/focus).

## Features

- **Flexible dataset loading**: OSCAR corpus, local plaintext files, directory-based loading, concatenation, and temperature-scaled multinomial sampling
- **FOCUS integration**: Optional vocabulary specialization with new tokenizer training using SentencePiece
- **Hydra configuration**: Composable YAML configs for easy experimentation
- **Per-language evaluation**: Automatic per-language dev set tracking for multilingual training

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
python -m src dataset.type=oscar dataset.language=hy
```

### FOCUS Training

Train with vocabulary specialization:

```bash
python -m src focus.enabled=true focus.vocab_size=32768 focus.num_samples=1000000
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
python -m src dataset=multilingual
```

### Local Data

Train on your own plaintext files:

```bash
python -m src dataset.type=plaintext dataset.path=/path/to/data.txt
```

## Project Structure

- `src/` - Main source code
  - `__main__.py` - Training orchestration
  - `dataset_utils.py` - Dataset loading and processing
  - `model_utils.py` - Model and tokenizer initialization
  - `tokenizer_utils.py` - Tokenizer training and FOCUS operations
- `configs/` - Hydra configuration files
- `tests/` - Unit tests

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
