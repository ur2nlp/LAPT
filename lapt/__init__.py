"""LAPT: Language-Adaptive Pre-Training.

Deliberately empty of re-exports. Importing a submodule here would pull torch
and transformers into every consumer -- including the `tools/` scripts that
only want a formatting helper -- and turn a fast CLI into a slow one. Import
what you need directly: `from lapt.dataset_utils import load_tokenized_dataset`.
"""
