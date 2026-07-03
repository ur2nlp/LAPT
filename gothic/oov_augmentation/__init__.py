"""OOV-robustness augmentation for Gothic instruction tuning.

Generates plausible non-words and rewrites verified word-aligned translation
pairs into targets that flag uncertainty instead of hallucinating a fluent
translation. See `.claude/gothic/oov_robustness_augmentation.md`.
"""
