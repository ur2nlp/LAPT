"""Shared formatting helpers for Gothic instruction-tuning prompts.

The PTEx tokenizer used for the Gothic models is a Unigram tokenizer with
byte-fallback disabled and no newline piece, so any ``\\n`` in a prompt is
tokenized as ``<unk>``. Newlines therefore carry no usable signal and, at the
prompt/response boundary, fragment the cue (``\\nResponse:`` ->
``<unk>``, ``Respons``, ``e``, ``:`` instead of the canonical ``Response:`` ->
``Respons``... only when preceded by a space). To keep structural separators
tokenizing as ordinary whitespace, instruction prompts are kept on a single
line: every generator collapses newlines to single spaces via
``flatten_prompt`` at the point it assembles a prompt.

This is a stopgap tied to the tokenizer limitation tracked in
``.claude/TODO.md`` ("Newlines tokenize to <unk>"). Once the tokenizer gains a
newline representation (byte-fallback or a user-defined symbol), templates can
carry real newlines again and this flattening can be dropped.
"""

import re


def flatten_prompt(prompt: str) -> str:
    """Collapse newline-bearing whitespace runs to single spaces.

    Any run of whitespace that contains at least one newline is replaced by a
    single space; leading and trailing whitespace is stripped. Runs of plain
    spaces (no newline) are left untouched, so intentional spacing inside a
    template is preserved.

    Args:
        prompt: A prompt string that may contain newline separators.

    Returns:
        The single-line prompt with newlines replaced by single spaces.
    """
    return re.sub(r"[ \t]*\n[ \t\n]*", " ", prompt).strip()
