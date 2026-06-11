"""Shared schema helpers for the canonical word-spotting alignments format.

The canonical ``{train,test}_alignments.jsonl`` files (produced by
``assign_alignment_ids.py``) carry a ``status`` on every alignment recording its
verification provenance. Every generator that turns these alignments into
training data — ``expand_to_instruction.py`` (word-spotting projections) and
``expand_to_cot.py`` (chain-of-thought translation) — must train on the *same*
subset, or the two datasets silently diverge. This module defines that subset
once so the criterion cannot drift between generators.

See ``.claude/gothic/word_spotting.md`` § "v0.5 single-source cutover".
"""

# Alignment statuses trustworthy enough to train on. Deliberately excludes
# ``replaced_in_diversification`` (correct but downsampled for balance),
# ``rejected`` (verified wrong), and ``unverified`` (not yet reviewed).
TRAINABLE_STATUSES = ("verified_correct", "kept_edited")


def trainable_alignments(entry: dict) -> list[dict]:
    """Return the entry's alignments whose status is trustworthy for training.

    Args:
        entry: A canonical alignments JSONL entry.

    Returns:
        The sublist of alignments with a status in ``TRAINABLE_STATUSES``. The
        canonical schema carries a ``status`` on every alignment; older
        finalized files without it are treated as all-trainable.
    """
    kept = []
    for alignment in entry["alignments"]:
        status = alignment.get("status")
        if status is None or status in TRAINABLE_STATUSES:
            kept.append(alignment)
    return kept
