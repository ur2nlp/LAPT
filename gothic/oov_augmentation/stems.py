"""Assemble the pseudo-stem trigram model for Gothic non-word generation.

Ties together the pieces built in steps 1-3 (see
`.claude/gothic/oov_robustness_augmentation.md`):

    train vocab -> function-word filter -> affix scoring/selection (both edges)
                -> longest-match stripping of prefix + suffix -> pseudo-stems
                -> fit trigram (phonotactics of stems only)
                -> rejection vocab = real word forms

Generation of a non-word for a given source word grafts *that word's own*
canonical prefix and suffix back onto a freshly generated stem, so the non-word
carries plausible morphology while its stem is genuinely novel. Junctions on both
edges are validated against characters attested across that affix boundary in
real words.
"""

from dataclasses import dataclass

from gothic.oov_augmentation.affixes import (
    attested_junction_chars,
    build_pseudo_stems,
    score_affixes,
    select_affixes,
    sort_by_length,
    strip_both,
)
from gothic.oov_augmentation.nonword import TrigramNonwordModel
from gothic.oov_augmentation.vocab import (
    content_vocab,
    load_rejection_vocab,
    load_training_vocab,
)


@dataclass
class StemModel:
    """A fitted stem trigram model plus the affix inventories used with it.

    Attributes:
        model: The trigram model, fit on pseudo-stems, with ``known_words`` set to
            the real-word rejection vocabulary.
        prefixes_by_length: Canonical prefixes ordered longest-first.
        suffixes_by_length: Canonical suffixes ordered longest-first.
        post_prefix_chars: Per-prefix set of attested stem-initial characters.
        pre_suffix_chars: Per-suffix set of attested stem-final characters.
        min_residual: Minimum stem length used when stripping.
    """

    model: TrigramNonwordModel
    prefixes_by_length: list[str]
    suffixes_by_length: list[str]
    post_prefix_chars: dict[str, set[str]]
    pre_suffix_chars: dict[str, set[str]]
    min_residual: int


def build_stem_model(
    # T=0.6 is the knee of the plausibility/diversity sweep (`diversity.py`):
    # the highest temperature before per-char-NLL plausibility starts dropping
    # (0.91 -> 0.85 at 0.7), and it dominates 0.5 (same plausibility, more
    # diversity). Acceptance-rate tuning was flat/useless; see the note.
    temperature: float = 0.6,
    top_p: float = 1.0,
    seed: int = 1,
    function_word_top_k: int = 50,
    min_word_length: int = 3,
    # Affix-inventory sizes are Gothic-specific, read off where each direction's
    # weighted-MI ranking visibly degrades (not universal, not a tidy ratio):
    #   - suffixes stay genuine well past 40 (`-aize`, `-jaiþ` at ranks 41-43), so
    #     40 slightly under-covers if anything;
    #   - the deduped prefix ranking hard-stops at 8 (`ga…af`); rank 9-10 are
    #     stem-onset noise (`hv`, `waur`), so 10 would add noise for no gain.
    # For a new language, inspect the ranking (or use `select_affixes(top_n=None)`
    # for the elbow, noting it is a dominance cut). See
    # `.claude/gothic/oov_robustness_augmentation.md`.
    suffix_top_n: int = 40,
    prefix_top_n: int = 8,
    max_affix_length: int = 4,
    min_residual: int = 3,
    include_test_in_rejection: bool = False,
) -> StemModel:
    """Build the full stem model from the prepared Gothic training corpus.

    Args:
        temperature: Sampling temperature for generation.
        top_p: Nucleus-sampling cutoff for generation.
        seed: RNG seed.
        function_word_top_k: Drop the this-many most frequent types as function
            words before fitting.
        min_word_length: Minimum content-word length kept by the function filter.
        suffix_top_n: Size of the canonical suffix inventory (Gothic-specific;
            see the comment at the defaults).
        prefix_top_n: Size of the canonical prefix inventory (Gothic-specific;
            far smaller — Gothic has a small closed prefix set).
        max_affix_length: Longest affix considered during scoring.
        min_residual: Minimum stem length left when stripping affixes.
        include_test_in_rejection: Whether to reject collisions with test words
            too (default: train only; see ``load_rejection_vocab``).

    Returns:
        A StemModel ready for ``generate_nonword``.
    """
    content = content_vocab(
        load_training_vocab(),
        min_length=min_word_length,
        top_k=function_word_top_k,
    )
    words = list(content)

    suffix_scores = score_affixes(words, side="suffix", max_length=max_affix_length)
    suffixes_by_length = sort_by_length(select_affixes(suffix_scores, top_n=suffix_top_n))
    prefix_scores = score_affixes(words, side="prefix", max_length=max_affix_length)
    prefixes_by_length = sort_by_length(
        select_affixes(prefix_scores, top_n=prefix_top_n, dedup_nested=True)
    )

    stems = build_pseudo_stems(words, prefixes_by_length, suffixes_by_length, min_residual)
    model = TrigramNonwordModel(temperature=temperature, top_p=top_p, seed=seed)
    model.fit(stems)
    model.known_words = set(load_rejection_vocab(include_test=include_test_in_rejection))

    post_prefix_chars = attested_junction_chars(
        words, prefixes_by_length, side="prefix", min_residual=min_residual
    )
    pre_suffix_chars = attested_junction_chars(
        words, suffixes_by_length, side="suffix", min_residual=min_residual
    )

    return StemModel(
        model=model,
        prefixes_by_length=prefixes_by_length,
        suffixes_by_length=suffixes_by_length,
        post_prefix_chars=post_prefix_chars,
        pre_suffix_chars=pre_suffix_chars,
        min_residual=min_residual,
    )


def generate_nonword(
    stem_model: StemModel,
    source_word: str,
    reject_known: bool = True,
) -> str | None:
    """Generate a non-word for a source word, grafting its prefix and suffix.

    The source word is stripped on both edges to find its canonical prefix and
    suffix; a fresh stem is generated and the affixes are grafted back on, so the
    non-word shares the source word's morphology but not its stem. Both junctions
    are validated against attested adjacent characters.

    Args:
        stem_model: A model from ``build_stem_model``.
        source_word: The romanized real word being replaced.
        reject_known: Whether to reject candidates that collide with a real word.
            Set False only when measuring the collision rate during tuning.

    Returns:
        A romanized non-word (not colliding with a real word when
        ``reject_known``), or None if generation failed within the attempt budget.
    """
    prefix, _, suffix = strip_both(
        source_word,
        stem_model.prefixes_by_length,
        stem_model.suffixes_by_length,
        stem_model.min_residual,
    )
    prefix_allowed = stem_model.post_prefix_chars.get(prefix) if prefix else None
    suffix_allowed = stem_model.pre_suffix_chars.get(suffix) if suffix else None

    accept = None
    if prefix_allowed or suffix_allowed:
        def accept(candidate: str) -> bool:
            if prefix_allowed and candidate[len(prefix)] not in prefix_allowed:
                return False
            if suffix_allowed and candidate[-(len(suffix) + 1)] not in suffix_allowed:
                return False
            return True

    return stem_model.model.generate(
        prefix=prefix or None,
        suffix=suffix or None,
        accept=accept,
        reject_known=reject_known,
    )
