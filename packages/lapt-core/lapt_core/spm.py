"""Build HuggingFace tokenizer backends from a trained SentencePiece model.

A SentencePiece model is not directly usable as a `tokenizers.Tokenizer`: its
pieces and scores have to be loaded into a Unigram or BPE model, and the text
handling SentencePiece implies -- no normalization, `▁` for a leading space --
has to be reinstalled on the backend. This module does that, and nothing else.

It is deliberately free of any policy about *special* tokens. Where those end up
in the id space is a decision that depends on the base model, and the three
viable answers are documented in the guide on special-token policies; every one
of them builds its backend the same way, through the functions here.

Unlike `lapt_core.artifacts`, this module needs `tokenizers`, `transformers` and
`sentencepiece` -- install them with the `tokenizers` extra. The imports are
function-local so that importing this module stays cheap and so that the BPE
path is the only one that pays for `transformers`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer


SPM_SPACE = "▁"


def apply_spm_pipeline(backend_tokenizer) -> None:
    """Install the normalizer, pre-tokenizer, and decoder SentencePiece implies.

    Shared by the Unigram and BPE branches so both reproduce identical text
    handling regardless of the underlying model:

    - An empty normalizer, matching training with `normalization_rule_name`
      set to `identity`.
    - A Metaspace pre-tokenizer, so a leading space becomes `▁`.
    - A Metaspace decoder, so `▁` becomes a space again. Without it, encode and
      decode are not inverses and every decoded string loses its spacing.

    This *replaces* a byte-level pipeline rather than reusing it when the base
    checkpoint had one. The learned pieces are SentencePiece pieces, and a
    ByteLevel pre-tokenizer would never emit a string that matches one.

    Args:
        backend_tokenizer: A `tokenizers.Tokenizer` to configure in place.
    """
    from tokenizers import decoders, normalizers
    from tokenizers.pre_tokenizers import Metaspace

    backend_tokenizer.normalizer = normalizers.Sequence(normalizers=[])
    backend_tokenizer.pre_tokenizer = Metaspace(replacement=SPM_SPACE, prepend_scheme="always")
    backend_tokenizer.decoder = decoders.Metaspace(replacement=SPM_SPACE, prepend_scheme="always")


def create_unigram_backend(
    vocab_scores: list[tuple[str, float]],
    unk_id: int = 0,
) -> "Tokenizer":
    """Build a `tokenizers.Tokenizer` around a Unigram model.

    Args:
        vocab_scores: (piece, score) pairs from the SentencePiece model, in
            piece-id order. The index of a pair becomes the token's id.
        unk_id: Id of the unknown piece. Must match the `unk_id` SentencePiece
            was trained with, since the model indexes it by id rather than by
            string.

    Returns:
        A configured `tokenizers.Tokenizer`, ready for `PreTrainedTokenizerFast`.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram

    # byte_fallback=False sends unknown characters to <unk>, which is what a
    # model trained without byte fallback expects.
    backend_tokenizer = Tokenizer(Unigram(vocab_scores, unk_id=unk_id, byte_fallback=False))
    apply_spm_pipeline(backend_tokenizer)
    return backend_tokenizer


def create_bpe_backend(
    spm_model_path: str,
    vocab_scores: list[tuple[str, float]],
    unk_token: str,
) -> "Tokenizer":
    """Build a `tokenizers.Tokenizer` around a BPE model.

    A SentencePiece BPE model stores pieces and their scores but no explicit
    merge list, so the merges have to be reconstructed. This mirrors
    HuggingFace's own `SpmConverter`: merges are derived from the piece scores
    (higher score = earlier merge) and ids follow piece order.

    Args:
        spm_model_path: Path to the trained SentencePiece `.model` file.
        vocab_scores: (piece, score) pairs from that model, in piece-id order.
        unk_token: The unknown piece as a string. BPE indexes it by string,
            where Unigram indexes it by id.

    Returns:
        A configured `tokenizers.Tokenizer`, ready for `PreTrainedTokenizerFast`.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers.convert_slow_tokenizer import SentencePieceExtractor

    _vocab, merges = SentencePieceExtractor(spm_model_path).extract(vocab_scores)
    bpe_vocab = {piece: index for index, (piece, _score) in enumerate(vocab_scores)}

    # fuse_unk=True and byte_fallback=False match the SpmConverter defaults for
    # a non-byte-level SentencePiece BPE model trained without byte fallback.
    backend_tokenizer = Tokenizer(
        BPE(
            bpe_vocab,
            merges,
            unk_token=unk_token,
            fuse_unk=True,
            byte_fallback=False,
            dropout=None,
        )
    )
    apply_spm_pipeline(backend_tokenizer)
    return backend_tokenizer
