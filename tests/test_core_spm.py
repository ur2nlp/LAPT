"""Tests for lapt_core.spm (SentencePiece -> HuggingFace backend construction)."""

import pytest

from lapt_core.spm import (
    SPM_SPACE,
    apply_spm_pipeline,
    create_bpe_backend,
    create_unigram_backend,
)


pytest.importorskip("tokenizers", reason="lapt_core.spm needs the tokenizers extra")


@pytest.fixture
def vocab_scores():
    """Pieces in piece-id order, as SentencePiece reports them."""
    pieces = ["<unk>", f"{SPM_SPACE}a", f"{SPM_SPACE}ab", f"{SPM_SPACE}c", "b", "c"]
    return [(piece, -float(index)) for index, piece in enumerate(pieces)]


class TestUnigramBackend:
    def test_ids_follow_piece_order(self, vocab_scores):
        backend = create_unigram_backend(vocab_scores)

        vocab = backend.get_vocab()
        assert [vocab[piece] for piece, _score in vocab_scores] == [0, 1, 2, 3, 4, 5]

    def test_pipeline_round_trips_spaces(self, vocab_scores):
        """
        Without the Metaspace decoder, encode and decode are not inverses and
        every decoded string loses its spacing.
        """
        backend = create_unigram_backend(vocab_scores)

        encoded = backend.encode("ab c")
        assert backend.decode(encoded.ids) == "ab c"

    def test_unk_id_is_honoured(self, vocab_scores):
        """
        Unigram indexes the unknown piece by id, so a mismatch with the id
        SentencePiece trained with silently makes the wrong piece the unknown.
        """
        reordered = [vocab_scores[1], vocab_scores[0]] + vocab_scores[2:]

        backend = create_unigram_backend(reordered, unk_id=1)

        assert backend.get_vocab()["<unk>"] == 1


class TestSpmPipeline:
    def test_applies_metaspace_to_both_ends(self, vocab_scores):
        backend = create_unigram_backend(vocab_scores)

        assert "Metaspace" in str(backend.pre_tokenizer)
        assert "Metaspace" in str(backend.decoder)

    def test_replaces_an_existing_byte_level_pipeline(self, vocab_scores):
        """
        A base checkpoint may arrive with ByteLevel installed. It would never
        emit a string matching a SentencePiece piece, so it has to go.
        """
        from tokenizers.pre_tokenizers import ByteLevel

        backend = create_unigram_backend(vocab_scores)
        backend.pre_tokenizer = ByteLevel()

        apply_spm_pipeline(backend)

        assert "ByteLevel" not in str(backend.pre_tokenizer)
        assert "Metaspace" in str(backend.pre_tokenizer)


class TestBpeBackend:
    def test_builds_from_a_trained_model(self, tmp_path):
        """
        The BPE branch needs a real .model file, since the merges are
        reconstructed from it rather than passed in.
        """
        spm = pytest.importorskip("sentencepiece")

        corpus = tmp_path / "corpus.txt"
        words = [
            f"{first}{second}{third}"
            for first in "abcdef"
            for second in "aeiou"
            for third in "lmnrst"
        ]
        lines = [" ".join(words[index : index + 6]) for index in range(0, len(words), 6)]
        corpus.write_text("\n".join(lines * 5))
        prefix = str(tmp_path / "spm")
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=prefix,
            model_type="bpe",
            vocab_size=64,
            character_coverage=1.0,
            normalization_rule_name="identity",
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )
        processor = spm.SentencePieceProcessor()
        processor.Load(f"{prefix}.model")
        scores = [
            (processor.id_to_piece(index), processor.get_score(index))
            for index in range(processor.get_piece_size())
        ]

        backend = create_bpe_backend(f"{prefix}.model", scores, unk_token="<unk>")

        assert type(backend.model).__name__ == "BPE"
        assert "Metaspace" in str(backend.pre_tokenizer)
        assert backend.decode(backend.encode("ab ac").ids) == "ab ac"
