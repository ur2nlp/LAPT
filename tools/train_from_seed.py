"""
Train a SentencePiece tokenizer from an existing seed vocabulary file.

Useful for testing the effect of seed vocab modifications (e.g., scaled counts)
without re-running the full seed generation pipeline.

Usage:
    # Train from an existing seed_vocab.txt
    python tools/train_from_seed.py \
        --seed-vocab path/to/seed_vocab.txt \
        --text-file path/to/training_subset_spm.txt \
        --output path/to/output_tokenizer \
        --vocab-size 32768

    # Scale all counts by 1000x to test whether SentencePiece uses absolute vs relative counts
    python tools/train_from_seed.py \
        --seed-vocab path/to/seed_vocab.txt \
        --text-file path/to/training_subset_spm.txt \
        --output path/to/output_tokenizer_scaled \
        --vocab-size 32768 \
        --scale-counts 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tokenizer_utils import (
    _copy_base_post_processor,
    _create_unigram_tokenizer,
    _detect_tokenizer_algorithm,
    _extract_special_tokens,
    _resolve_hf_special_tokens,
    _train_sentencepiece_model,
    _validate_tokenizer,
)


def train_from_seed(
    seed_vocab_path: str,
    text_file_path: str,
    output_path: str,
    vocab_size: int,
    base_tokenizer_name: str = "facebook/xglm-564M",
    character_coverage: float = 1.0,
    inherit_additional_special_tokens: bool = True,
    scale_counts: float = None,
    seed_sentencepiece_size: int = None,
):
    import os

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if os.path.exists(os.path.join(output_path, "tokenizer.json")):
        print(f"Tokenizer already exists at {output_path}, delete it first to retrain")
        return

    # Optionally scale counts in the seed file
    actual_seed_path = seed_vocab_path
    if scale_counts is not None and scale_counts != 1.0:
        import tempfile
        scaled_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='_scaled_seed.txt', delete=False
        )
        actual_seed_path = scaled_file.name
        print(f"Scaling seed vocab counts by {scale_counts}x")
        with open(seed_vocab_path, encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 2:
                    token, count = parts[0], int(parts[1])
                    scaled_count = max(1, int(count * scale_counts))
                    scaled_file.write(f"{token}\t{scaled_count}\n")
        scaled_file.close()
        print(f"Scaled seed vocab written to {actual_seed_path}")

    # Load base tokenizer for algorithm detection and special tokens
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, use_fast=True)
    model_type = _detect_tokenizer_algorithm(base_tokenizer)
    print(f"Detected tokenizer algorithm: {model_type}")

    special_tokens_config = _extract_special_tokens(
        base_tokenizer,
        inherit_additional=inherit_additional_special_tokens,
        vocab_size=vocab_size,
    )

    os.makedirs(output_path, exist_ok=True)

    # Train SentencePiece with the seed file
    sp_model = _train_sentencepiece_model(
        text_file_path=text_file_path,
        model_type=model_type,
        vocab_size=vocab_size,
        special_tokens_config=special_tokens_config,
        output_path=output_path,
        character_coverage=character_coverage,
        seed_sentencepieces_file=actual_seed_path,
        seed_sentencepiece_size=seed_sentencepiece_size,
    )

    # Convert to HuggingFace tokenizer
    actual_vocab_size = sp_model.get_piece_size()
    vocab_with_scores = [
        (sp_model.id_to_piece(i), sp_model.get_score(i))
        for i in range(actual_vocab_size)
    ]

    if model_type == 'bpe':
        from tokenizers import SentencePieceBPETokenizer, decoders
        model_file = os.path.join(output_path, 'spm.model')
        backend_tokenizer = SentencePieceBPETokenizer.from_file(
            vocab=model_file,
            replacement="▁",
            add_prefix_space=True
        )
        backend_tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    else:
        backend_tokenizer = _create_unigram_tokenizer(
            vocab_with_scores,
            unk_id=special_tokens_config['unk_id'],
        )

    _copy_base_post_processor(backend_tokenizer, base_tokenizer, special_tokens_config)

    trained_vocab = {piece for piece, _score in vocab_with_scores}
    hf_special_tokens = _resolve_hf_special_tokens(
        base_tokenizer,
        special_tokens_config,
        trained_vocab,
    )
    new_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        bos_token=hf_special_tokens['bos_token'],
        eos_token=hf_special_tokens['eos_token'],
        unk_token=hf_special_tokens['unk_token'],
        pad_token=hf_special_tokens['pad_token'],
        clean_up_tokenization_spaces=True,
    )

    if inherit_additional_special_tokens:
        if hasattr(base_tokenizer, 'additional_special_tokens') and base_tokenizer.additional_special_tokens:
            new_tokenizer.add_special_tokens({
                'additional_special_tokens': base_tokenizer.additional_special_tokens
            })

    new_tokenizer.save_pretrained(output_path)
    print(f"Tokenizer saved to {output_path}")

    _validate_tokenizer(new_tokenizer, vocab_size)

    # Print overlap with base
    new_vocab = set(new_tokenizer.get_vocab().keys())
    base_vocab = set(base_tokenizer.get_vocab().keys())
    overlap = new_vocab & base_vocab
    print(f"\nVocab size: {len(new_vocab)}")
    print(f"Base overlap: {len(overlap)} ({100 * len(overlap) / len(new_vocab):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Train a tokenizer from an existing seed vocabulary file"
    )
    parser.add_argument(
        "--seed-vocab", required=True,
        help="Path to seed_vocab.txt (token\\tcount format)"
    )
    parser.add_argument(
        "--text-file", required=True,
        help="Path to SentencePiece training text file"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for the trained tokenizer"
    )
    parser.add_argument(
        "--vocab-size", type=int, required=True,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "--base-tokenizer", default="facebook/xglm-564M",
        help="Base tokenizer for algorithm detection and special tokens"
    )
    parser.add_argument(
        "--character-coverage", type=float, default=1.0,
        help="SentencePiece character coverage (default: 1.0)"
    )
    parser.add_argument(
        "--scale-counts", type=float, default=None,
        help="Multiply all seed vocab counts by this factor (for testing absolute vs relative)"
    )
    parser.add_argument(
        "--seed-sentencepiece-size", type=int, default=None,
        help="Max seed pieces to keep (top-k by count). Default: SentencePiece's 1M"
    )

    args = parser.parse_args()

    train_from_seed(
        seed_vocab_path=args.seed_vocab,
        text_file_path=args.text_file,
        output_path=args.output,
        vocab_size=args.vocab_size,
        base_tokenizer_name=args.base_tokenizer,
        character_coverage=args.character_coverage,
        scale_counts=args.scale_counts,
        seed_sentencepiece_size=args.seed_sentencepiece_size,
    )


if __name__ == "__main__":
    main()
