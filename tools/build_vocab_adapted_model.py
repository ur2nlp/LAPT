"""
Build a vocab-adapted model checkpoint without any CPT, for diagnostic purposes.

This tool applies vocabulary adaptation (embedding resize/replacement) to a base model
using a pre-built tokenizer, then saves the result without any training. Useful for
isolating whether embedding initialization or CPT is responsible for generation quality
issues.

The resulting model is saved alongside its tokenizer so it can be used directly with
interactive_prompt.py or any other HuggingFace-compatible tool.

Usage:
    python tools/build_vocab_adapted_model.py \\
        --base-model facebook/xglm-564M \\
        --tokenizer tokenizers/xglm564m_pruned-c4-p999 \\
        --output-dir models/diagnostic/xglm564m_pruned-no-cpt

    # With optional perplexity eval on a plaintext file:
    python tools/build_vocab_adapted_model.py \\
        --base-model facebook/xglm-564M \\
        --tokenizer tokenizers/xglm564m_pruned-c4-p999 \\
        --output-dir models/diagnostic/xglm564m_pruned-no-cpt \\
        --eval-data path/to/eval.txt \\
        --eval-samples 2000 \\
        --device cuda

Notes:
    - For tokenizers with no novel tokens (prune-only), embeddings are copied directly
      from the base model without requiring any training data.
    - For tokenizers with novel tokens, pre-computed FOCUS embeddings must be cached
      in the tokenizer directory (under focus_embs/<hash>.input.pt, or the legacy
      focus_input_embeddings.pt at the tokenizer root). These are created
      automatically when running training with focus.tokenizer_path set. This tool
      passes reuse_policy="any" — it requires that exactly one cached embedding
      set exist for the tokenizer.
    - Eval perplexity is computed over whole tokenized lines up to --max-length tokens.
      Each line is treated as an independent example with standard causal LM loss.
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Allow imports from src/ when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from tokenizer_utils import apply_focus_initialization

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_vocab_adapted_model(
    base_model: str,
    tokenizer_path: str,
) -> tuple:
    """
    Load a base model and apply vocabulary adaptation using a pre-built tokenizer.

    Embedding matrices are resized and replaced by copying weights for tokens
    that appear in both vocabularies. For tokenizers with novel tokens, exactly
    one cached FOCUS embedding set must exist under tokenizer_path (either
    focus_embs/<hash>.input.pt or the legacy focus_input_embeddings.pt).

    Args:
        base_model: HuggingFace model ID or local path for the base model.
        tokenizer_path: Path to the pre-built target tokenizer directory.

    Returns:
        Tuple of (adapted_model, target_tokenizer).
    """
    print(f"Loading base model: {base_model}", file=sys.stderr)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    config = AutoConfig.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, config=config)

    print(f"Loading target tokenizer: {tokenizer_path}", file=sys.stderr)
    target_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    print(
        f"Vocab sizes — base: {len(base_tokenizer)}, target: {len(target_tokenizer)}",
        file=sys.stderr,
    )

    # Check for novel tokens (tokens in target but not in base).
    # apply_focus_initialization handles both cases:
    #   - No novel tokens: direct embedding copy (no training data needed)
    #   - Novel tokens: loads from cache in tokenizer_path (training data not used)
    # We pass a dummy training_data_path since it is only consumed by the FastText
    # code path, which is bypassed when embeddings are cached or there are no novel tokens.
    dummy_training_data_path = os.path.join(tokenizer_path, 'training_subset.jsonl')

    new_input_embeddings, new_output_embeddings = apply_focus_initialization(
        source_model=model,
        source_tokenizer=base_tokenizer,
        target_tokenizer=target_tokenizer,
        training_data_path=dummy_training_data_path,
        fasttext_model_min_count=1,
        cache_dir=tokenizer_path,
        reuse_policy="any",
    )

    # Resize embedding layers and replace weights.
    model.resize_token_embeddings(len(target_tokenizer))

    new_input_embedding_layer = torch.nn.Embedding.from_pretrained(
        new_input_embeddings,
        padding_idx=target_tokenizer.pad_token_id,
    )
    model.set_input_embeddings(new_input_embedding_layer)

    if hasattr(model.config, 'tie_word_embeddings') and not model.config.tie_word_embeddings:
        if new_output_embeddings is not None:
            model.get_output_embeddings().weight.data = new_output_embeddings  # type: ignore
    else:
        model.tie_weights()

    # Sync special token IDs — PTEx reorders pad/eos/unk relative to base XGLM.
    model.config.vocab_size = len(target_tokenizer)
    model.config.pad_token_id = target_tokenizer.pad_token_id
    model.config.bos_token_id = target_tokenizer.bos_token_id
    model.config.eos_token_id = target_tokenizer.eos_token_id

    del base_tokenizer
    return model, target_tokenizer


class PlaintextDataset(Dataset):
    """Lines from a plaintext file, tokenized and truncated."""

    def __init__(self, path: str, tokenizer, max_length: int, max_samples: int):
        self.examples = []
        # Character count of the original text, used for BPC computation.
        self.char_lens = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=True)
                if len(ids) > max_length:
                    ids = ids[:max_length]
                self.examples.append(torch.tensor(ids, dtype=torch.long))
                self.char_lens.append(len(text))
                if len(self.examples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.examples[idx], self.char_lens[idx]


def collate_fn(batch: list[tuple[torch.Tensor, int]], pad_id: int) -> dict:
    """Pad a batch of variable-length sequences to the longest example."""
    seqs, char_lens = zip(*batch)
    max_len = max(t.shape[0] for t in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
    labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)

    for i, seq in enumerate(seqs):
        seq_len = seq.shape[0]
        input_ids[i, :seq_len] = seq
        attention_mask[i, :seq_len] = 1
        # Causal LM loss: predict each token from the previous ones.
        # Shift labels left by one position (HuggingFace CausalLM handles this internally).
        labels[i, :seq_len] = seq

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'total_chars': sum(char_lens),
    }


def evaluate_metrics(
    model,
    tokenizer,
    eval_path: str,
    max_samples: int,
    device: str,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    """
    Compute BPC and token-level NLL on a plaintext file.

    BPC (bits per character) normalizes for tokenizer granularity and is the
    preferred metric when comparing models with different vocabularies:
        BPC = (total_nll_nats / ln(2)) / total_chars

    Args:
        model: The (adapted) model to evaluate.
        tokenizer: Corresponding tokenizer.
        eval_path: Path to a plaintext file (one example per line).
        max_samples: Maximum number of lines to evaluate.
        device: torch device string.
        max_length: Maximum token length per example.
        batch_size: Evaluation batch size.

    Returns:
        Dict with keys 'bpc', 'nll', and 'perplexity'.
    """
    dataset = PlaintextDataset(eval_path, tokenizer, max_length, max_samples)
    print(f"Evaluating on {len(dataset)} examples from {eval_path}", file=sys.stderr)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        shuffle=False,
    )

    model.eval()
    model.to(device)
    total_nll_nats = 0.0
    total_tokens = 0
    total_chars = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # outputs.loss is mean NLL over non-masked tokens in the batch.
            # Recover the sum by multiplying by the number of non-padding label tokens.
            num_tokens = (labels != -100).sum().item()
            total_nll_nats += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            total_chars += batch['total_chars']

    import math
    mean_nll = total_nll_nats / total_tokens
    bpc = (total_nll_nats / math.log(2)) / total_chars
    perplexity = math.exp(mean_nll)

    return {'bpc': bpc, 'nll': mean_nll, 'perplexity': perplexity}


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Build a vocab-adapted model checkpoint without any CPT, '
            'or evaluate an already-built one. '
            'If --output-dir already contains a saved model, '
            '--base-model and --tokenizer are not required.'
        ),
    )
    parser.add_argument(
        '--base-model',
        default=None,
        help='HuggingFace model ID or local path (e.g. facebook/xglm-564M). '
             'Required when --output-dir does not already contain a saved model.',
    )
    parser.add_argument(
        '--tokenizer',
        default=None,
        help='Path to the pre-built target tokenizer directory. '
             'Required when --output-dir does not already contain a saved model.',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save (or load) the adapted model and tokenizer.',
    )
    parser.add_argument(
        '--eval-data',
        default=None,
        help='Optional: path to a plaintext file for perplexity evaluation',
    )
    parser.add_argument(
        '--eval-samples',
        type=int,
        default=2000,
        help='Maximum number of lines to use for evaluation (default: 2000)',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum token length per example (default: 512)',
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation (default: 16)',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Device for evaluation (default: cpu; use cuda for GPU)',
    )
    parser.add_argument(
        '--dtype',
        choices=['float32', 'bfloat16', 'float16'],
        default='bfloat16',
        help='Model dtype (default: bfloat16)',
    )
    args = parser.parse_args()

    already_built = os.path.exists(os.path.join(args.output_dir, 'config.json'))

    if already_built:
        print(f"Loading existing model from {args.output_dir}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model = AutoModelForCausalLM.from_pretrained(args.output_dir)
    else:
        if args.base_model is None or args.tokenizer is None:
            parser.error(
                '--base-model and --tokenizer are required when --output-dir does not '
                'contain an already-built model.'
            )
        model, tokenizer = build_vocab_adapted_model(
            base_model=args.base_model,
            tokenizer_path=args.tokenizer,
        )

    dtype_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }
    model = model.to(dtype_map[args.dtype])

    if args.eval_data is not None:
        metrics = evaluate_metrics(
            model=model,
            tokenizer=tokenizer,
            eval_path=args.eval_data,
            max_samples=args.eval_samples,
            device=args.device,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
        )
        print(f"BPC: {metrics['bpc']:.4f}  NLL: {metrics['nll']:.4f}  PPL: {metrics['perplexity']:.2f}")

    print(f"Saving model and tokenizer to {args.output_dir}", file=sys.stderr)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.", file=sys.stderr)


if __name__ == '__main__':
    main()
