"""
Estimate E[log Z] of the base (unpruned) model's output distribution on a
reference text corpus. Z is the softmax partition function over the full
vocabulary at each position.

Motivation: after pruning the vocabulary, continued-training cross-entropy
gradients are distorted by a factor of Z_S/Z (where Z_S is the partition
function over the kept vocabulary). Multiplying gradients by Z_S/Z recovers
the unpruned signal. Z_S is cheap to compute live from the pruned model's
forward pass; Z must be estimated once from the unpruned model and saved.

Output: writes a small JSON file with the scalar estimate. Training code
loads this file from the tokenizer directory and installs gradient-correction
hooks if present.

Usage:
    python tools/estimate_partition_function.py \\
        --base-model facebook/xglm-564M \\
        --text-path data/english_reference.txt \\
        --output-path tokenizers/xglm564m_pruned-c4-p999/z_estimate.json \\
        --max-samples 2000 \\
        --max-length 512 \\
        --batch-size 8 \\
        --device cuda
"""

import argparse
import json
import logging
import math
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PlaintextLinesDataset(Dataset):
    """Tokenize lines from a plaintext file, truncate to max_length, cap at max_samples."""

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        max_samples: int,
    ):
        self.examples: list[torch.Tensor] = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=True)
                if len(ids) > max_length:
                    ids = ids[:max_length]
                if len(ids) < 2:
                    continue
                self.examples.append(torch.tensor(ids, dtype=torch.long))
                if len(self.examples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]


def collate(batch: list[torch.Tensor], pad_id: int) -> dict:
    max_len = max(t.shape[0] for t in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.shape[0]
        input_ids[i, :seq_len] = seq
        attention_mask[i, :seq_len] = 1
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


@torch.no_grad()
def estimate_log_z(
    model,
    loader: DataLoader,
    device: str,
) -> tuple[float, int]:
    """
    Compute the mean of log Z over all non-pad positions of all batches.

    Z is the softmax partition function over the full (unpruned) vocabulary:
    Z(h) = sum_j exp(lm_head(h)_j). We compute log Z via logsumexp directly
    from the model's output logits for numerical stability.

    Returns:
        (mean_log_z, n_positions) — the scalar estimate and how many positions
        it was averaged over.
    """
    model.eval()
    sum_log_z = 0.0
    n_positions = 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # log Z at every position, shape (batch, seq_len)
        log_z = torch.logsumexp(logits.float(), dim=-1)

        # Include only non-pad positions in the average.
        mask = attention_mask.bool()
        valid_log_z = log_z[mask]
        sum_log_z += valid_log_z.sum().item()
        n_positions += valid_log_z.numel()

        if (batch_idx + 1) % 20 == 0:
            running = sum_log_z / n_positions
            print(
                f"  batch {batch_idx + 1}: running mean log Z = {running:.4f}"
                f" over {n_positions:,} positions",
                file=sys.stderr,
            )

    if n_positions == 0:
        raise RuntimeError("No valid positions found; corpus may be empty.")

    return sum_log_z / n_positions, n_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True, help='Base (unpruned) HF model ID or path')
    parser.add_argument('--text-path', required=True, help='Plaintext file, one example per line')
    parser.add_argument(
        '--output-path',
        required=True,
        help='Where to write the JSON estimate (e.g. tokenizers/<pruned>/z_estimate.json)',
    )
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    args = parser.parse_args()

    torch_dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[
        args.dtype
    ]

    print(f"Loading base model: {args.base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype)
    model.to(args.device)

    print(f"Loading text from: {args.text_path}", file=sys.stderr)
    dataset = PlaintextLinesDataset(
        path=args.text_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    print(f"  {len(dataset):,} examples loaded", file=sys.stderr)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    print("Estimating E[log Z] on unpruned base model...", file=sys.stderr)
    mean_log_z, n_positions = estimate_log_z(model, loader, args.device)
    mean_z = math.exp(mean_log_z)

    print(
        f"\nResult: mean log Z = {mean_log_z:.6f} (Z ≈ {mean_z:.4e})"
        f" over {n_positions:,} positions",
        file=sys.stderr,
    )

    output = {
        'mean_log_z': mean_log_z,
        'n_positions': n_positions,
        'base_model': args.base_model,
        'text_path': args.text_path,
        'max_samples': args.max_samples,
        'max_length': args.max_length,
        'vocab_size': model.config.vocab_size,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {args.output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
