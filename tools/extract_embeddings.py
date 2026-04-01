"""Extract an embedding matrix from a HuggingFace checkpoint to a .pt file.

Reads weights directly from the checkpoint without instantiating the full model.
Supports both safetensors and pytorch_model.bin formats.

Usage:
    python tools/extract_embeddings.py models/old_germanic/checkpoint-5000/ embeddings.pt
    python tools/extract_embeddings.py models/old_germanic/checkpoint-5000/ embeddings.pt \\
        --key lm_head.weight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def load_tensor(checkpoint_dir: Path, key: str) -> torch.Tensor:
    """Load a single tensor from a checkpoint directory.

    Tries safetensors first (single-file and sharded), then pytorch_model.bin.

    Args:
        checkpoint_dir: Path to the HuggingFace checkpoint directory.
        key: State-dict key for the desired tensor.

    Returns:
        The requested tensor on CPU.
    """
    # Try single-file safetensors
    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path, device="cpu")
        return _extract_key(state_dict, key, str(safetensors_path))

    # Try sharded safetensors (model-00001-of-NNNNN.safetensors)
    shards = sorted(checkpoint_dir.glob("model-*-of-*.safetensors"))
    if shards:
        from safetensors import safe_open
        for shard_path in shards:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                if key in f.keys():
                    return f.get_tensor(key)
        available = _sample_keys_from_shards(shards)
        raise KeyError(
            f"Key {key!r} not found in any of {len(shards)} safetensors shards. "
            f"Sample keys: {available}"
        )

    # Try pytorch_model.bin
    bin_path = checkpoint_dir / "pytorch_model.bin"
    if bin_path.exists():
        state_dict = torch.load(bin_path, weights_only=True, map_location="cpu")
        return _extract_key(state_dict, key, str(bin_path))

    raise FileNotFoundError(
        f"No checkpoint weights found in {checkpoint_dir}. "
        "Expected model.safetensors, model-*-of-*.safetensors, or pytorch_model.bin."
    )


def _extract_key(state_dict: dict, key: str, source: str) -> torch.Tensor:
    """Extract key from state dict with a helpful error if missing."""
    if key not in state_dict:
        sample = [k for k in list(state_dict.keys())[:10]]
        raise KeyError(
            f"Key {key!r} not found in {source}. "
            f"First 10 keys: {sample}"
        )
    return state_dict[key]


def _sample_keys_from_shards(shards: list[Path], n: int = 10) -> list[str]:
    """Return up to n keys from the first shard for error messages."""
    from safetensors import safe_open
    with safe_open(shards[0], framework="pt", device="cpu") as f:
        return list(f.keys())[:n]


def main():
    parser = argparse.ArgumentParser(
        description="Extract an embedding matrix from a HuggingFace checkpoint to a .pt file.",
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the checkpoint directory.",
    )
    parser.add_argument(
        "output",
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--key",
        default="model.embed_tokens.weight",
        help=(
            "State-dict key for the embedding matrix. "
            "Default: model.embed_tokens.weight (XGLM / LLaMA-style)."
        ),
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_dir():
        print(f"Error: {checkpoint_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.key!r} from {checkpoint_dir} ...", file=sys.stderr)
    tensor = load_tensor(checkpoint_dir, args.key)
    print(f"  shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}", file=sys.stderr)

    torch.save(tensor, args.output)
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
