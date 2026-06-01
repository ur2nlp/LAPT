"""
Patch stale special-token IDs in a saved model's generation_config.json.

Background:
    When a model is built with a vocabulary-adapted (PTEx/FOCUS) tokenizer, the
    special tokens (pad/eos/unk) are reordered relative to the base model. The
    build code syncs model.config to the new IDs, but transformers writes a
    separate generation_config.json that retains the *base* model's IDs. Because
    generation_config takes precedence during generate()/pipeline(), the model
    halts on the wrong EOS id and never stops on its real end-of-sequence token,
    producing output that runs on until max_new_tokens.

    This script rewrites the eos/bos/pad token IDs in generation_config.json to
    match the model's own tokenizer. It is idempotent: already-correct files are
    left unchanged.

Usage:
    python tools/fix_generation_config.py path/to/model [path/to/model ...]

    The path may be a checkpoint directory containing both a tokenizer and a
    generation_config.json.
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def patch_generation_config(model_dir: Path) -> bool:
    """Sync generation_config.json special-token IDs to the model's tokenizer.

    Args:
        model_dir: Directory containing a tokenizer and generation_config.json.

    Returns:
        True if the file was changed, False if it was already correct.
    """
    gen_config_path = model_dir / "generation_config.json"
    if not gen_config_path.exists():
        print(f"  no generation_config.json in {model_dir}, skipping", file=sys.stderr)
        return False

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    correct_ids = {
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with open(gen_config_path) as config_file:
        gen_config = json.load(config_file)

    changes = {}
    for key, correct_value in correct_ids.items():
        if correct_value is None:
            continue
        if gen_config.get(key) != correct_value:
            changes[key] = (gen_config.get(key), correct_value)
            gen_config[key] = correct_value

    if not changes:
        print(f"  {gen_config_path}: already correct")
        return False

    with open(gen_config_path, "w") as config_file:
        json.dump(gen_config, config_file, indent=2)
        config_file.write("\n")

    for key, (old_value, new_value) in changes.items():
        print(f"  {gen_config_path}: {key} {old_value} -> {new_value}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patch stale special-token IDs in generation_config.json"
    )
    parser.add_argument(
        "model_dirs",
        nargs="+",
        help="One or more model/checkpoint directories to patch",
    )
    args = parser.parse_args()

    any_changed = False
    for model_dir in args.model_dirs:
        path = Path(model_dir)
        print(f"Patching {path}...")
        if patch_generation_config(path):
            any_changed = True

    if any_changed:
        print("Done. Generation configs updated.")
    else:
        print("Done. Nothing to change.")


if __name__ == "__main__":
    main()
