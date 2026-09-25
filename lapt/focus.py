"""FOCUS: initialize a new vocabulary's embeddings from the pretrained ones.

Three stages, in the order the pipeline runs them:

1. **Corpus preparation** — sample the target-language text FOCUS trains its
   fastText model on, written as JSONL.
2. **Embedding initialization** — run FOCUS over the source model's embedding
   matrix to produce rows for the new vocabulary.
3. **Sidecar caching** — embeddings are expensive, so they are cached beside the
   tokenizer under `focus_embs/<hash>/`, keyed by the inputs that determine
   their values rather than by the tokenizer's identity. One tokenizer can
   therefore host several embedding sets, one per FOCUS-training mix.

Building the tokenizer those embeddings are *for* is `lapt.tokenizer`.
"""

import glob
import json
import os
import random
import sys

import torch
import yaml
from datasets import load_from_disk
from transformers import PreTrainedTokenizerBase


def prepare_focus_training_data(
    num_samples: int,
    output_jsonl_path: str,
    seed: int = 1,
    train_dataset_cache: str = None,
    dataset_config = None
) -> str:
    """
    Extract a random subset of untokenized data and convert to JSONL format.

    Args:
        num_samples: Number of samples to extract
        output_jsonl_path: Path where JSONL file will be saved
        seed: Random seed for reproducible sampling
        train_dataset_cache: Path to training dataset cache directory (for reusing training data)
        dataset_config: Optional separate dataset configuration for FOCUS

    Returns:
        Path to the created JSONL file

    NOTE: Parameters affecting FOCUS training data (num_samples, seed, dataset source) are
    tracked via TokenizerConfig in artifact_configs.py since this data is only used
    for tokenizer training.
    """
    if os.path.exists(output_jsonl_path):
        print(f"JSONL data already exists at {output_jsonl_path}, skipping generation", file=sys.stderr)
        return output_jsonl_path

    print(f"Preparing FOCUS training data: {num_samples} samples", file=sys.stderr)

    # If dataset_config provided, load that dataset; otherwise use training dataset
    if dataset_config is not None:
        # Import here to avoid circular dependency (dataset_utils imports tokenizer)
        from lapt.dataset_utils import load_untokenized_dataset
        # Use the JSONL output directory as the cache for the FOCUS dataset
        focus_cache = os.path.dirname(output_jsonl_path)
        untokenized_path = load_untokenized_dataset(
            dataset_config=dataset_config,
            cache_dir=focus_cache,
            dev_size=-1  # FOCUS doesn't need dev split (only uses train for tokenizer/embeddings)
        )
        dataset = load_from_disk(untokenized_path)
    else:
        if train_dataset_cache is None:
            raise ValueError("Either train_dataset_cache or dataset_config must be provided")
        untokenized_path = os.path.join(train_dataset_cache, "untokenized")
        if os.path.exists(untokenized_path):
            dataset = load_from_disk(untokenized_path)
        else:
            raise FileNotFoundError(
                f"Untokenized dataset not found at {untokenized_path}. "
                "Please ensure the dataset is loaded first."
            )

    train_data = dataset['train']
    total_samples = len(train_data)

    if num_samples > total_samples:
        print(
            f"Warning: Requested {num_samples} samples but dataset only has {total_samples}. "
            f"Using all available samples.",
            file=sys.stderr
        )
        num_samples = total_samples

    random.seed(seed)
    indices = random.sample(range(total_samples), num_samples)
    # Sort indices for efficient sequential access to memory-mapped dataset
    indices.sort()

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        written_count = 0
        for idx in indices:
            text = train_data[idx]['text']
            # Skip blank lines
            if text.strip():
                json.dump({'text': text}, f, ensure_ascii=False)
                f.write('\n')
                written_count += 1

        if written_count < num_samples:
            print(
                f"Warning: Filtered out {num_samples - written_count} blank lines from FOCUS training data",
                file=sys.stderr
            )

    print(f"JSONL data saved to {output_jsonl_path}", file=sys.stderr)
    return output_jsonl_path


FOCUS_EMBS_SUBDIR = 'focus_embs'
LEGACY_INPUT_NAME = 'focus_input_embeddings.pt'
LEGACY_OUTPUT_NAME = 'focus_output_embeddings.pt'


def _sidecar_paths(cache_dir: str, embedding_hash: str) -> tuple[str, str, str]:
    """Return (input_pt, output_pt, meta_yaml) sidecar paths for a hash."""
    sub = os.path.join(cache_dir, FOCUS_EMBS_SUBDIR)
    return (
        os.path.join(sub, f"{embedding_hash}.input.pt"),
        os.path.join(sub, f"{embedding_hash}.output.pt"),
        os.path.join(sub, f"{embedding_hash}.meta.yaml"),
    )


def _enumerate_cached_embeddings(cache_dir: str) -> list[tuple[str, str | None]]:
    """
    List all cached FOCUS embedding sets under cache_dir.

    Returns a list of (input_pt_path, output_pt_path_or_None) tuples covering
    both the new focus_embs/<hash>.input.pt layout and the legacy unhashed
    files at the tokenizer-dir root.
    """
    found: list[tuple[str, str | None]] = []

    sub = os.path.join(cache_dir, FOCUS_EMBS_SUBDIR)
    if os.path.isdir(sub):
        for input_pt in sorted(glob.glob(os.path.join(sub, '*.input.pt'))):
            output_pt = input_pt[: -len('.input.pt')] + '.output.pt'
            found.append((input_pt, output_pt if os.path.exists(output_pt) else None))

    legacy_input = os.path.join(cache_dir, LEGACY_INPUT_NAME)
    if os.path.exists(legacy_input):
        legacy_output = os.path.join(cache_dir, LEGACY_OUTPUT_NAME)
        found.append(
            (legacy_input, legacy_output if os.path.exists(legacy_output) else None)
        )

    return found


def resolve_cached_embedding_paths(
    cache_dir: str | None,
    embedding_hash: str | None,
    reuse_policy: str | None,
) -> tuple[str, str | None] | None:
    """
    Resolve which cached FOCUS embedding sidecar to load, if any.

    Args:
        cache_dir: Tokenizer directory containing focus_embs/ and/or legacy
            unhashed embedding files. None disables caching.
        embedding_hash: Hash for this run's mix + FOCUS knobs.
        reuse_policy: One of:
            - None: only load on exact hash match.
            - "any": accept any single cached set across both layouts; ambiguous
              if more than one exists.
            - "<hash>": load that specific sidecar; error if absent.

    Returns:
        (input_pt_path, output_pt_path_or_None) if a cache hit, else None.
    """
    if cache_dir is None:
        return None

    if reuse_policy == 'any':
        candidates = _enumerate_cached_embeddings(cache_dir)
        if len(candidates) == 0:
            return None
        if len(candidates) > 1:
            listing = "\n  ".join(p for p, _ in candidates)
            raise ValueError(
                f"focus.reuse_embeddings='any' but {len(candidates)} cached "
                f"embedding sets exist under {cache_dir}:\n  {listing}\n"
                f"Specify focus.reuse_embeddings=<hash> to disambiguate."
            )
        return candidates[0]

    if reuse_policy and reuse_policy != 'any':
        # Explicit hash request.
        input_pt, output_pt, _ = _sidecar_paths(cache_dir, reuse_policy)
        if not os.path.exists(input_pt):
            raise ValueError(
                f"focus.reuse_embeddings='{reuse_policy}' but no embeddings "
                f"found at {input_pt}"
            )
        return (input_pt, output_pt if os.path.exists(output_pt) else None)

    # Default: exact-hash match only.
    if embedding_hash is None:
        return None
    input_pt, output_pt, _ = _sidecar_paths(cache_dir, embedding_hash)
    if os.path.exists(input_pt):
        return (input_pt, output_pt if os.path.exists(output_pt) else None)
    return None


def _copy_embeddings_directly(
    source_model,
    source_token_strings: set[str],
    source_tokenizer: PreTrainedTokenizerBase,
    target_token_strings: list[str],
    has_separate_output: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Build new embedding matrices by copying source embeddings for each target token.

    Used when all target tokens exist in the source vocabulary (e.g., prune-only PTEx),
    where FOCUS would crash because its novel-token matrix is empty.

    Args:
        source_model: Source pretrained model
        source_token_strings: Set of all token strings in the source vocabulary
        source_tokenizer: Tokenizer for the source model
        target_token_strings: Ordered list of token strings in the target vocabulary
        has_separate_output: Whether the model uses separate input/output embeddings

    Returns:
        Tuple of (input_embeddings, output_embeddings); output_embeddings is None
        if the model uses tied embeddings.
    """
    # Build source string → id lookup
    source_string_to_id = {
        source_tokenizer.convert_ids_to_tokens(i): i
        for i in range(len(source_tokenizer))
    }

    source_input_embeddings = source_model.get_input_embeddings().weight.detach()
    hidden_dim = source_input_embeddings.shape[1]
    target_vocab_size = len(target_token_strings)

    new_input_embeddings = torch.zeros(target_vocab_size, hidden_dim)
    missing_count = 0
    for target_id, token_str in enumerate(target_token_strings):
        source_id = source_string_to_id.get(token_str)
        if source_id is not None:
            new_input_embeddings[target_id] = source_input_embeddings[source_id]
        else:
            # Fallback to mean initialization; caller already verified this shouldn't happen
            new_input_embeddings[target_id] = source_input_embeddings.mean(dim=0)
            missing_count += 1

    if missing_count > 0:
        print(
            f"Warning: {missing_count} target tokens not found in source vocabulary; "
            f"initialized from embedding mean.",
            file=sys.stderr,
        )

    new_output_embeddings = None
    if has_separate_output:
        source_output_embeddings = source_model.get_output_embeddings().weight.detach()
        new_output_embeddings = torch.zeros(target_vocab_size, hidden_dim)
        for target_id, token_str in enumerate(target_token_strings):
            source_id = source_string_to_id.get(token_str)
            if source_id is not None:
                new_output_embeddings[target_id] = source_output_embeddings[source_id]
            else:
                new_output_embeddings[target_id] = source_output_embeddings.mean(dim=0)

    return new_input_embeddings, new_output_embeddings


def apply_focus_initialization(
    source_model,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    training_data_path: str | None,
    fasttext_model_min_count: int = 4,
    cache_dir: str | None = None,
    embedding_hash: str | None = None,
    embedding_meta: dict | None = None,
    reuse_policy: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Apply FOCUS to generate new input embeddings and optionally output embeddings.

    Embedding tensors are cached as sidecars under cache_dir/focus_embs/, keyed
    by embedding_hash so that the same tokenizer can host multiple cached
    embedding sets (one per FOCUS-training mix).

    Args:
        source_model: Source pretrained model
        source_tokenizer: Tokenizer for the source model
        target_tokenizer: Target language-specific tokenizer
        training_data_path: Path to JSONL training data for FOCUS. May be None
            only when a cache hit is guaranteed by reuse_policy.
        fasttext_model_min_count: Minimum occurrences for FastText embeddings (default: 4)
        cache_dir: Directory where embeddings should be cached (typically tokenizer directory)
        embedding_hash: 8-char hash of mix + FOCUS knobs; used as sidecar key.
        embedding_meta: Provenance dict written next to a freshly-computed sidecar.
        reuse_policy: None (strict hash match), "any" (load the sole cached set
            across both new and legacy layouts), or an explicit hash string.

    Returns:
        Tuple of (input_embeddings, output_embeddings)
        output_embeddings will be None if model ties word embeddings
    """
    cached = resolve_cached_embedding_paths(cache_dir, embedding_hash, reuse_policy)
    if cached is not None:
        input_pt, output_pt = cached
        print(f"Loading cached FOCUS embeddings from {input_pt}", file=sys.stderr)
        new_input_embeddings = torch.load(input_pt, weights_only=True)

        has_separate_output = (
            hasattr(source_model.config, 'tie_word_embeddings')
            and not source_model.config.tie_word_embeddings
        )

        if has_separate_output:
            if output_pt is None:
                print(
                    f"Warning: Found cached input embeddings at {input_pt} but "
                    f"missing matching output embeddings. Regenerating both.",
                    file=sys.stderr,
                )
            else:
                new_output_embeddings = torch.load(output_pt, weights_only=True)
                print(
                    f"FOCUS embeddings loaded from cache. Vocab size: {len(target_tokenizer)}",
                    file=sys.stderr,
                )
                return new_input_embeddings, new_output_embeddings
        else:
            print(
                f"FOCUS embeddings loaded from cache. Vocab size: {len(target_tokenizer)}",
                file=sys.stderr,
            )
            return new_input_embeddings, None


    # Check whether any target tokens are absent from the source vocabulary.
    # FOCUS crashes (fastdist TypingError) when the novel-token set is empty,
    # because the cosine similarity matrix degenerates to a scalar.
    # In that case (pure prune-only tokenizer) skip FOCUS and copy directly.
    source_token_strings = {
        source_tokenizer.convert_ids_to_tokens(i)
        for i in range(len(source_tokenizer))
    }
    target_token_strings = [
        target_tokenizer.convert_ids_to_tokens(i)
        for i in range(len(target_tokenizer))
    ]
    novel_token_count = sum(1 for t in target_token_strings if t not in source_token_strings)

    has_separate_output = (
        hasattr(source_model.config, 'tie_word_embeddings')
        and not source_model.config.tie_word_embeddings
    )

    if novel_token_count == 0:
        print(
            "All target tokens exist in source vocabulary — skipping FOCUS, "
            "copying embeddings directly.",
            file=sys.stderr,
        )
        new_input_embeddings, new_output_embeddings = _copy_embeddings_directly(
            source_model=source_model,
            source_token_strings=source_token_strings,
            source_tokenizer=source_tokenizer,
            target_token_strings=target_token_strings,
            has_separate_output=has_separate_output,
        )
    else:
        if training_data_path is None:
            raise ValueError(
                "apply_focus_initialization: training_data_path is required "
                "when novel tokens are present and no cached embeddings were loaded."
            )
        print(
            f"Applying FOCUS to initialize embeddings "
            f"({novel_token_count} novel tokens)",
            file=sys.stderr,
        )

        try:
            from deepfocus import FOCUS
        except ImportError:
            raise ImportError(
                "deepfocus package not found. Please install it with: pip install deepfocus"
            )

        source_embeddings = source_model.get_input_embeddings().weight

        new_input_embeddings = FOCUS(
            source_embeddings=source_embeddings,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            target_training_data_path=training_data_path,
            fasttext_model_min_count=fasttext_model_min_count
        )

        new_output_embeddings = None
        if has_separate_output:
            print("Model uses separate output embeddings, applying FOCUS to output embeddings", file=sys.stderr)
            source_output_embeddings = source_model.get_output_embeddings().weight
            new_output_embeddings = FOCUS(
                source_embeddings=source_output_embeddings,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                target_training_data_path=training_data_path,
                fasttext_model_min_count=fasttext_model_min_count
            )

    print(f"Embedding initialization complete. New vocab size: {len(target_tokenizer)}", file=sys.stderr)

    # Cache the embeddings if cache_dir + embedding_hash provided
    if cache_dir is not None and embedding_hash is not None:
        input_pt, output_pt, meta_yaml = _sidecar_paths(cache_dir, embedding_hash)
        os.makedirs(os.path.dirname(input_pt), exist_ok=True)

        print(f"Saving FOCUS embeddings to {input_pt}", file=sys.stderr)
        torch.save(new_input_embeddings, input_pt)
        if new_output_embeddings is not None:
            torch.save(new_output_embeddings, output_pt)
        if embedding_meta is not None:
            with open(meta_yaml, 'w') as f:
                yaml.dump(embedding_meta, f, default_flow_style=False, sort_keys=False)

        print("FOCUS embeddings cached", file=sys.stderr)

    return new_input_embeddings, new_output_embeddings
