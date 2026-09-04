"""
Utilities for loading and processing datasets for language-adaptive pretraining.

This module handles downloading OSCAR corpus data, converting it to line-based format,
tokenizing with provided tokenizers, and caching results.
"""

import glob
import json
import os
import shutil
import sys

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer

from lapt.artifact_configs import (
    TokenizedDatasetConfig,
    multinomial_mix_slug,
    resolve_dev_size,
)
from lapt.sources import (
    ConcatDataset,
    HuggingFaceDataset,
    InstructionHFDataset,
    InstructionJsonlDataset,
    MultinomialDataset,
    OscarDataset,
    PlaintextDataset,
)
from lapt.sources.base import SourceDataset
from lapt.sources.concat import source_id
from lapt.sources.factory import build_source
from lapt.sources.sampling import compute_sampling_probs
from lapt.sources.text_processing import (
    read_instruction_jsonl,
)
from lapt_core.artifacts import ArtifactConfig, CachedArtifact
from lapt_core.dataset_artifacts import DatasetArtifact


def load_untokenized_dataset(
    dataset_config,
    cache_dir: str,
    dev_size: float = None,
    seed: int = 1,
) -> str:
    """
    Load untokenized dataset based on configuration.

    Thin path-returning wrapper over `build_source`, which maps the config's
    ``type`` to a source class through the registry and applies any
    ``substitutions`` the entry carries. Kept so callers that exchange paths
    keep working; new code should use `build_source` and hold the artifact.

    Args:
        dataset_config: Dataset configuration object with type and source info
        cache_dir: Base directory for caching dataset artifacts
        dev_size: Fraction of data for dev set (only used for multinomial sampling)
        seed: Global random seed. Sources that subsample record it, so it must
            reach them rather than being read only from the global RNG state.

    Returns:
        Path to the untokenized dataset

    NOTE: The parameters a source is keyed on are declared by its class in
    `lapt/sources/`, in a single `config()` method that doubles as the
    cache-validation record. Adding a dataset type means adding a class there
    and registering it; there is no separate list to keep in step.
    """
    source = build_source(cache_dir, dataset_config, seed, dev_size)
    source.resolve()
    return source.path


def build_untokenized_source(args: DictConfig) -> SourceDataset:
    """Construct the untokenized corpus source a full Hydra config describes.

    The single entry point from the training pipeline into `lapt.sources`. The
    returned artifact owns its cache path, the config record beside it, and the
    validate-or-build decision, so a caller resolves it and reads `.path`.

    Args:
        args: Full Hydra configuration, read for `dataset` and `seed`.

    Returns:
        An unresolved source. For a multinomial mix this resolves into the
        mix-keyed subfolder while per-source subdirectories stay shared at the
        parent level; with substitutions configured it is the `_sub_{digest}`
        sibling of whatever the underlying type produced.
    """
    return build_source(
        args.dataset.cache_dir,
        args.dataset,
        args.seed,
        resolve_dev_size(args),
    )


def _load_oscar_dataset(cache_dir: str, language_code: str) -> str:
    """
    Load or download OSCAR dataset for a specific language.

    Thin path-returning wrapper over `OscarDataset`; see `_load_plaintext_dataset`.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        language_code: Two-letter language code for OSCAR corpus

    Returns:
        Path to the untokenized dataset
    """
    source = OscarDataset(cache_dir, language_code)
    source.resolve()
    return source.path


def _load_huggingface_dataset(
    cache_dir: str,
    name: str,
    config: str = None,
    split: str = 'train',
    text_column: str = 'text',
    max_samples: int = None,
    min_words_per_line: int = None,
    oversampling_factor: int = 3,
    split_into_lines: bool = True,
    seed: int = 1,
) -> str:
    """
    Load a generic HuggingFace dataset.

    Thin path-returning wrapper over `HuggingFaceDataset`; see
    `_load_plaintext_dataset`.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        name: HuggingFace dataset name (e.g., 'wikitext', 'c4')
        config: Dataset configuration/subset (e.g., 'wikitext-103-v1'), optional
        split: Which split to load (default: 'train')
        text_column: Name of the column containing text (default: 'text')
        max_samples: Maximum number of examples to load, uses streaming if specified
        min_words_per_line: Minimum number of space-separated words per example
        oversampling_factor: Download this many times more documents than estimated
            needed, to maintain document diversity (default: 3)
        split_into_lines: Split each document into one example per line (default: True)
        seed: Seed for the subsample taken when max_samples is set

    Returns:
        Path to the untokenized dataset
    """
    source = HuggingFaceDataset(
        cache_dir,
        name,
        config=config,
        split=split,
        text_column=text_column,
        max_samples=max_samples,
        min_words_per_line=min_words_per_line,
        oversampling_factor=oversampling_factor,
        split_into_lines=split_into_lines,
        seed=seed,
    )
    source.resolve()
    return source.path


def _load_plaintext_dataset(cache_dir: str, file_path: str) -> str:
    """
    Load plaintext file(s) and convert to dataset format.

    Thin path-returning wrapper over `PlaintextDataset`, which owns the cache
    path, the config record, and the validate-or-build decision. Kept so the
    dispatcher and the composite loaders can keep exchanging paths while the
    remaining source types are converted.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        file_path: Path to plaintext file (one line per training example)

    Returns:
        Path to the untokenized dataset
    """
    source = PlaintextDataset(cache_dir, file_path)
    source.resolve()
    return source.path


def _load_plaintext_dir_dataset(
    cache_dir: str,
    directory: str,
    pattern: str = '*.txt',
    seed: int = 1,
) -> str:
    """
    Load all plaintext files from a directory and concatenate them.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        directory: Directory containing text files
        pattern: Glob pattern for matching files (e.g., "*.txt", "*.on.txt")

    Returns:
        Path to the untokenized concatenated dataset
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")

    # Find all matching files
    file_paths = sorted(glob.glob(os.path.join(directory, pattern)))

    if not file_paths:
        raise ValueError(f"No files found matching pattern '{pattern}' in {directory}")

    print(f"Found {len(file_paths)} files matching '{pattern}' in {directory}", file=sys.stderr)

    # Create sources list for concat (reuse plaintext loader for each file)
    sources = [
        {'type': 'plaintext', 'path': path}
        for path in file_paths
    ]

    # Reuse concat implementation
    return _load_concat_dataset(cache_dir, sources, seed=seed)


def _load_instruction_jsonl_dataset(cache_dir: str, file_path: str) -> str:
    """
    Load instruction-tuning data from JSONL file(s).

    Thin path-returning wrapper over `InstructionJsonlDataset`; see
    `_load_plaintext_dataset`.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        file_path: Path to JSONL file

    Returns:
        Path to the untokenized dataset (with 'prompt' and 'response' columns)
    """
    source = InstructionJsonlDataset(cache_dir, file_path)
    source.resolve()
    return source.path


def _load_instruction_hf_dataset(
    cache_dir: str,
    name: str,
    config: str | None = None,
    split: str = 'train',
    messages_column: str = 'messages',
    prompt_template: str = '{user} Response:',
    response_template: str = ' {assistant}',
    max_samples: int | None = None,
    seed: int = 1,
) -> str:
    """
    Load an instruction-tuning dataset from HuggingFace.

    Thin path-returning wrapper over `InstructionHFDataset`; see
    `_load_plaintext_dataset`.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        name: HuggingFace dataset name (e.g., 'HuggingFaceH4/no_robots')
        config: Dataset configuration/subset, optional
        split: Which split to load (default: 'train')
        messages_column: Column name holding the list of {role, content} dicts
        prompt_template: Format string with a {user} placeholder
        response_template: Format string with an {assistant} placeholder
        max_samples: Optional cap on number of examples (random subsample)
        seed: Seed for that subsample

    Returns:
        Path to the untokenized dataset (with 'prompt' and 'response' columns)
    """
    source = InstructionHFDataset(
        cache_dir,
        name,
        config=config,
        split=split,
        messages_column=messages_column,
        prompt_template=prompt_template,
        response_template=response_template,
        max_samples=max_samples,
        seed=seed,
    )
    source.resolve()
    return source.path


def _load_concat_dataset(
    cache_dir: str,
    sources: list,
    parent_id: str = None,
    seed: int = 1,
) -> str:
    """
    Concatenate multiple dataset sources into a single dataset.

    Thin path-returning wrapper over `ConcatDataset`; see `_load_plaintext_dataset`.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        sources: List of dataset source configurations (may include 'id' field for naming)
        parent_id: Optional id from parent concat config (used for fallback naming)
        seed: Global random seed, passed to children that subsample

    Returns:
        Path to the untokenized concatenated dataset
    """
    source = ConcatDataset(cache_dir, sources, parent_id=parent_id, seed=seed)
    source.resolve()
    return source.path


def _load_multinomial_dataset(
    cache_dir: str,
    sources: list,
    alpha: float,
    total_samples: int,
    dev_size: float = None,
    seed: int = 1,
) -> str:
    """
    Sample from multiple dataset sources using temperature-scaled multinomial sampling.

    Thin path-returning wrapper over `MultinomialDataset`; see `_load_plaintext_dataset`.

    Args:
        cache_dir: Base directory for caching dataset artifacts
        sources: List of dataset source configurations
        alpha: Temperature parameter for reweighting (< 1 upsamples smaller datasets)
        total_samples: Total number of training examples to sample
        dev_size: Global default fraction of each source to use for dev set, or -1 to skip
        seed: Global random seed, which selects the sampled examples

    Returns:
        Path to the untokenized sampled dataset (DatasetDict with train and per-source dev splits)
    """
    source = MultinomialDataset(cache_dir, sources, alpha, total_samples, dev_size, seed=seed)
    source.resolve()
    return source.path


def _tokenize_plaintext_with_labels(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> dict:
    """
    Tokenize plaintext examples and add labels for causal LM loss.

    Used for plaintext splits in mixed instruction/plaintext datasets, where the
    DataCollatorForInstructionTuning expects all examples to have 'labels'.
    For plaintext, labels = input_ids (loss on all tokens).

    Args:
        examples: Batch with 'text' field
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels' fields
    """
    tokenized = tokenizer(
        examples['text'], max_length=max_length, truncation=True
    )
    # For plaintext, labels = input_ids (standard causal LM loss on all tokens)
    tokenized['labels'] = [ids.copy() for ids in tokenized['input_ids']]
    return tokenized


def _tokenize_instruction_examples(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> dict:
    """
    Tokenize instruction examples with label masking.

    For each example, tokenizes prompt and response separately, then concatenates.
    Creates labels where prompt tokens are masked (-100) and only response tokens
    contribute to the loss.

    Also handles mixed datasets where some examples have prompt/response (instruction)
    and others have only text (plaintext). Plaintext examples get labels = input_ids
    (standard causal LM loss on all tokens).

    Args:
        examples: Batch with 'prompt' and 'response' fields, optionally 'text'
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length (prompt + response combined)

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels' fields
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    # Get text column if it exists (for mixed datasets)
    texts = examples.get('text', [None] * len(examples['prompt']))

    for prompt, response, text in zip(examples['prompt'], examples['response'], texts):
        # Check if this is an instruction example or plaintext
        is_instruction = prompt is not None and response is not None

        if is_instruction:
            # Instruction example: tokenize prompt and response separately
            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False
            )

            response_tokens = tokenizer(
                response,
                add_special_tokens=False,
                truncation=False
            )

            # Append EOS so the model learns to terminate responses
            response_ids = response_tokens['input_ids'] + [tokenizer.eos_token_id]
            response_mask = response_tokens['attention_mask'] + [1]

            # Concatenate
            # TODO: fix linting issue here
            input_ids = prompt_tokens['input_ids'] + response_ids
            attention_mask = prompt_tokens['attention_mask'] + response_mask

            # Create labels: -100 for prompt (masked), actual tokens for response
            prompt_length = len(prompt_tokens['input_ids'])
            labels = [-100] * prompt_length + response_ids
        else:
            # Plaintext example: standard tokenization, labels = input_ids
            if text is None:
                raise ValueError(
                    "Example has neither valid prompt/response nor text. "
                    "Mixed datasets must have 'text' for plaintext examples."
                )

            tokens = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )

            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            # Standard causal LM: predict all tokens
            labels = list(input_ids)

        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'labels': all_labels
    }


# Plan / per-source tokenization for multinomial mixes.
# The training-time multinomial pipeline upsamples by repeating row indices
# rather than duplicating tokenized rows. The pieces below implement that:
#
#   <cache_dir>/<source_id>/untokenized/                      (text, mix-agnostic)
#   <cache_dir>/<source_id>/tokenized_<tok>_ml<L>_{labels,nolabels}/  (mix-agnostic)
#   <mix_dir>/train_plan.npz                                  (shuffled global indices)
#   <mix_dir>/dev/                                            (per-source tokenized dev)
#
# The training Dataset is built as
# `concatenate_datasets(per_source_tokenized).select(global_indices)`,
# which produces an Arrow indices-mapped view; no rows are duplicated.

def _tokenized_source_dirname(
    tokenizer_id: str,
    max_length: int,
    add_labels: bool,
    variant_suffix: str = "",
) -> str:
    """
    Build the per-source tokenized cache directory name.

    The optional ``variant_suffix`` carries the untokenized variant (e.g.
    ``_sub_<hash>`` for a substituted source) so a substituted source tokenizes
    into a distinct cache rather than colliding with the raw one. An empty
    suffix reproduces the original ``tokenized_<id>_ml<L>_<labels>`` name, so
    pre-existing raw caches stay valid.
    """
    label_suffix = "labels" if add_labels else "nolabels"
    return f"tokenized{variant_suffix}_{tokenizer_id}_ml{max_length}_{label_suffix}"


def _dev_splits_dirname(
    tokenizer_id: str,
    max_length: int,
    add_labels: bool,
) -> str:
    """
    Build the mix-level tokenized dev-splits cache directory name.

    The dev splits hold token ids, so the cache must be keyed by the same
    parameters as the per-source tokenized caches. The mix slug that names the
    parent directory deliberately excludes the tokenizer (sources and plans are
    shared across tokenizers), so without this suffix a dev cache written by one
    model's tokenizer would be silently reused by a model with a different
    tokenizer.
    """
    label_suffix = "labels" if add_labels else "nolabels"
    return f"dev_{tokenizer_id}_ml{max_length}_{label_suffix}"


def _source_has_instruction_columns(untokenized_path: str) -> bool:
    """Return True if the source's untokenized 'train' split has prompt/response columns."""
    data = load_from_disk(untokenized_path)
    if isinstance(data, DatasetDict):
        split = data['train'] if 'train' in data else data[list(data.keys())[0]]
    else:
        split = data
    cols = split.column_names
    return 'prompt' in cols and 'response' in cols


class TokenizedSourceArtifact(DatasetArtifact):
    """One mix source's rows, tokenized once and cached independent of any mix.

    Detects instruction vs plaintext from the source's own column schema at
    build time. `add_labels` is a decision the caller makes mix-wide (any
    instruction source in the mix forces labels for every source), not
    something this class infers.

    Constructed from the *exact* untokenized path a source artifact returned
    (`SourceDataset.resolve()` / `.path`) -- never re-derived as
    `{source_cache_dir}/untokenized`. A source declaring `substitutions`
    returns an `untokenized_sub_<hash>` variant; tokenizing a re-derived path
    instead would silently tokenize the raw, unsubstituted text. This was a
    real regression in the pre-artifact version of this code -- see
    architecture.md's "Two tokenization paths" note -- so the constructor
    takes the resolved path directly rather than a cache_dir it could
    re-derive one from.
    """

    name = "tokenized_source"
    config_filename = "tokenized_config.yaml"

    def __init__(
        self,
        untokenized_path: str,
        tokenizer: PreTrainedTokenizer,
        tokenizer_id: str,
        max_length: int,
        add_labels: bool,
    ):
        """Initialize the artifact.

        Args:
            untokenized_path: Path a source artifact's `.resolve()` returned.
            tokenizer: Tokenizer to use.
            tokenizer_id: Stable identifier embedded in the cache directory name.
            max_length: Truncation length for tokenization.
            add_labels: Whether to materialize a labels column for plaintext
                rows. Required when this source is mixed alongside an
                instruction source so the data collator sees a uniform schema.

        Raises:
            FileNotFoundError: If `untokenized_path` does not exist.
        """
        if not os.path.exists(untokenized_path):
            raise FileNotFoundError(
                f"Source untokenized cache missing at {untokenized_path}; "
                "resolve the source artifact first."
            )
        super().__init__(root=os.path.dirname(untokenized_path))
        self.untokenized_path = untokenized_path
        self.tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id
        self.max_length = max_length
        self.add_labels = add_labels

    @property
    def path(self) -> str:
        # The variant suffix ('', or '_sub_<hash>' for a substituted source)
        # comes from the untokenized dir name, so a substituted source
        # tokenizes into its own cache rather than colliding with the raw one.
        untokenized_name = os.path.basename(self.untokenized_path)
        variant_suffix = untokenized_name[len("untokenized"):]
        dirname = _tokenized_source_dirname(
            self.tokenizer_id, self.max_length, self.add_labels, variant_suffix
        )
        return os.path.join(self.root, dirname)

    def config(self) -> dict:
        return {
            'tokenizer_id': self.tokenizer_id,
            'max_length': self.max_length,
            'add_labels': self.add_labels,
        }

    def build(self, deps) -> Dataset:
        """Tokenize the source's rows into a single (non-split) `Dataset`.

        Train/dev partitioning is a mix-time concern and does not affect this
        cache -- it holds every row of the source.
        """
        print(
            f"Tokenizing source at {self.root} -> {os.path.basename(self.path)} "
            f"(tokenizer={self.tokenizer_id}, max_length={self.max_length}, "
            f"add_labels={self.add_labels})",
            file=sys.stderr,
        )

        ds = load_from_disk(self.untokenized_path)
        # Per-source untokenized caches are stored as DatasetDict with a single 'train' split.
        if isinstance(ds, DatasetDict):
            if 'train' not in ds:
                raise ValueError(
                    f"Source untokenized cache at {self.untokenized_path} has no "
                    f"'train' split (found splits: {list(ds.keys())})"
                )
            data = ds['train']
        else:
            data = ds

        cols = data.column_names
        has_instruction = 'prompt' in cols and 'response' in cols
        has_text = 'text' in cols

        if has_instruction:
            cols_to_remove = ['prompt', 'response'] + (['text'] if has_text else [])
            tokenized = data.map(
                lambda examples: _tokenize_instruction_examples(
                    examples, self.tokenizer, self.max_length
                ),
                batched=True,
                remove_columns=cols_to_remove,
            )
        elif has_text:
            if self.add_labels:
                tokenized = data.map(
                    lambda examples: _tokenize_plaintext_with_labels(
                        examples, self.tokenizer, self.max_length
                    ),
                    batched=True,
                    remove_columns='text',
                )
            else:
                tokenized = data.map(
                    lambda examples: self.tokenizer(
                        examples['text'], max_length=self.max_length, truncation=True
                    ),
                    batched=True,
                    remove_columns='text',
                )
        else:
            raise ValueError(
                f"Source at {self.untokenized_path} has neither instruction columns "
                f"(prompt/response) nor a 'text' column. Found: {cols}"
            )

        print(f"  Tokenized {len(tokenized)} rows", file=sys.stderr)
        return tokenized


class TrainPlanArtifact(CachedArtifact):
    """The upsampled training split as shuffled global indices into the
    concatenation of per-source tokenized datasets -- no row duplication on
    disk.

    `source_sizes` lives in `config()` rather than being checked ad hoc: a
    source whose tokenized cache changed size now surfaces as an ordinary
    config mismatch (with the standard remediation message) instead of a
    bespoke `ValueError`.

    Behavior/layout change from the pre-artifact version: the plan used to be
    a single `train_plan.npz` file directly in the mix directory; it is now a
    `train_plan/` subdirectory holding `plan.npz` plus the config record
    `CachedArtifact` needs. A plan predating this change is simply
    regenerated once -- cheap, since it is a permutation over already-cached
    tokenized data, not a rebuild of anything expensive.
    """

    name = "train_plan"

    def __init__(
        self,
        mix_dir: str,
        source_ids: list[str],
        source_sizes: list[int],
        samples_per_source: list[int],
        shuffle_seed: int,
        train_pools: list[np.ndarray] | None = None,
    ):
        """Initialize the artifact.

        Args:
            mix_dir: The mix's cache directory.
            source_ids: Source identifiers, in concatenation order.
            source_sizes: Each source's tokenized row count, in the same order.
            samples_per_source: Target training sample count per source.
            shuffle_seed: Seed for shuffling the global training plan.
            train_pools: Each source's train-pool row indices (into its own
                tokenized dataset), required only when the cache turns out to
                be cold. Callers that can tell in advance the cache is warm
                may skip preparing this.
        """
        super().__init__(root=mix_dir)
        self.source_ids = source_ids
        self.source_sizes = source_sizes
        self.samples_per_source = samples_per_source
        self.shuffle_seed = shuffle_seed
        self.train_pools = train_pools

    def config(self) -> dict:
        return {
            'source_ids': list(self.source_ids),
            'source_sizes': list(self.source_sizes),
            'samples_per_source': list(self.samples_per_source),
            'shuffle_seed': self.shuffle_seed,
        }

    def build(self, deps) -> np.ndarray:
        if self.train_pools is None:
            raise ValueError(
                f"No cached train plan at {self.path}, but train_pools was not "
                "provided to build one."
            )

        offsets = np.zeros(len(self.source_sizes) + 1, dtype=np.int64)
        for i, size in enumerate(self.source_sizes):
            offsets[i + 1] = offsets[i] + size

        rng = np.random.default_rng(self.shuffle_seed)
        chunks = []
        for src_idx, (pool, n_samples) in enumerate(
            zip(self.train_pools, self.samples_per_source)
        ):
            if n_samples == 0:
                continue
            pool_size = len(pool)
            if n_samples <= pool_size:
                # Sample without replacement within the source's train pool.
                local_indices = rng.choice(pool_size, size=n_samples, replace=False)
            else:
                # Exhaust-first: every pool row at least once, then sample remainder.
                extra = rng.integers(0, pool_size, size=n_samples - pool_size)
                local_indices = np.concatenate(
                    [np.arange(pool_size, dtype=np.int64), extra]
                )
            chunks.append(pool[local_indices].astype(np.int64) + offsets[src_idx])

        global_indices = np.concatenate(chunks)
        rng.shuffle(global_indices)
        print(f"Built train plan ({len(global_indices)} samples)", file=sys.stderr)
        return global_indices

    def write(self, value: np.ndarray, path: str) -> None:
        np.savez(os.path.join(path, "plan.npz"), global_indices=value)

    def read(self, path: str) -> np.ndarray:
        return np.load(os.path.join(path, "plan.npz"))['global_indices']


class DevSplitsArtifact(DatasetArtifact):
    """Per-source dev splits at one mix's tokenizer/max_length/labels key.

    The mix directory itself is tokenizer-agnostic (sources and the train
    plan are shared across tokenizers), so this cache -- which holds token
    ids -- must carry its own tokenizer key, or a second model reusing the
    mix would silently inherit the first model's dev token ids.

    `source_ids` is the full set of sources in the mix (cheap, known from
    config alone), not just the ones that end up with a nonempty dev split --
    that lets `config()` be answered without touching `dev_splits`, so a
    caller that already knows the cache is warm can construct without
    materializing anything.
    """

    def __init__(
        self,
        mix_dir: str,
        tokenizer_id: str,
        max_length: int,
        add_labels: bool,
        source_ids: list[str],
        dev_splits: dict[str, Dataset] | None = None,
    ):
        """Initialize the artifact.

        Args:
            mix_dir: The mix's cache directory.
            tokenizer_id: Stable identifier embedded in the cache directory name.
            max_length: Truncation length used to tokenize the sources.
            add_labels: Whether plaintext sources were tokenized with labels.
            source_ids: Every source identifier in the mix.
            dev_splits: `{source_id: Dataset}` for sources that have a nonempty
                dev split, required only when the cache turns out to be cold.
        """
        super().__init__(root=mix_dir)
        self.tokenizer_id = tokenizer_id
        self.max_length = max_length
        self.add_labels = add_labels
        self.source_ids = source_ids
        self.dev_splits = dev_splits

    @property
    def path(self) -> str:
        return os.path.join(
            self.root,
            _dev_splits_dirname(self.tokenizer_id, self.max_length, self.add_labels),
        )

    def config(self) -> dict:
        return {
            'tokenizer_id': self.tokenizer_id,
            'max_length': self.max_length,
            'add_labels': self.add_labels,
            'source_ids': list(self.source_ids),
        }

    def exists(self) -> bool:
        """Report a cache with no config record as absent, so it is rebuilt.

        This stage had no config tracking at all before it became an artifact
        -- the old code checked only whether the directory was there -- so
        every dev cache written before this port carries no record.
        `CachedArtifact` would refuse those outright, which is the right call
        when an unverifiable cache might have trained a model. Here it is a
        `.select()` over rows that are already tokenized: seconds to redo, and
        rebuilding is self-healing where refusing would mean deleting 26
        directories by hand.
        """
        return super().exists() and os.path.exists(self.config_path)

    def build(self, deps) -> DatasetDict:
        if self.dev_splits is None:
            raise ValueError(
                f"No cached dev splits at {self.path}, but dev_splits was not "
                "provided to build them."
            )
        if self.dev_splits:
            print(
                f"Saving tokenized dev splits ({list(self.dev_splits)})",
                file=sys.stderr,
            )
        return DatasetDict(self.dev_splits)

    def write(self, value: DatasetDict, path: str) -> None:
        """Write the splits, clearing any pre-port cache sitting in the way.

        `exists()` reports a record-less directory as absent, so one can still
        be on disk here. Removing it rather than saving over it keeps a stale
        split directory from surviving beside the new ones -- `save_to_disk`
        would rewrite `dataset_dict.json`, which then would not list it, but
        the rows would stay on disk unreferenced.
        """
        if os.path.isdir(path) and os.listdir(path):
            print(
                f"Replacing pre-artifact dev cache at {path} (no config record)",
                file=sys.stderr,
            )
            shutil.rmtree(path)
        value.save_to_disk(path)


class TokenizedMultinomialMix:
    """The tokenized, upsampled view of a multinomial mix for one tokenizer.

    Deliberately *not* a `CachedArtifact`. Everything this mix persists lives
    in the three artifacts it composes -- `TokenizedSourceArtifact` per
    source, `TrainPlanArtifact`, `DevSplitsArtifact` -- and the upsampled
    train split itself is never written to disk, which is the whole reason
    the index-based path exists. An artifact wrapped around that would have
    nothing left to cache: `write` a no-op, `read` identical to `build`, and
    a config record whose every field is already encoded in the path, so
    `validate` could never fail. That is the shape of the retired
    `UntokenizedDataset` wrapper, which architecture.md describes as stacking
    a second caching system on one it does not control. This class is the
    orchestration instead, and the caching lives where the data does.

    It still parallels `MultinomialDataset` (the untokenized-layer composite)
    in resolving its children dynamically, off the `sources` list rather than
    through a static `depends_on`. It does not route through
    `MultinomialDataset` itself, whose `write` materializes the whole
    upsampled corpus.

    Note that `mix_dir` and `resolve()` both need `add_labels`, which cannot
    be known without resolving every source's untokenized cache to look for
    instruction columns -- inherent to the problem, not a cost this design
    adds. The result is memoized per instance.
    """

    def __init__(
        self,
        base_cache_dir: str,
        sources: list[dict],
        alpha: float | None,
        total_samples: int,
        dev_size: float,
        tokenizer: PreTrainedTokenizer,
        tokenizer_id: str,
        max_length: int,
        shuffle_seed: int = 1,
    ):
        """Initialize the mix.

        Args:
            base_cache_dir: Parent cache directory holding both the mix
                subdirectory and the shared per-source caches.
            sources: Configuration entries for the sources to sample from,
                same schema as `MultinomialDataset`.
            alpha: Temperature for reweighting unpinned sources.
            total_samples: Target size of the training split, dev excluded.
            dev_size: Default fraction of each source held out, or -1 to skip.
            tokenizer: Tokenizer used to materialize the per-source tokenized
                caches.
            tokenizer_id: Stable identifier for the tokenizer (drives cache
                directory naming).
            max_length: Truncation length.
            shuffle_seed: Seed for shuffling the global training plan.

        Raises:
            ValueError: On an empty source list, a non-positive `total_samples`
                or `alpha`, or a missing `dev_size`.
        """
        if not sources:
            raise ValueError("Cannot sample from datasets: sources list is empty")
        if total_samples <= 0:
            raise ValueError(f"total_samples must be positive, got {total_samples}")
        if alpha is not None and alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if dev_size is None:
            raise ValueError("dev_size must be provided")

        self.base_cache_dir = base_cache_dir
        self.sources = sources
        self.alpha = alpha
        self.total_samples = total_samples
        self.dev_size = dev_size
        self.tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id
        self.max_length = max_length
        self.shuffle_seed = shuffle_seed
        self._sources_info_cache: tuple[list[str], list[str], bool] | None = None

    def _mix_config(self) -> dict:
        return {
            'alpha': self.alpha,
            'total_samples': self.total_samples,
            'dev_size': self.dev_size,
            'sources': [
                OmegaConf.to_container(DictConfig(s), resolve=True) for s in self.sources
            ],
        }

    @property
    def mix_dir(self) -> str:
        """Directory holding this mix and everything keyed on it.

        Tokenized dev splits, the training plan, and the per-source tokenized
        caches' parent all live under here.
        """
        return os.path.join(self.base_cache_dir, multinomial_mix_slug(self._mix_config()))

    def _sources_info(self) -> tuple[list[str], list[str], bool]:
        """Resolve every source's untokenized cache; memoized on first call.

        Returns:
            (source_ids, untokenized_paths, add_labels) -- add_labels is True
            if any source has instruction (prompt/response) columns, which
            forces every source in the mix to tokenize with labels.
        """
        if self._sources_info_cache is not None:
            return self._sources_info_cache

        source_ids = []
        untokenized_paths = []
        for idx, source_config in enumerate(self.sources):
            source_dict = DictConfig(source_config)
            child_id = source_id(source_dict, fallback=f"source_{idx}")
            source_cache = os.path.join(self.base_cache_dir, child_id)
            path = load_untokenized_dataset(dataset_config=source_dict, cache_dir=source_cache)
            source_ids.append(child_id)
            untokenized_paths.append(path)

        add_labels = any(_source_has_instruction_columns(p) for p in untokenized_paths)
        if add_labels:
            print(
                "Mix contains instruction data; tokenizing all sources with labels.",
                file=sys.stderr,
            )

        self._sources_info_cache = (source_ids, untokenized_paths, add_labels)
        return self._sources_info_cache

    def resolve(self) -> DatasetDict:
        """Resolve the three sub-artifacts and assemble the mix.

        Returns:
            A `DatasetDict` with an indices-mapped `train` view over the
            concatenated per-source tokenized caches, plus one entry per
            source that has a dev split.
        """
        print(
            f"Multinomial sampling from {len(self.sources)} sources with alpha={self.alpha}",
            file=sys.stderr,
        )
        print(f"Mix cache directory: {self.mix_dir}", file=sys.stderr)

        source_ids, untokenized_paths, add_labels = self._sources_info()
        source_dev_sizes = [
            getattr(DictConfig(s), 'dev_size', self.dev_size) for s in self.sources
        ]

        source_artifacts = [
            TokenizedSourceArtifact(
                untokenized_path=untokenized_path,
                tokenizer=self.tokenizer,
                tokenizer_id=self.tokenizer_id,
                max_length=self.max_length,
                add_labels=add_labels,
            )
            for untokenized_path in untokenized_paths
        ]
        # Re-read from disk rather than keep resolve()'s return value: on a
        # cold build, resolve() returns the in-memory build() result without
        # round-tripping through write()+read(), and that object's datasets
        # fingerprint differs from what load_from_disk() gives the same
        # on-disk data. concatenate_datasets/.select() below are
        # fingerprint-sensitive (their own content-addressed caching, not
        # just data-sensitive), so a freshly-built source and a
        # cache-warm one must present identical fingerprints or the dev-split
        # selection silently loses its cache hit on the next run.
        for artifact in source_artifacts:
            artifact.resolve()
        per_source_tokenized = [load_from_disk(artifact.path) for artifact in source_artifacts]
        source_sizes = [len(d) for d in per_source_tokenized]

        train_pools = []
        dev_indices_per_source = []
        for size, src_dev in zip(source_sizes, source_dev_sizes):
            train_idx, dev_idx = _partition_source_indices(size, src_dev, seed=1)
            train_pools.append(train_idx)
            dev_indices_per_source.append(dev_idx)

        train_pool_sizes = [len(p) for p in train_pools]
        if all(s == 0 for s in train_pool_sizes):
            raise ValueError("Cannot sample: every source has an empty train pool")
        sampling_probs = compute_sampling_probs(self.sources, train_pool_sizes, self.alpha)
        samples_per_source = [int(p * self.total_samples) for p in sampling_probs]
        remaining = self.total_samples - sum(samples_per_source)
        for i in range(remaining):
            samples_per_source[i % len(self.sources)] += 1

        print("Train sampling distribution (plan-based):", file=sys.stderr)
        for idx, count in enumerate(samples_per_source):
            pct = 100 * count / self.total_samples
            pinned = self.sources[idx].get('sampling_prob') is not None
            marker = " (pinned)" if pinned else ""
            print(
                f"  {source_ids[idx]}: {count} samples ({pct:.2f}%){marker}",
                file=sys.stderr,
            )

        plan_artifact = TrainPlanArtifact(
            mix_dir=self.mix_dir,
            source_ids=source_ids,
            source_sizes=source_sizes,
            samples_per_source=samples_per_source,
            shuffle_seed=self.shuffle_seed,
            train_pools=train_pools,
        )
        global_indices = plan_artifact.resolve()

        dev_splits = {}
        for src_id, ds, dev_idx in zip(source_ids, per_source_tokenized, dev_indices_per_source):
            if len(dev_idx) == 0:
                continue
            dev_splits[src_id] = ds.select(dev_idx.tolist()).flatten_indices()

        dev_artifact = DevSplitsArtifact(
            mix_dir=self.mix_dir,
            tokenizer_id=self.tokenizer_id,
            max_length=self.max_length,
            add_labels=add_labels,
            source_ids=source_ids,
            dev_splits=dev_splits,
        )
        dev_dict = dev_artifact.resolve()

        # Assemble the virtual training view. concatenate + select store an
        # Arrow indices map; no row duplication occurs on disk or in memory.
        concat = concatenate_datasets(per_source_tokenized)
        train = concat.select(global_indices.tolist())

        result = {'train': train}
        for key, value in dev_dict.items():
            result[key] = value
        return DatasetDict(result)


class TokenizedDatasetArtifact(DatasetArtifact):
    """The whole-corpus tokenized dataset for non-multinomial dataset types.

    Tokenizes the untokenized `DatasetDict` in one shot and splits train/dev,
    unless the untokenized data already carries per-source dev splits (the
    legacy materialized multinomial path -- reachable by calling
    `load_untokenized_dataset` directly on a `type: multinomial` config, but
    not the production path for it; see `TokenizedMultinomialMix`).
    """

    def __init__(
        self,
        cache_dir: str,
        tokenized_dataset_config: TokenizedDatasetConfig,
        untokenized_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        dev_size: float,
    ):
        """Initialize the artifact.

        Args:
            cache_dir: The dataset's cache directory (parent of `path`).
            tokenized_dataset_config: Tracked parameters -- also the source of
                `path`, via `cache_dir(cache_dir)`.
            untokenized_path: Path to the untokenized dataset.
            tokenizer: Tokenizer to use for tokenization.
            max_length: Maximum sequence length for tokenization.
            dev_size: Fraction (0 < dev_size < 1) or absolute count
                (dev_size >= 1) of data to hold out (ignored if the dataset is
                already split).
        """
        super().__init__(root=cache_dir)
        self.tokenized_dataset_config = tokenized_dataset_config
        self.untokenized_path = untokenized_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dev_size = dev_size

    @property
    def path(self) -> str:
        return self.tokenized_dataset_config.cache_dir(self.root)

    def config(self) -> dict:
        return self.tokenized_dataset_config.to_dict()

    def artifact_config(self) -> ArtifactConfig:
        return self.tokenized_dataset_config

    def build(self, deps) -> DatasetDict:
        print(
            f"Tokenizing dataset with {self.tokenizer.name_or_path} "
            f"(vocab size {len(self.tokenizer)})",
            file=sys.stderr,
        )
        dataset = load_from_disk(self.untokenized_path)

        # Check if this is an instruction dataset (has 'prompt'/'response' instead of 'text')
        # Always check 'train' split since multinomial datasets may have per-source dev splits
        # with different column schemas (e.g., 'eng' dev has only 'text', 'train' has mixed)
        sample_split = 'train' if 'train' in dataset else list(dataset.keys())[0]
        has_instruction_data = (
            'prompt' in dataset[sample_split].column_names
            and 'response' in dataset[sample_split].column_names
        )

        if has_instruction_data:
            print(
                "Detected instruction dataset format, tokenizing with label masking",
                file=sys.stderr,
            )

        # Process each split individually since they may have different column schemas
        # (e.g., multinomial datasets with per-source dev splits)
        tokenized_splits = {}
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            split_columns = split_data.column_names

            split_has_instruction = 'prompt' in split_columns and 'response' in split_columns
            split_has_text = 'text' in split_columns

            if split_has_instruction:
                columns_to_remove = ['prompt', 'response']
                if split_has_text:
                    columns_to_remove.append('text')
                tokenized_splits[split_name] = split_data.map(
                    lambda examples: _tokenize_instruction_examples(
                        examples, self.tokenizer, self.max_length
                    ),
                    batched=True,
                    remove_columns=columns_to_remove,
                )
            elif split_has_text:
                if has_instruction_data:
                    tokenized_splits[split_name] = split_data.map(
                        lambda examples: _tokenize_plaintext_with_labels(
                            examples, self.tokenizer, self.max_length
                        ),
                        batched=True,
                        remove_columns='text',
                    )
                else:
                    tokenized_splits[split_name] = split_data.map(
                        lambda examples: self.tokenizer(
                            examples['text'], max_length=self.max_length, truncation=True
                        ),
                        batched=True,
                        remove_columns='text',
                    )
            else:
                raise ValueError(
                    f"Split '{split_name}' has neither instruction columns (prompt/response) "
                    f"nor text column. Found columns: {split_columns}"
                )

        dataset = DatasetDict(tokenized_splits)

        # Check if dataset already has dev splits (from multinomial sampling)
        has_dev_splits = any(key != 'train' and key != 'test' for key in dataset.keys())

        if not has_dev_splits:
            if self.dev_size <= 0:
                raise ValueError(f"dev_size must be positive, got {self.dev_size}")
            test_size = int(self.dev_size) if self.dev_size >= 1 else self.dev_size
            dataset = dataset['train'].train_test_split(test_size=test_size)
        else:
            print("Dataset already has per-source dev splits", file=sys.stderr)

        return dataset


def _partition_source_indices(
    num_rows: int,
    dev_size: float,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the train/dev row partition for a single source.

    Uses a numpy RNG seeded deterministically so the partition is reproducible
    and independent of HuggingFace's internal split implementation.

    Args:
        num_rows: Size of the source's tokenized dataset.
        dev_size: -1 to skip dev split, fractional (0 < x < 1) for proportion,
            or absolute (>= 1) for an explicit row count.
        seed: RNG seed for the permutation.

    Returns:
        (train_indices, dev_indices) as uint32 numpy arrays. dev_indices is empty
        when dev_size == -1.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_rows).astype(np.uint32)
    if dev_size == -1:
        return perm, np.empty(0, dtype=np.uint32)
    if dev_size >= 1:
        dev_count = int(dev_size)
    elif 0 < dev_size < 1:
        dev_count = int(round(dev_size * num_rows))
    else:
        raise ValueError(f"Invalid dev_size {dev_size}")
    dev_count = min(dev_count, num_rows)
    dev_indices = perm[:dev_count]
    train_indices = perm[dev_count:]
    return train_indices, dev_indices


def load_external_eval_set(
    eval_config: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    add_labels: bool = False,
) -> Dataset:
    """
    Load and tokenize an external evaluation dataset.

    Args:
        eval_config: Dictionary with 'name', 'path', and optional 'format' keys
            - name: Name for the eval set (used in metrics)
            - path: Path to the data file
            - format: 'plaintext' (default) or 'jsonl'
            - text_column: Column name for jsonl format (default: 'text')
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length for tokenization
        add_labels: If True, add a 'labels' column (=input_ids) for plaintext/jsonl
            formats so the set is compatible with DataCollatorForInstructionTuning.
            instruction_jsonl format always includes masked labels regardless.

    Returns:
        Tokenized Dataset ready for evaluation
    """
    name = eval_config['name']
    path = eval_config['path']
    file_format = eval_config.get('format', 'plaintext')
    text_column = eval_config.get('text_column', 'text')

    print(f"Loading external eval set '{name}' from {path}", file=sys.stderr)

    if not os.path.exists(path):
        raise ValueError(f"External eval set file not found: {path}")

    # Load data based on format
    if file_format == 'plaintext':
        # Read lines from plaintext file
        with open(path, encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        dataset = Dataset.from_dict({'text': lines})
        is_instruction = False

    elif file_format == 'jsonl':
        # Load JSONL file
        data = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if text_column in obj:
                        data.append(obj[text_column])
                    else:
                        raise ValueError(
                            f"JSONL file missing '{text_column}' column: {path}"
                        )
        dataset = Dataset.from_dict({'text': data})
        is_instruction = False

    elif file_format == 'instruction_jsonl':
        # Load instruction JSONL with prompt/response fields
        prompts, responses = read_instruction_jsonl(path)
        dataset = Dataset.from_dict({'prompt': prompts, 'response': responses})
        is_instruction = True

    else:
        raise ValueError(
            f"Unsupported format '{file_format}' for external eval set. "
            f"Supported formats: 'plaintext', 'jsonl', 'instruction_jsonl'"
        )

    print(f"  Loaded {len(dataset)} examples", file=sys.stderr)

    # Tokenize the dataset
    if is_instruction:
        # Instruction format: use label masking (loss only on response)
        dataset = dataset.map(
            lambda examples: _tokenize_instruction_examples(examples, tokenizer, max_length),
            batched=True,
            remove_columns=['prompt', 'response'],
            desc=f"Tokenizing external eval set '{name}'"
        )
    else:
        # Plain text format: standard tokenization
        if add_labels:
            tokenize_fn = lambda examples: _tokenize_plaintext_with_labels(
                examples, tokenizer, max_length
            )
        else:
            tokenize_fn = lambda examples: tokenizer(
                examples['text'], max_length=max_length, truncation=True
            )
        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=['text'],
            desc=f"Tokenizing external eval set '{name}'"
        )

    return dataset


def prepare_eval_datasets(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    external_eval_sets: list = None
):
    """
    Prepare evaluation datasets from loaded data and optional external sources.

    Handles both:
    - Extracting dev splits from tokenized dataset (single or per-language)
    - Loading and merging external evaluation sets

    Args:
        dataset: Tokenized DatasetDict with train and dev/test splits
        tokenizer: Tokenizer for tokenizing external eval sets
        max_length: Max sequence length for tokenization
        external_eval_sets: Optional list of external eval configs, each with
            'name', 'path', and optional 'format' keys

    Returns:
        Either a single Dataset (standard case) or dict of Datasets (multinomial
        or when external eval sets are added)
    """
    # Dev splits are any non-train splits except 'test'
    dev_splits = [key for key in dataset.keys() if key != 'train' and key != 'test']

    if dev_splits:
        # Multinomial sampling case: multiple per-language dev sets
        eval_dataset = {key: dataset[key] for key in dev_splits}
        print(
            f"Using {len(eval_dataset)} per-language eval sets: {', '.join(dev_splits)}",
            file=sys.stderr
        )
    else:
        # Standard case: single dev/test split
        eval_dataset = dataset['test']

    # Load external evaluation sets if configured
    if external_eval_sets:
        # If eval_dataset is not already a dict, convert it
        if not isinstance(eval_dataset, dict):
            eval_dataset = {'dev': eval_dataset}
            print("Converted single eval dataset to dict for external eval sets", file=sys.stderr)

        # If the existing eval sets carry labels (instruction-tuning run), plaintext
        # external eval sets must also have labels to be compatible with
        # DataCollatorForInstructionTuning.
        existing_has_labels = any(
            'labels' in ds.column_names for ds in eval_dataset.values()
        )

        # Load and add each external eval set
        for eval_config in external_eval_sets:
            name = eval_config['name']

            # Check for name conflicts
            if name in eval_dataset:
                raise ValueError(
                    f"External eval set name '{name}' conflicts with existing eval set. "
                    f"Existing eval sets: {list(eval_dataset.keys())}"
                )

            external_dataset = load_external_eval_set(
                eval_config=eval_config,
                tokenizer=tokenizer,
                max_length=max_length,
                add_labels=existing_has_labels,
            )
            eval_dataset[name] = external_dataset
            print(
                f"Added external eval set '{name}' with {len(external_dataset)} examples",
                file=sys.stderr
            )

    return eval_dataset


class DataCollatorForInstructionTuning:
    """
    Data collator for instruction tuning with pre-computed labels.

    This collator expects examples that already have 'labels' field with
    prompt tokens masked as -100. It handles padding for:
    - input_ids: padded with tokenizer.pad_token_id
    - attention_mask: padded with 0
    - labels: padded with -100 (ignored by CrossEntropyLoss)

    Unlike DataCollatorForLanguageModeling, this collator does NOT create labels
    from input_ids - it uses the pre-computed labels from the dataset.

    Args:
        tokenizer: Tokenizer used for padding
        padding: Padding strategy ('longest', 'max_length', or False)
        max_length: Maximum length when padding='max_length'
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding: str = 'longest',
        max_length: int = None
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token_id is None:
            # Use EOS token as pad token if not set
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features: list) -> dict:
        """
        Collate a batch of features.

        Args:
            features: List of dicts with 'input_ids', 'attention_mask', and 'labels'

        Returns:
            Batch dict with padded tensors
        """
        import torch

        # Separate labels from other features for custom padding
        labels = [f['labels'] for f in features]
        # Remove labels temporarily for tokenizer padding
        features_without_labels = [{k: v for k, v in f.items() if k != 'labels'} for f in features]

        # Use tokenizer's padding for input_ids and attention_mask
        batch = self.tokenizer.pad(
            features_without_labels,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Pad labels with -100 (ignored by loss function)
        max_label_length = max(len(l) for l in labels)
        padded_labels = []
        for label in labels:
            padding_length = max_label_length - len(label)
            # Pad on the right with -100
            padded_label = label + [-100] * padding_length
            padded_labels.append(padded_label)

        batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


def is_instruction_dataset(dataset) -> bool:
    """
    Check if a dataset is an instruction-tuning dataset (has pre-computed labels).

    Args:
        dataset: A Dataset or DatasetDict

    Returns:
        True if the dataset has 'labels' column, indicating instruction format
    """
    if hasattr(dataset, 'keys'):
        # DatasetDict - check the first split
        sample_split = list(dataset.keys())[0]
        return 'labels' in dataset[sample_split].column_names
    else:
        # Single Dataset
        return 'labels' in dataset.column_names
