"""LAPT's concrete artifact configurations.

Each artifact (tokenizer, dataset, model) has a config class that:

1. Defines all parameters that affect the artifact
2. Provides from_args() to extract from the full Hydra config
3. Provides cache_path() to generate consistent cache paths
4. Provides save() / check_cached() for config tracking

The tracking machinery itself -- the `ArtifactConfig` base class, config
diffing, and path digests -- lives in `lapt_core.artifacts`, which is shared
with sibling projects. This module holds only what is specific to LAPT's Hydra
schema and cache layout.
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from lapt_core.artifacts import (
    ArtifactConfig,
    ConfigMismatchError,
    config_digest,
    dict_diff,
    format_number,
)


def resolve_dev_size(args: DictConfig):
    """
    Resolve dev_size from the Hydra config, with fallback for backwards compatibility.

    Preferred location is args.dataset.dev_size. Falls back to args.training.dev_size
    with a deprecation warning. Raises ValueError if found in neither.

    Args:
        args: Full Hydra configuration

    Returns:
        The resolved dev_size value
    """
    dataset_dev_size = getattr(args.dataset, 'dev_size', None)
    if dataset_dev_size is not None:
        return dataset_dev_size

    training_dev_size = getattr(args.training, 'dev_size', None)
    if training_dev_size is not None:
        warnings.warn(
            "dev_size is set under 'training' config but should be under 'dataset'. "
            "Please move it to your dataset config. "
            "Falling back to training.dev_size for now.",
            FutureWarning,
            stacklevel=2,
        )
        return training_dev_size

    # TODO: currently dev_size is required, but it would be reasonable to allow
    # skipping it when using only external eval sets. Would need dev_size=None
    # support in load_tokenized_dataset.
    raise ValueError(
        "dev_size not found in dataset or training config. "
        "Please set dataset.dev_size in your config."
    )


def get_model_shortname(hf_model: str) -> str:
    """
    Extract a short identifier from HuggingFace model name.

    Args:
        hf_model: Full HuggingFace model name (e.g., "facebook/xglm-564M")

    Returns:
        Short identifier (e.g., "xglm564m")
    """
    model_name = hf_model.split('/')[-1]
    return model_name.lower().replace('-', '').replace('.', '')


DEFAULT_SEED = 1


def multinomial_mix_slug(dataset_config: dict) -> str:
    """
    Build a deterministic subdirectory name for a multinomial dataset mix.

    The upsampled training split produced by multinomial sampling depends on
    alpha, total_samples, dev_size, per-source sampling_prob /
    upsampling_factor / dev_size overrides, and the seed, but NOT on the
    underlying source datasets (which live in parent-level subdirectories and
    can be shared across mixes). Caching mix-dependent artifacts inside {cache_dir}/{slug}/
    instead of directly under {cache_dir}/ means sweeping alpha or sample
    counts no longer clobbers the previous mix and source caches are
    transparently shared.

    Args:
        dataset_config: Dict with at least 'total_samples' and 'sources'. Must
            correspond to a multinomial dataset. 'alpha' is optional, since it is
            omissible for mixes where it cannot affect the sampling probabilities.
            'seed' is optional and defaults to DEFAULT_SEED.

    Returns:
        Slug like "mix_a0.5_s5m_ab12cd34", or "mix_s5m_ab12cd34" without alpha.
    """
    alpha = dataset_config.get('alpha')
    total_samples = dataset_config['total_samples']

    mix_keys = {
        'alpha': alpha,
        'total_samples': total_samples,
        'dev_size': dataset_config.get('dev_size'),
        'sources': [
            {
                'id': source.get('id') or source.get('language'),
                'sampling_prob': source.get('sampling_prob'),
                'upsampling_factor': source.get('upsampling_factor'),
                'dev_size': source.get('dev_size'),
                'substitutions': source.get('substitutions'),
            }
            for source in dataset_config.get('sources', [])
        ],
    }
    # a non-default seed changes which examples are sampled and repeated, so
    # mixes that differ only by seed must not share a directory. the key is
    # omitted at the default so that slugs predating seed-keying are unchanged
    # -- every mix built before this was built at DEFAULT_SEED, so the omission
    # records a fact rather than papering over one. the seed is recorded in the
    # config unconditionally either way, so validation is unaffected.
    seed = dataset_config.get('seed', DEFAULT_SEED)
    if seed != DEFAULT_SEED:
        mix_keys['seed'] = seed

    digest = config_digest(mix_keys)

    # omit the alpha segment when the config has no alpha, rather than writing
    # "aNone" into the directory name; slugs for configs that do set it are unchanged
    alpha_part = f"a{alpha}_" if alpha is not None else ""
    return f"mix_{alpha_part}s{format_number(total_samples)}_{digest}"


@dataclass
class TokenizerConfig(ArtifactConfig):
    """
    Configuration for FOCUS tokenizer training.

    Contains all parameters that affect the tokenizer artifact. Used for:
    - Generating cache paths
    - Validating cached tokenizers match current config
    - Passing to train_new_tokenizer()
    """
    artifact_name = "Tokenizer"

    # Core tokenizer parameters
    hf_model: str
    vocab_size: int
    num_samples: int
    character_coverage: float
    inherit_additional_special_tokens: bool

    # Seed vocabulary parameters
    use_seed_vocabulary: bool
    seed_vocab_multiplier: float
    seed_lambda: float
    seed_min_frequency: int
    seed_round_mode: str
    seed_score_mode: str

    # Embedding initialization
    fasttext_model_min_count: int

    # Reproducibility
    seed: int

    # Fresh-tokenizer algorithm: None inherits the base tokenizer's algorithm;
    # 'unigram' or 'bpe' overrides it when training a fresh tokenizer.
    tokenizer_algorithm: str | None = None

    # Model identifier (optional override for local model paths)
    init_model_id: str | None = None

    # Pre-built tokenizer path (bypasses training, e.g. PTEx tokenizer)
    tokenizer_path: str | None = None

    # Data source (one of these will be set)
    train_dataset_cache: str | None = None
    focus_dataset: dict | None = field(default=None)

    @classmethod
    def from_args(cls, args: DictConfig) -> Optional['TokenizerConfig']:
        """
        Extract TokenizerConfig from full Hydra config.

        Args:
            args: Full Hydra configuration

        Returns:
            TokenizerConfig if FOCUS is enabled, None otherwise
        """
        if not args.focus.enabled:
            return None

        # Determine data source
        train_dataset_cache = None
        focus_dataset = None
        if hasattr(args.focus, 'dataset') and args.focus.dataset is not None:
            focus_dataset = OmegaConf.to_container(args.focus.dataset, resolve=True)
        else:
            train_dataset_cache = args.dataset.cache_dir

        init_model_id = getattr(args, 'init_model_id', None) or None
        tokenizer_path = getattr(args.focus, 'tokenizer_path', None) or None

        tokenizer_algorithm = args.focus.get('tokenizer_algorithm', None) or None
        if tokenizer_algorithm is not None:
            tokenizer_algorithm = tokenizer_algorithm.lower()
            if tokenizer_algorithm not in ('unigram', 'bpe'):
                raise ValueError(
                    f"focus.tokenizer_algorithm must be 'unigram' or 'bpe', "
                    f"got {tokenizer_algorithm!r}"
                )

        return cls(
            hf_model=args.hf_model,
            vocab_size=args.focus.vocab_size,
            num_samples=args.focus.num_samples,
            character_coverage=args.focus.get('character_coverage', 1.0),
            inherit_additional_special_tokens=args.focus.get(
                'inherit_additional_special_tokens', True
            ),
            tokenizer_algorithm=tokenizer_algorithm,
            use_seed_vocabulary=args.focus.get('use_seed_vocabulary', False),
            seed_vocab_multiplier=args.focus.get('seed_vocab_multiplier', 5.0),
            seed_lambda=args.focus.get('seed_lambda', 0.5),
            seed_min_frequency=args.focus.get('seed_min_frequency', 1),
            seed_round_mode=args.focus.get('seed_round_mode', 'round'),
            seed_score_mode=args.focus.get('seed_score_mode', 'count'),
            fasttext_model_min_count=args.focus.get('fasttext_model_min_count', 4),
            seed=args.seed,
            init_model_id=init_model_id,
            tokenizer_path=tokenizer_path,
            train_dataset_cache=train_dataset_cache,
            focus_dataset=focus_dataset,
        )

    def _build_suffix(self) -> str:
        """
        Build suffix string encoding key parameters for cache paths.

        Not all parameters are encoded - just those that differentiate cache directories.
        Full config validation catches mismatches for parameters not in the path.
        """
        vocab_str = format_number(self.vocab_size)
        samples_str = format_number(self.num_samples)
        suffix = f"focus-v{vocab_str}-s{samples_str}"

        # Only encode the algorithm when explicitly set, so existing inherited
        # (None) caches keep their paths.
        if self.tokenizer_algorithm is not None:
            suffix += f"_{self.tokenizer_algorithm}"

        if not self.inherit_additional_special_tokens:
            suffix += "_no-additional"

        if self.focus_dataset is not None:
            suffix += "_customdata"

        if self.use_seed_vocabulary:
            suffix += "_seeded"
            suffix += f"-{self.seed_vocab_multiplier}x"
            suffix += f"-lambda{self.seed_lambda}"
            if self.seed_min_frequency > 1:
                suffix += f"-min{self.seed_min_frequency}"
            if self.seed_score_mode != 'count':
                suffix += f"-{self.seed_score_mode}"

        return suffix

    def _model_shortname(self) -> str:
        """Extract short model name, preferring init_model_id for local paths."""
        if self.init_model_id:
            return self.init_model_id
        return get_model_shortname(self.hf_model)

    def cache_dir(self, language: str) -> str:
        """
        Generate tokenizer cache directory path.

        Args:
            language: Language code for the tokenizer

        Returns:
            Path like "tokenizers/got/xglm564m_focus-v16k-s100k_seeded-5.0x-lambda0.5"
        """
        return f"tokenizers/{language}/{self.tokenizer_id()}"

    def seed_tokenizer_suffix(self) -> str:
        """
        Generate directory name for the intermediate seed tokenizer.

        The seed tokenizer is shared across lambda values since it only depends
        on vocab_size, num_samples, and multiplier.

        Returns:
            String like "xglm564m_focus-v16k-s200k_seed-5.0x"
        """
        model_short = self._model_shortname()
        vocab_str = format_number(self.vocab_size)
        samples_str = format_number(self.num_samples)
        return f"{model_short}_focus-v{vocab_str}-s{samples_str}_seed-{self.seed_vocab_multiplier}x"

    def tokenizer_id(self) -> str:
        """
        Generate unique identifier for this tokenizer configuration.

        Used as a path component in tokenizer cache, tokenized dataset,
        model output, and FOCUS training data directories.

        When tokenizer_path is set (pre-built tokenizer, e.g. PTEx), the ID
        is derived from the directory name and num_samples rather than from
        tokenizer training parameters that don't apply.

        Returns:
            String like "xglm564m_focus-v16k-s100k_seeded-5.0x-lambda0.5"
            or "xglm564m_ptex_test-s5m" for pre-built tokenizers
        """
        model_short = self._model_shortname()
        if self.tokenizer_path:
            tokenizer_name = os.path.basename(os.path.normpath(self.tokenizer_path))
            samples_str = format_number(self.num_samples)
            # Avoid doubling the base-model prefix (e.g. xglm564m_xglm564m_...)
            # when the supplied tokenizer name already encodes the same base model.
            prefix = f"{model_short}_"
            if tokenizer_name.startswith(prefix):
                return f"{tokenizer_name}-s{samples_str}"
            return f"{model_short}_{tokenizer_name}-s{samples_str}"
        suffix = self._build_suffix()
        return f"{model_short}_{suffix}"

    # Fields that affect FOCUS embeddings but NOT the tokenizer artifact
    # itself (or, in the pre-built-tokenizer case, neither). Stripped from
    # both sides during check_cached so that a mix / FOCUS-knob change does
    # not invalidate an otherwise-reusable tokenizer cache. Embedding
    # provenance lives in the per-mix sidecar meta under focus_embs/.
    _embedding_only_fields = (
        'train_dataset_cache',
        'focus_dataset',
        'fasttext_model_min_count',
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for saving to YAML.

        When tokenizer_path is set (pre-built tokenizer), only fields that
        actually affect the downstream artifacts are included. Tokenizer
        training params (vocab_size, character_coverage, seed vocab settings,
        etc.) are excluded since they were bypassed.
        """
        if self.tokenizer_path:
            return {
                'hf_model': self.hf_model,
                'init_model_id': self.init_model_id,
                'tokenizer_path': self.tokenizer_path,
                'num_samples': self.num_samples,
                'seed': self.seed,
            }
        from dataclasses import asdict
        d = asdict(self)
        for k in self._embedding_only_fields:
            d.pop(k, None)
        return d

    def check_cached(self, config_path: str, error_on_mismatch: bool = True) -> bool:
        """Validate cached tokenizer config, tolerating legacy embedding-only fields.

        Older tokenizer caches recorded fields like `train_dataset_cache`,
        `focus_dataset`, and `fasttext_model_min_count` that we now consider
        embedding-level (not tokenizer-level) provenance. Strip them from the
        cached config before diffing so previously valid caches keep working
        and a mix change no longer forces a full tokenizer retrain.
        """
        if not os.path.exists(config_path):
            return True
        with open(config_path) as f:
            cached = yaml.safe_load(f) or {}
        for k in self._embedding_only_fields:
            cached.pop(k, None)
        # Backward compat: caches predating tokenizer_algorithm were all trained
        # with the inherited algorithm (None), so treat a missing key as such.
        cached.setdefault('tokenizer_algorithm', None)
        # the filtering above is why this cannot just delegate to the base
        # implementation; the message itself is still shared
        diffs = dict_diff(cached, self.to_dict())
        if not diffs:
            return True
        error_msg = self._format_mismatch(config_path, diffs)
        if error_on_mismatch:
            raise ConfigMismatchError(error_msg)
        print(error_msg, file=sys.stderr)
        return False


class DatasetConfig(ArtifactConfig):
    """
    Configuration for untokenized dataset caching.

    Unlike TokenizerConfig, this is not a dataclass because the tracked
    parameters vary by dataset type. Internally stores a config dict
    built by the type-dispatch logic in from_args().
    """
    artifact_name = "Untokenized Dataset"

    def __init__(self, config: dict):
        self._config = config

    @classmethod
    def from_args(cls, args: DictConfig) -> 'DatasetConfig':
        """
        Extract DatasetConfig from full Hydra config.

        Args:
            args: Full Hydra configuration

        Returns:
            DatasetConfig instance
        """
        config = {
            'type': args.dataset.type,
            'seed': args.seed,
        }

        dataset_type = args.dataset.type
        if dataset_type == 'oscar':
            config['language'] = args.dataset.language
        elif dataset_type == 'huggingface':
            config['name'] = args.dataset.name
            config['config'] = getattr(args.dataset, 'config', None)
            config['split'] = getattr(args.dataset, 'split', 'train')
            config['text_column'] = getattr(args.dataset, 'text_column', 'text')
            config['max_samples'] = getattr(args.dataset, 'max_samples', None)
            config['min_words_per_line'] = getattr(args.dataset, 'min_words_per_line', None)
            config['oversampling_factor'] = getattr(args.dataset, 'oversampling_factor', 3)
        elif dataset_type == 'plaintext':
            config['path'] = args.dataset.path
        elif dataset_type == 'plaintext_dir':
            config['directory'] = args.dataset.directory
            config['pattern'] = getattr(args.dataset, 'pattern', '*.txt')
        elif dataset_type == 'concat':
            config['sources'] = OmegaConf.to_container(args.dataset.sources, resolve=True)
        elif dataset_type == 'multinomial':
            config['sources'] = OmegaConf.to_container(args.dataset.sources, resolve=True)
            config['alpha'] = args.dataset.get('alpha')
            config['total_samples'] = args.dataset.total_samples
            config['dev_size'] = resolve_dev_size(args)

        return cls(config)

    def to_dict(self) -> dict:
        """Return config dictionary for saving to YAML."""
        return dict(self._config)

    def effective_cache_dir(self, base_cache_dir: str) -> str:
        """
        Resolve the cache directory for mix-dependent artifacts.

        For multinomial datasets, returns a subdirectory keyed on the mix
        parameters so untokenized/tokenized caches don't clobber each other
        across alpha/sample sweeps, while the per-source subdirs remain at
        the parent level and are transparently shared. For other dataset
        types, returns the base cache dir unchanged.

        Args:
            base_cache_dir: The cache_dir as specified in the dataset config.

        Returns:
            Effective cache directory (possibly a mix subfolder of base).
        """
        if self._config.get('type') == 'multinomial':
            return os.path.join(base_cache_dir, multinomial_mix_slug(self._config))
        return base_cache_dir


def effective_dataset_cache_dir(args: DictConfig) -> str:
    """
    Convenience wrapper: resolve the effective dataset cache dir from full args.

    Computed from `args.dataset` without loading the dataset, so it can be
    used in path-construction helpers that run before the dataset is built.
    For non-multinomial datasets this returns `args.dataset.cache_dir`
    unchanged without requiring that every dataset field be populated.
    """
    base_cache_dir = args.dataset.cache_dir
    dataset_type = getattr(args.dataset, 'type', None)
    if dataset_type != 'multinomial':
        return base_cache_dir
    return DatasetConfig.from_args(args).effective_cache_dir(base_cache_dir)


def focus_embedding_hash(args: DictConfig) -> str:
    """
    Build a deterministic 8-char hash identifying the inputs that determine
    FOCUS embedding values, independent of tokenizer hyperparameters.

    Hashes the FOCUS-training data spec (separate `focus.dataset` if set,
    otherwise the training-dataset spec), `focus.num_samples`, `seed`, and
    `focus.fasttext_model_min_count` — every input that changes what
    fastText sees or how it is sampled. The same tokenizer can therefore
    host multiple cached embedding sets keyed by this hash, one per mix.
    """
    if hasattr(args.focus, 'dataset') and args.focus.dataset is not None:
        data_spec = OmegaConf.to_container(args.focus.dataset, resolve=True)
    else:
        data_spec = DatasetConfig.from_args(args).to_dict()

    keys = {
        'data': data_spec,
        'num_samples': args.focus.num_samples,
        'seed': args.seed,
        'fasttext_model_min_count': args.focus.get('fasttext_model_min_count', 4),
    }
    return config_digest(keys)


class TokenizedDatasetConfig(ArtifactConfig):
    """
    Configuration for tokenized dataset caching.

    Composes DatasetConfig and TokenizerConfig (or base model identity when
    FOCUS is disabled) along with tokenization-specific parameters.
    """
    artifact_name = "Tokenized Dataset"

    def __init__(
        self,
        max_length: int,
        dev_size: float,
        dataset_config: DatasetConfig,
        tokenizer_id: str,
        tokenizer_config: TokenizerConfig | None = None,
        hf_model: str | None = None,
        init_model_id: str | None = None,
    ):
        self.max_length = max_length
        self.dev_size = dev_size
        self.dataset_config = dataset_config
        self._tokenizer_id = tokenizer_id
        self.tokenizer_config = tokenizer_config
        self.hf_model = hf_model
        self.init_model_id = init_model_id

    @classmethod
    def from_args(cls, args: DictConfig) -> 'TokenizedDatasetConfig':
        """
        Extract TokenizedDatasetConfig from full Hydra config.

        Args:
            args: Full Hydra configuration

        Returns:
            TokenizedDatasetConfig instance
        """
        dataset_config = DatasetConfig.from_args(args)
        tokenizer_config = TokenizerConfig.from_args(args)
        dev_size = resolve_dev_size(args)

        hf_model = None
        init_model_id = None
        if tokenizer_config is not None:
            tokenizer_id = tokenizer_config.tokenizer_id()
        else:
            hf_model = getattr(args, 'hf_model', None)
            init_model_id = getattr(args, 'init_model_id', None)
            if init_model_id:
                tokenizer_id = init_model_id
            elif hf_model:
                tokenizer_id = get_model_shortname(hf_model)
            else:
                raise ValueError("No model identifier available for tokenized path")

        return cls(
            max_length=args.training.max_length,
            dev_size=dev_size,
            dataset_config=dataset_config,
            tokenizer_id=tokenizer_id,
            tokenizer_config=tokenizer_config,
            hf_model=hf_model,
            init_model_id=init_model_id,
        )

    def cache_dir(self, dataset_cache_dir: str) -> str:
        """
        Generate tokenized dataset cache directory path.

        Args:
            dataset_cache_dir: Parent dataset cache directory

        Returns:
            Path like "{dataset_cache_dir}/tokenized_xglm564m_focus-v16k-s100k"
        """
        return f"{dataset_cache_dir}/tokenized_{self._tokenizer_id}"

    def to_dict(self) -> dict:
        """Convert to dictionary for saving to YAML."""
        config = {
            'max_length': self.max_length,
            'dev_size': self.dev_size,
            'dataset': self.dataset_config.to_dict(),
        }

        if self.tokenizer_config is not None:
            config['tokenizer'] = self.tokenizer_config.to_dict()
        else:
            if self.hf_model:
                config['hf_model'] = self.hf_model
            if self.init_model_id:
                config['init_model_id'] = self.init_model_id

        return config


class ModelConfig(ArtifactConfig):
    """
    Configuration for model training.

    Captures the full Hydra config for reproducibility. Unlike other artifact
    configs, this doesn't generate cache paths or do selective extraction —
    it saves the entire resolved config alongside model checkpoints.
    """
    artifact_name = "Model"

    def __init__(self, config: dict):
        self._config = config

    @classmethod
    def from_args(cls, args: DictConfig) -> 'ModelConfig':
        """
        Extract ModelConfig from full Hydra config.

        Args:
            args: Full Hydra configuration

        Returns:
            ModelConfig instance containing the full resolved config
        """
        config = OmegaConf.to_container(args, resolve=True)

        # When a pre-built tokenizer is plugged in, FOCUS bypasses SentencePiece
        # training entirely — vocab_size, character_coverage, and the various
        # seed-vocab / inherit-additional-special-tokens knobs are never
        # consulted. Null them out in the saved config so they don't appear to
        # document how the plugged-in tokenizer was trained (they don't).
        focus_cfg = config.get('focus') if isinstance(config, dict) else None
        if focus_cfg and focus_cfg.get('tokenizer_path'):
            unused_focus_keys = [
                'vocab_size',
                'character_coverage',
                'inherit_additional_special_tokens',
                'use_seed_vocabulary',
                'seed_vocab_multiplier',
                'seed_lambda',
                'seed_min_frequency',
                'seed_round_mode',
                'seed_score_mode',
            ]
            for key in unused_focus_keys:
                if key in focus_cfg:
                    focus_cfg[key] = None

        return cls(config)

    def to_dict(self) -> dict:
        """Return config dictionary for saving to YAML."""
        return dict(self._config)
