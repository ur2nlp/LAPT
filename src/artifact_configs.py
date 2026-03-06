"""
Configuration classes for LAPT artifacts.

Each artifact (tokenizer, dataset, model) has a config class that:
1. Defines all parameters that affect the artifact
2. Provides from_args() to extract from full Hydra config
3. Provides cache_path() to generate consistent cache paths
4. Provides save() / check_cached() for config tracking
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import yaml
from omegaconf import DictConfig, OmegaConf


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


class ArtifactConfig:
    """Base class for artifact configuration tracking.

    Subclasses must implement to_dict() and set artifact_name.
    """
    artifact_name: str

    def to_dict(self) -> dict:
        raise NotImplementedError

    def save(self, config_path: str):
        """Save this config to a YAML file.

        Args:
            config_path: Full path to the config file to write
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"Saved {self.artifact_name} config to {config_path}", file=sys.stderr)

    def check_cached(self, config_path: str, error_on_mismatch: bool = True) -> bool:
        """Check if cached config at path matches this config.

        Args:
            config_path: Full path to the cached config file
            error_on_mismatch: If True (default), raise ValueError on mismatch.
                               If False, print warning and return False.

        Returns:
            True if configs match (or no cached config exists),
            False if mismatch and error_on_mismatch is False

        Raises:
            ValueError: If configs don't match and error_on_mismatch is True
        """
        if not os.path.exists(config_path):
            return True

        with open(config_path, 'r') as f:
            cached_config = yaml.safe_load(f)

        if cached_config is None:
            return True

        diffs = _dict_diff(cached_config, self.to_dict())
        if not diffs:
            return True

        error_msg = (
            f"\n{'=' * 70}\n"
            f"CONFIG MISMATCH: {self.artifact_name}\n"
            f"{'=' * 70}\n"
            f"Cached artifact was created with different parameters:\n\n"
            + "\n".join(f"  {diff}" for diff in diffs)
            + f"\n\n"
            f"The cached config file represents what actually created this artifact.\n"
            f"To proceed, either:\n\n"
            f"  1. Regenerate with current config:\n"
            f"     Add fresh_dataset=true (dataset), fresh_tokenizer=true (tokenizer),\n"
            f"     or fresh_model=true (model) to your command\n\n"
            f"  2. Update your config to match the cached version\n"
            f"{'=' * 70}\n"
        )

        if error_on_mismatch:
            raise ValueError(error_msg)
        else:
            print(error_msg, file=sys.stderr)
            return False


def _dict_diff(dict1: dict, dict2: dict, path: str = "") -> list[str]:
    """
    Recursively find differences between two dictionaries.

    Args:
        dict1: First dictionary (cached)
        dict2: Second dictionary (current)
        path: Current path in nested structure (for error messages)

    Returns:
        List of difference descriptions
    """
    diffs = []

    only_in_1 = set(dict1.keys()) - set(dict2.keys())
    for key in only_in_1:
        diffs.append(
            f"{path}.{key}" if path
            else f"{key}: present in cached config but not in current"
        )

    only_in_2 = set(dict2.keys()) - set(dict1.keys())
    for key in only_in_2:
        diffs.append(
            f"{path}.{key}" if path
            else f"{key}: present in current config but not in cached"
        )

    for key in set(dict1.keys()) & set(dict2.keys()):
        val1 = dict1[key]
        val2 = dict2[key]
        current_path = f"{path}.{key}" if path else key

        if isinstance(val1, dict) and isinstance(val2, dict):
            diffs.extend(_dict_diff(val1, val2, current_path))
        elif val1 != val2:
            diffs.append(f"{current_path}: {val1} (cached) != {val2} (current)")

    return diffs


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


def format_number(n: int) -> str:
    """
    Format large numbers with k/m suffix for directory names.

    Uses integer division (truncation) for consistency in paths.

    Args:
        n: Number to format

    Returns:
        Formatted string like "50k", "1m" (always truncated, never decimal)
    """
    if n >= 1_000_000:
        return f"{n // 1_000_000}m"
    elif n >= 1000:
        return f"{n // 1000}k"
    else:
        return str(n)


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

    # Model identifier (optional override for local model paths)
    init_model_id: Optional[str] = None

    # Data source (one of these will be set)
    train_dataset_cache: Optional[str] = None
    focus_dataset: Optional[dict] = field(default=None)

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

        return cls(
            hf_model=args.hf_model,
            vocab_size=args.focus.vocab_size,
            num_samples=args.focus.num_samples,
            character_coverage=args.focus.get('character_coverage', 1.0),
            inherit_additional_special_tokens=args.focus.get(
                'inherit_additional_special_tokens', True
            ),
            use_seed_vocabulary=args.focus.get('use_seed_vocabulary', False),
            seed_vocab_multiplier=args.focus.get('seed_vocab_multiplier', 5.0),
            seed_lambda=args.focus.get('seed_lambda', 0.5),
            seed_min_frequency=args.focus.get('seed_min_frequency', 1),
            seed_round_mode=args.focus.get('seed_round_mode', 'round'),
            seed_score_mode=args.focus.get('seed_score_mode', 'count'),
            fasttext_model_min_count=args.focus.get('fasttext_model_min_count', 4),
            seed=args.seed,
            init_model_id=init_model_id,
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

        Returns:
            String like "xglm564m_focus-v16k-s100k_seeded-5.0x-lambda0.5"
        """
        model_short = self._model_shortname()
        suffix = self._build_suffix()
        return f"{model_short}_{suffix}"

    def to_dict(self) -> dict:
        """Convert to dictionary for saving to YAML."""
        from dataclasses import asdict
        return asdict(self)


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
            config['alpha'] = args.dataset.alpha
            config['total_samples'] = args.dataset.total_samples
            config['dev_size'] = resolve_dev_size(args)

        return cls(config)

    def to_dict(self) -> dict:
        """Return config dictionary for saving to YAML."""
        return dict(self._config)


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
        tokenizer_config: Optional[TokenizerConfig] = None,
        hf_model: Optional[str] = None,
        init_model_id: Optional[str] = None,
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
        return cls(OmegaConf.to_container(args, resolve=True))

    def to_dict(self) -> dict:
        """Return config dictionary for saving to YAML."""
        return dict(self._config)
