"""
Configuration classes for LAPT artifacts.

Each artifact (tokenizer, dataset, model) has a config class that:
1. Defines all parameters that affect the artifact
2. Provides from_args() to extract from full Hydra config
3. Provides cache_path() to generate consistent cache paths
4. Supports == comparison for cache validation
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import DictConfig, OmegaConf


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
class TokenizerConfig:
    """
    Configuration for FOCUS tokenizer training.

    Contains all parameters that affect the tokenizer artifact. Used for:
    - Generating cache paths
    - Validating cached tokenizers match current config
    - Passing to train_new_tokenizer()
    """
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
        model_name = self.hf_model.split('/')[-1]
        return model_name.lower().replace('-', '').replace('.', '')

    def cache_dir(self, language: str) -> str:
        """
        Generate tokenizer cache directory path.

        Args:
            language: Language code for the tokenizer

        Returns:
            Path like "tokenizers/got/xglm564m_focus-v16k-s100k_seeded-5.0x-lambda0.5"
        """
        model_short = self._model_shortname()
        suffix = self._build_suffix()
        return f"tokenizers/{language}/{model_short}_{suffix}"

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

    def focus_suffix(self) -> str:
        """
        Generate the full FOCUS suffix used in various paths.

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
