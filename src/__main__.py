import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from artifact_configs import (
    DatasetConfig,
    ModelConfig,
    TokenizedDatasetConfig,
    TokenizerConfig,
    resolve_dev_size,
)
from custom_trainer import FlooredPerExampleLossTrainer
from dataset_utils import (
    DataCollatorForInstructionTuning,
    is_instruction_dataset,
    load_tokenized_dataset,
    load_tokenized_multinomial_dataset,
    load_untokenized_dataset,
    prepare_eval_datasets,
)
from eval_utils import (
    BPCCallback,
    GenerationChrfCallback,
    compute_chars_per_token,
    compute_ttr_metrics,
    preprocess_logits_for_metrics,
)
from gradient_correction import (
    GradientCorrectionLogCallback,
    install_gradient_correction,
    load_z_estimate,
)
from model_utils import (
    get_init_model_identifier,
    get_tokenized_path,
    initialize_model_and_tokenizer,
    is_local_model_path,
    set_random_seeds,
)

OmegaConf.register_new_resolver("divide", lambda x, y: int(x / y))


class DetectBrokenLossCallback(TrainerCallback):
    """
    Callback to detect if the training loss goes to zero (indicating divergence) and stop training
    with an error if so
    """
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if 'loss' in state.log_history[-1] and state.log_history[-1]['loss'] <= 0.0:
            raise RuntimeError("Training loss dropped to zero, indicating divergence")


class InitialFreezeCallback(TrainerCallback):
    """
    Callback to freeze model parameters at the beginning of training.

    The parameter `model_freeze_prefix` controls which parameters to freeze
    (as a prefix of their name). This is useful for freezing the main transformer
    body while allowing embeddings to adapt during initial training.
    """
    def __init__(self, trainer: Trainer, model_freeze_prefix: str):
        self.trainer = trainer
        self.model_freeze_prefix = model_freeze_prefix

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        for name, param in self.trainer.model.named_parameters():
            if name.startswith(self.model_freeze_prefix):
                param.requires_grad = False
        print(
            f"\nAll parameters with prefix '{self.model_freeze_prefix}' frozen",
            file=sys.stderr
        )


class UnfreezeCallback(TrainerCallback):
    """
    Callback to unfreeze the entire model at a certain point in training.

    The parameter `unfreeze_step_ratio` controls when to unfreeze (as a ratio of
    the maximum training steps). For example, 0.1 means unfreeze after 10% of training.
    This allows embeddings to adapt to the model before fine-tuning the entire network.
    """
    def __init__(self, trainer: Trainer, unfreeze_step_ratio: float):
        self.trainer = trainer
        self.unfreeze_step_ratio = unfreeze_step_ratio
        self.already_unfrozen = False

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        reached_unfreeze_step = state.global_step >= int(self.unfreeze_step_ratio * state.max_steps)
        if reached_unfreeze_step and not self.already_unfrozen:
            # Save checkpoint before unfreezing (won't be deleted by checkpoint rotation)
            checkpoint_name = f"checkpoint-{state.global_step}-before-unfreeze"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
            self.trainer.save_model(checkpoint_path)
            print(
                f"\nSaved checkpoint before unfreezing: {checkpoint_path}",
                file=sys.stderr
            )

            # Unfreeze all parameters
            for param in self.trainer.model.parameters():
                param.requires_grad = True
            self.already_unfrozen = True
            print(
                f"All model parameters unfrozen after global step {state.global_step}",
                file=sys.stderr
            )


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    """
    Early stopping callback that delays activation until a certain point in training.

    This is useful when freezing parameters initially - you don't want early stopping
    to trigger before the full model has been unfrozen and had a chance to adapt.

    Args:
        early_stopping_patience: Number of evaluations without improvement before stopping
        early_stopping_delay_ratio: Don't allow early stopping until this proportion of
                                   training is complete (e.g., 0.2 = wait until 20%)
    """
    def __init__(self, early_stopping_patience: int, early_stopping_delay_ratio: float = 0.0):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.delay_ratio = early_stopping_delay_ratio
        self.delay_passed = False

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        # Check if we've passed the delay period
        if not self.delay_passed:
            delay_steps = int(self.delay_ratio * state.max_steps)
            if state.global_step >= delay_steps:
                self.delay_passed = True
                print(
                    f"\nEarly stopping now active (passed delay at step {state.global_step})",
                    file=sys.stderr
                )
            else:
                # Skip early stopping check - return without calling parent
                return

        # Delay has passed, use normal early stopping logic
        return super().on_evaluate(args, state, control, **kwargs)


def _get_tokenizer_path(args: DictConfig) -> str:
    """
    Compute tokenizer path based on FOCUS configuration.

    Args:
        args: Hydra configuration object

    Returns:
        Path to tokenizer directory, or None if FOCUS is disabled
    """
    tokenizer_config = TokenizerConfig.from_args(args)
    if tokenizer_config is None:
        return None
    return tokenizer_config.cache_dir(args.dataset.language)


def _validate_init_model_id(args: DictConfig):
    """
    Validate that init_model_id is provided when required.

    When hf_model points to a local checkpoint path (rather than a HuggingFace Hub model),
    init_model_id is required because we can't derive a meaningful short identifier from local paths.

    Args:
        args: Hydra configuration object

    Raises:
        ValueError: If init_model_id is required but not provided
    """
    init_model_id = getattr(args, 'init_model_id', None)
    if init_model_id:
        return  # init_model_id provided, all good

    if is_local_model_path(args.hf_model):
        raise ValueError(
            f"init_model_id is required when hf_model is a local path.\n"
            f"  hf_model: {args.hf_model}\n"
            f"  Add init_model_id=<short_identifier> to your config to identify this starting checkpoint.\n"
            f"  Example: init_model_id=v81 (for a Stage 1 checkpoint from experiment v81)"
        )


def _get_output_dir(args: DictConfig) -> str:
    """
    Compute model output directory based on configuration.

    Args:
        args: Hydra configuration object

    Returns:
        Path to model output directory
    """
    if args.model_name:
        return f"{args.output_dir}/{args.model_name}"

    training_config = args.training.name.replace('_', '-')
    init_model_identifier = get_init_model_identifier(args)

    # Build base path
    tokenizer_config = TokenizerConfig.from_args(args)
    if tokenizer_config is not None:
        base_path = f"{args.output_dir}/{args.dataset.language}/{tokenizer_config.tokenizer_id()}_{training_config}"
    else:
        base_path = f"{args.output_dir}/{args.dataset.language}/{init_model_identifier}_{training_config}"

    # Append experiment_id if provided
    experiment_id = getattr(args, 'experiment_id', None)
    if experiment_id:
        return f"{base_path}_{experiment_id}"
    else:
        return base_path


def _handle_cache_cleanup(args: DictConfig):
    """
    Handle selective cache cleanup based on fresh_* flags.

    Supports three levels of cleanup:
    - fresh_model: Clear only model checkpoints
    - fresh_tokenizer: Clear tokenizer, tokenized dataset, and model
    - fresh_dataset: Clear everything (dataset, tokenizer, model)

    Args:
        args: Hydra configuration object
    """
    import shutil

    fresh_dataset = getattr(args, 'fresh_dataset', False)
    fresh_tokenizer = getattr(args, 'fresh_tokenizer', False)
    fresh_model = getattr(args, 'fresh_model', False)

    if not any([fresh_dataset, fresh_tokenizer, fresh_model]):
        return

    print("=" * 60, file=sys.stderr)
    print("CACHE CLEANUP", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Compute paths that might need cleaning
    tokenizer_path = _get_tokenizer_path(args)
    tokenized_path = get_tokenized_path(args)
    output_dir = _get_output_dir(args)

    if fresh_dataset:
        # Clear everything
        print("fresh_dataset=true: Clearing dataset cache, tokenizer, and model", file=sys.stderr)
        if os.path.exists(args.dataset.cache_dir):
            print(f"  Removing {args.dataset.cache_dir}", file=sys.stderr)
            shutil.rmtree(args.dataset.cache_dir)
        if tokenizer_path and os.path.exists(tokenizer_path):
            # Remove entire tokenizer language directory (includes seed tokenizers and all variants)
            tokenizer_lang_dir = os.path.dirname(tokenizer_path)
            print(f"  Removing {tokenizer_lang_dir}", file=sys.stderr)
            shutil.rmtree(tokenizer_lang_dir)
        if os.path.exists(output_dir):
            print(f"  Removing {output_dir}", file=sys.stderr)
            shutil.rmtree(output_dir)

    elif fresh_tokenizer:
        # Clear tokenizer and downstream artifacts
        print("fresh_tokenizer=true: Clearing tokenizer, tokenized dataset, and model", file=sys.stderr)
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"  Removing {tokenizer_path}", file=sys.stderr)
            shutil.rmtree(tokenizer_path)

            # Also remove the seed tokenizer if using seed vocabulary
            tokenizer_config = TokenizerConfig.from_args(args)
            if tokenizer_config is not None and tokenizer_config.use_seed_vocabulary:
                parent_dir = os.path.dirname(tokenizer_path)
                seed_tokenizer_path = os.path.join(
                    parent_dir, tokenizer_config.seed_tokenizer_suffix()
                )
                if os.path.exists(seed_tokenizer_path):
                    print(f"  Removing {seed_tokenizer_path}", file=sys.stderr)
                    shutil.rmtree(seed_tokenizer_path)

        if os.path.exists(tokenized_path):
            print(f"  Removing {tokenized_path}", file=sys.stderr)
            shutil.rmtree(tokenized_path)
        if os.path.exists(output_dir):
            print(f"  Removing {output_dir}", file=sys.stderr)
            shutil.rmtree(output_dir)

        # Note: the FOCUS training_subset_*.jsonl file is now shared across
        # FOCUS runs on the same dataset mix, so fresh_tokenizer does NOT
        # delete it. Rebuilding the tokenizer from the same subset is fine;
        # use fresh_dataset to regenerate the subset itself.

    elif fresh_model:
        # Clear only model outputs
        print("fresh_model=true: Clearing model checkpoints only", file=sys.stderr)
        if os.path.exists(output_dir):
            print(f"  Removing {output_dir}", file=sys.stderr)
            shutil.rmtree(output_dir)

    print("=" * 60, file=sys.stderr)


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def lapt(args: DictConfig):
    set_random_seeds(args.seed)

    # Validate init_model_id is provided when hf_model is a local path
    _validate_init_model_id(args)

    # Handle cache cleanup if requested
    _handle_cache_cleanup(args)

    # Build dataset config for tracking
    dev_size = resolve_dev_size(args)

    # Build dataset config for tracking. For multinomial datasets, the
    # untokenized split lives in a mix-keyed subfolder of the parent cache_dir;
    # DatasetConfig.effective_cache_dir() resolves to that subfolder so config
    # tracking points at the mix-specific artifact.
    dataset_config = DatasetConfig.from_args(args)
    effective_cache_dir = dataset_config.effective_cache_dir(args.dataset.cache_dir)
    dataset_config_path = f"{effective_cache_dir}/untokenized/config.yaml"

    # Check if dataset cache exists
    dataset_cache_exists = os.path.exists(f"{effective_cache_dir}/untokenized")

    # Verify config matches if cache exists
    if dataset_cache_exists:
        if os.path.exists(dataset_config_path):
            dataset_config.check_cached(dataset_config_path)
        else:
            print(
                f"Note: Using cached dataset at {effective_cache_dir}/untokenized"
                f" without config tracking\n"
                f"      (artifact was created before config tracking was implemented)",
                file=sys.stderr
            )

    # Load or download untokenized dataset first (needed for FOCUS or standard training).
    # For multinomial, load_untokenized_dataset internally routes the output
    # to the mix subfolder; we still pass the parent cache_dir here so source
    # subdirs remain shared at the parent level.
    untokenized_path = load_untokenized_dataset(
        dataset_config=args.dataset,
        cache_dir=args.dataset.cache_dir,
        dev_size=dev_size
    )

    # Save config if we just created the dataset
    if not dataset_cache_exists:
        dataset_config.save(dataset_config_path)

    # Initialize model and tokenizer (with optional FOCUS)
    model, tokenizer, tokenized_path = initialize_model_and_tokenizer(args)

    # Install gradient correction for vocabulary-pruned continued training.
    # Gated on BOTH an explicit training flag (non-standard feature, off by
    # default) and the presence of z_estimate.json next to the tokenizer.
    # See src/gradient_correction.py.
    gradient_correction_enabled = False
    if args.training.get('correct_pruned_gradients', False):
        z_estimate_source = (
            args.focus.tokenizer_path if args.focus.tokenizer_path
            else _get_tokenizer_path(args)
        )
        mean_log_z = load_z_estimate(z_estimate_source) if z_estimate_source else None
        if mean_log_z is None:
            raise FileNotFoundError(
                f"training.correct_pruned_gradients=true but no z_estimate.json "
                f"found in tokenizer directory: {z_estimate_source}. "
                f"Run tools/estimate_partition_function.py first."
            )
        install_gradient_correction(model, mean_log_z)
        gradient_correction_enabled = True

    # Determine output directory for checkpoints
    output_dir = _get_output_dir(args)

    # Multinomial training mixes go through a plan-based path that tokenizes
    # each source's unique rows exactly once and represents the upsampled training
    # split as a shuffled index array into the concatenation of per-source
    # tokenized datasets. Other dataset types use the legacy single-artifact path.
    if args.dataset.type == 'multinomial':
        tokenizer_id = os.path.basename(tokenized_path).replace("tokenized_", "", 1)
        dataset = load_tokenized_multinomial_dataset(
            sources=OmegaConf.to_container(args.dataset.sources, resolve=True),
            alpha=args.dataset.get('alpha'),
            total_samples=args.dataset.total_samples,
            dev_size=dev_size,
            base_cache_dir=args.dataset.cache_dir,
            tokenizer=tokenizer,
            tokenizer_id=tokenizer_id,
            max_length=args.training.max_length,
        )
    else:
        # Build tokenized dataset config for tracking
        tokenized_dataset_config = TokenizedDatasetConfig.from_args(args)
        tokenized_config_path = os.path.join(tokenized_path, "config.yaml")

        # Check if tokenized dataset cache exists
        tokenized_cache_exists = os.path.exists(tokenized_path)

        # Verify config matches if cache exists
        if tokenized_cache_exists:
            if os.path.exists(tokenized_config_path):
                tokenized_dataset_config.check_cached(tokenized_config_path)
            else:
                print(
                    f"Note: Using cached tokenized dataset at {tokenized_path}"
                    f" without config tracking\n"
                    f"      (artifact was created before config tracking was implemented)",
                    file=sys.stderr
                )

        # Tokenize dataset with appropriate tokenizer
        dataset = load_tokenized_dataset(
            untokenized_path=untokenized_path,
            tokenized_path=tokenized_path,
            tokenizer=tokenizer,
            max_length=args.training.max_length,
            dev_size=dev_size
        )

        # Save config if we just created the tokenized dataset
        if not tokenized_cache_exists:
            tokenized_dataset_config.save(tokenized_config_path)

    # Prepare eval datasets (handles per-language dev splits and external eval sets)
    # Check both direct override and config group for external eval sets
    external_eval_sets = args.get('external_eval_sets', None)
    if external_eval_sets is None and hasattr(args, 'external_eval'):
        external_eval_sets = args.external_eval.get('external_eval_sets', None)

    eval_dataset = prepare_eval_datasets(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.training.max_length,
        external_eval_sets=external_eval_sets
    )

    # Compute chars-per-token ratios for BPC metric
    chars_per_token_ratios = compute_chars_per_token(eval_dataset, tokenizer)
    for prefix, ratio in chars_per_token_ratios.items():
        print(f"  {prefix} chars_per_token = {ratio:.2f}", file=sys.stderr)

    # for sanity, make sure all parameters require gradients initially;
    # this is mostly in response to new embeddings not having grads, but might as
    # well make sure everything is trainable at first
    for p in model.parameters():
        p.requires_grad = True

    # initialize trainer class with training configs
    # Note: Trainer automatically handles device placement (GPU/CPU/multi-GPU)
    training_args = TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        log_level="info",
        num_train_epochs=args.training.num_train_epochs,
        max_steps=args.training.max_steps,
        learning_rate=float(args.training.learning_rate),
        per_device_train_batch_size=args.training.train_batch_size,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        logging_steps=args.training.logging_steps,
        eval_strategy=args.training.eval_strategy,
        metric_for_best_model=args.training.metric_for_best_model,
        greater_is_better=args.training.get('greater_is_better', None),
        per_device_eval_batch_size=args.training.eval_batch_size,
        eval_steps=args.training.eval_steps,
        save_steps=args.training.save_steps,
        save_total_limit=args.training.save_total_limit,
        load_best_model_at_end=True,
        output_dir=output_dir,
        overwrite_output_dir=True,
        lr_scheduler_type=args.training.lr_scheduler_type,
        warmup_ratio=float(args.training.warmup_ratio),
        max_grad_norm=args.training.max_grad_norm,
        weight_decay=args.training.get('weight_decay', 0.0),
        gradient_checkpointing=args.training.gradient_checkpointing,
        bf16=args.training.get('bf16', False),
        fp16=args.training.get('fp16', False),
        optim=args.training.get('optim', 'adamw_torch'),
        dataloader_num_workers=args.training.get('dataloader_num_workers', 0),
        dataloader_pin_memory=args.training.get('dataloader_pin_memory', True),
        torch_compile=args.training.get('torch_compile', False),
        torch_compile_backend=args.training.get('torch_compile_backend', None),
        torch_compile_mode=args.training.get('torch_compile_mode', None)
    )

    # Choose appropriate data collator based on dataset type
    if is_instruction_dataset(dataset):
        print("Using instruction tuning collator (loss masking enabled)", file=sys.stderr)
        data_collator = DataCollatorForInstructionTuning(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )

    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'data_collator': data_collator,
        'train_dataset': dataset['train'],
        'eval_dataset': eval_dataset,
    }

    # Conditionally enable TTR metric computation
    compute_ttr = args.training.get('compute_ttr', False)
    if compute_ttr:
        trainer_kwargs['compute_metrics'] = compute_ttr_metrics
        trainer_kwargs['preprocess_logits_for_metrics'] = preprocess_logits_for_metrics

    loss_type = args.training.get('loss_type', 'token_mean')
    if loss_type == 'token_mean':
        trainer = Trainer(**trainer_kwargs)
    elif loss_type == 'per_example':
        per_example_loss_floor = args.training.get('per_example_loss_floor', 1)
        trainer = FlooredPerExampleLossTrainer(
            per_example_loss_floor=per_example_loss_floor,
            **trainer_kwargs,
        )
        print(
            f"Using floored per-example training loss "
            f"(per_example_loss_floor={per_example_loss_floor})",
            file=sys.stderr,
        )
    else:
        raise ValueError(
            f"Unknown training.loss_type: {loss_type!r}. "
            f"Expected 'token_mean' or 'per_example'."
        )

    broken_loss_callback = DetectBrokenLossCallback(trainer)
    trainer.add_callback(broken_loss_callback)

    bpc_callback = BPCCallback(chars_per_token_ratios)
    trainer.add_callback(bpc_callback)

    # Generation chrF: score greedy generations on held-out instruction sets and
    # log eval_<name>_chrf alongside the forward-pass bpc. Off unless configured.
    # Check both a direct override and the config group, mirroring external_eval.
    chrf_eval_sets = args.get('chrf_eval_sets', None)
    if chrf_eval_sets is None and hasattr(args, 'chrf_eval'):
        chrf_eval_sets = args.chrf_eval.get('chrf_eval_sets', None)
    if chrf_eval_sets:
        chrf_callback = GenerationChrfCallback(
            model=model,
            tokenizer=tokenizer,
            chrf_eval_sets=OmegaConf.to_container(chrf_eval_sets, resolve=True),
            max_prompt_length=args.training.max_length,
        )
        trainer.add_callback(chrf_callback)

    if gradient_correction_enabled:
        trainer.add_callback(GradientCorrectionLogCallback(model))

    if args.training.get('early_stopping_patience', None):
        delay_ratio = args.training.get('early_stopping_delay_ratio', 0.0)
        early_stopping_callback = DelayedEarlyStoppingCallback(
            early_stopping_patience=args.training.early_stopping_patience,
            early_stopping_delay_ratio=delay_ratio
        )
        trainer.add_callback(early_stopping_callback)

    if args.training.get('freeze_main_model', False):
        freeze_callback = InitialFreezeCallback(trainer, args.training.model_freeze_prefix)
        trainer.add_callback(freeze_callback)

        if args.training.get('unfreeze_step_ratio', None):
            unfreeze_callback = UnfreezeCallback(trainer, args.training.unfreeze_step_ratio)
            trainer.add_callback(unfreeze_callback)

    # Save the full training configuration for reproducibility
    model_config_path = os.path.join(output_dir, "training_config.yaml")
    ModelConfig.from_args(args).save(model_config_path)

    # start training (resume from checkpoint if specified)
    resume_checkpoint = args.get('resume_from_checkpoint', None)
    if resume_checkpoint is None and args.get('preempt_resume', False):
        resume_checkpoint = get_last_checkpoint(output_dir)
        if resume_checkpoint:
            print(f"Preempt resume: resuming from {resume_checkpoint}", file=sys.stderr)
        else:
            print("Preempt resume: no checkpoint found, starting fresh", file=sys.stderr)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # save the best model (loaded by trainer at end) to a known location
    best_checkpoint_path = os.path.join(output_dir, 'best-checkpoint')
    trainer.save_model(best_checkpoint_path)
    trainer.save_state()

    # save config in best-checkpoint directory too
    best_config_path = os.path.join(best_checkpoint_path, 'training_config.yaml')
    with open(best_config_path, 'w') as f:
        OmegaConf.save(args, f)

    print(f"Best model saved to: {best_checkpoint_path}", file=sys.stderr)

    # evaluate model
    trainer.evaluate()

if __name__ == "__main__":
    lapt()
