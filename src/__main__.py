import hydra
import os
import sys

from omegaconf import DictConfig, OmegaConf

import torch
from transformers import (
    DataCollatorForLanguageModeling, EarlyStoppingCallback,
    Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
)

from dataset_utils import (
    load_untokenized_dataset, load_or_tokenize_dataset,
    load_and_tokenize_external_eval_set
)
from model_utils import initialize_model_and_tokenizer, set_random_seeds, get_tokenizer_suffix, get_model_shortname, get_seed_tokenizer_suffix
from eval_utils import compute_ttr_metrics, preprocess_logits_for_metrics
from config_utils import (
    check_dataset_config, save_dataset_config,
    check_tokenizer_config, save_tokenizer_config,
    check_tokenized_config, save_tokenized_config,
    save_model_config
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
    if not args.focus.enabled:
        return None

    model_short = get_model_shortname(args.hf_model)
    tokenizer_suffix = get_tokenizer_suffix(args)
    return f"tokenizers/{args.dataset.language}/{model_short}_{tokenizer_suffix}"


def _get_tokenized_path(args: DictConfig) -> str:
    """
    Compute tokenized dataset path based on configuration.

    Args:
        args: Hydra configuration object

    Returns:
        Path to tokenized dataset directory
    """
    if args.focus.enabled:
        model_short = get_model_shortname(args.hf_model)
        tokenizer_suffix = get_tokenizer_suffix(args)
        return f"{args.dataset.cache_dir}/tokenized_{model_short}_{tokenizer_suffix}"
    else:
        return f"{args.dataset.cache_dir}/tokenized"


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

    # Build base path
    if args.focus.enabled:
        model_short = get_model_shortname(args.hf_model)
        tokenizer_suffix = get_tokenizer_suffix(args)
        base_path = f"{args.output_dir}/{args.dataset.language}/{model_short}_{tokenizer_suffix}_{training_config}"
    else:
        base_path = f"{args.output_dir}/{args.dataset.language}/{training_config}"

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
    tokenized_path = _get_tokenized_path(args)
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
            if args.focus.enabled and args.focus.use_seed_vocabulary:
                parent_dir = os.path.dirname(tokenizer_path)
                model_short = get_model_shortname(args.hf_model)
                seed_suffix = get_seed_tokenizer_suffix(
                    vocab_size=args.focus.vocab_size,
                    num_samples=args.focus.num_samples,
                    seed_vocab_multiplier=args.focus.seed_vocab_multiplier
                )
                seed_tokenizer_path = os.path.join(parent_dir, f"{model_short}_{seed_suffix}")
                if os.path.exists(seed_tokenizer_path):
                    print(f"  Removing {seed_tokenizer_path}", file=sys.stderr)
                    shutil.rmtree(seed_tokenizer_path)

        if os.path.exists(tokenized_path):
            print(f"  Removing {tokenized_path}", file=sys.stderr)
            shutil.rmtree(tokenized_path)
        if os.path.exists(output_dir):
            print(f"  Removing {output_dir}", file=sys.stderr)
            shutil.rmtree(output_dir)

        # Also clear FOCUS training data if applicable
        if args.focus.enabled:
            model_short = get_model_shortname(args.hf_model)
            tokenizer_suffix = get_tokenizer_suffix(args)
            focus_suffix = f"{model_short}_{tokenizer_suffix}"
            # FOCUS data is stored in the cache_dir of the dataset it was sampled from
            if hasattr(args.focus, 'dataset') and args.focus.dataset is not None:
                # Using separate FOCUS dataset
                focus_data_dir = f"{args.focus.dataset.cache_dir}/{focus_suffix}"
            else:
                # Using training dataset
                focus_data_dir = f"{args.dataset.cache_dir}/{focus_suffix}"

            if os.path.exists(focus_data_dir):
                print(f"  Removing {focus_data_dir}", file=sys.stderr)
                shutil.rmtree(focus_data_dir)

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

    # Handle cache cleanup if requested
    _handle_cache_cleanup(args)

    # Check if dataset cache and config exist
    dataset_cache_exists = os.path.exists(f"{args.dataset.cache_dir}/untokenized")
    dataset_config_exists = os.path.exists(f"{args.dataset.cache_dir}/untokenized/config.yaml")

    # Verify config matches if both cache and config exist
    if dataset_cache_exists and dataset_config_exists:
        check_dataset_config(args, args.dataset.cache_dir)
    elif dataset_cache_exists and not dataset_config_exists:
        print(
            f"Note: Using cached dataset at {args.dataset.cache_dir}/untokenized without config tracking\n"
            f"      (artifact was created before config tracking was implemented)",
            file=sys.stderr
        )

    # Load or download untokenized dataset first (needed for FOCUS or standard training)
    untokenized_path = load_untokenized_dataset(
        dataset_config=args.dataset,
        cache_dir=args.dataset.cache_dir,
        dev_size=args.training.dev_size
    )

    # Save config if we just created the dataset
    if not dataset_cache_exists:
        save_dataset_config(args, args.dataset.cache_dir)

    # Initialize model and tokenizer (with optional FOCUS)
    model, tokenizer, tokenized_path = initialize_model_and_tokenizer(args)

    # Determine output directory for checkpoints
    output_dir = _get_output_dir(args)

    # Check if tokenized dataset cache and config exist
    tokenized_cache_exists = os.path.exists(tokenized_path)
    tokenized_config_exists = os.path.exists(os.path.join(tokenized_path, "config.yaml"))

    # Verify config matches if both cache and config exist
    if tokenized_cache_exists and tokenized_config_exists:
        check_tokenized_config(args, tokenized_path)
    elif tokenized_cache_exists and not tokenized_config_exists:
        print(
            f"Note: Using cached tokenized dataset at {tokenized_path} without config tracking\n"
            f"      (artifact was created before config tracking was implemented)",
            file=sys.stderr
        )

    # Tokenize dataset with appropriate tokenizer
    dataset = load_or_tokenize_dataset(
        untokenized_path=untokenized_path,
        tokenized_path=tokenized_path,
        tokenizer=tokenizer,
        max_length=args.training.max_length,
        dev_size=args.training.dev_size
    )

    # Save config if we just created the tokenized dataset
    if not tokenized_cache_exists:
        save_tokenized_config(args, tokenized_path)

    # Prepare eval datasets - either single 'test' split or dict of per-language dev splits
    # Dev splits are any non-train splits except 'test'
    dev_splits = [key for key in dataset.keys() if key != 'train' and key != 'test']
    if dev_splits:
        # Multinomial sampling case: multiple per-language dev sets
        eval_dataset = {key: dataset[key] for key in dev_splits}
        print(f"Using {len(eval_dataset)} per-language eval sets for evaluation: {', '.join(dev_splits)}", file=sys.stderr)
    else:
        # Standard case: single dev/test split
        eval_dataset = dataset['test']

    # Load external evaluation sets if configured
    # Check both direct override (args.external_eval_sets) and config group (args.external_eval.external_eval_sets)
    external_eval_sets = args.get('external_eval_sets', None)
    if external_eval_sets is None and hasattr(args, 'external_eval'):
        external_eval_sets = args.external_eval.get('external_eval_sets', None)

    if external_eval_sets:
        # If eval_dataset is not already a dict, convert it
        if not isinstance(eval_dataset, dict):
            # Keep the original dev/test set as 'dev'
            eval_dataset = {'dev': eval_dataset}
            print("Converted single eval dataset to dict for external eval sets", file=sys.stderr)

        # Load and add each external eval set
        for eval_config in external_eval_sets:
            name = eval_config['name']

            # Check for name conflicts
            if name in eval_dataset:
                raise ValueError(
                    f"External eval set name '{name}' conflicts with existing eval set. "
                    f"Existing eval sets: {list(eval_dataset.keys())}"
                )

            external_dataset = load_and_tokenize_external_eval_set(
                eval_config=eval_config,
                tokenizer=tokenizer,
                max_length=args.training.max_length
            )
            eval_dataset[name] = external_dataset
            print(f"Added external eval set '{name}' with {len(external_dataset)} examples", file=sys.stderr)

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

    trainer = Trainer(**trainer_kwargs)

    broken_loss_callback = DetectBrokenLossCallback(trainer)
    trainer.add_callback(broken_loss_callback)

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
    save_model_config(args, output_dir)

    # start training (resume from checkpoint if specified)
    resume_checkpoint = args.get('resume_from_checkpoint', None)
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
