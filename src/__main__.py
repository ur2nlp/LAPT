import hydra
import os
import sys

from omegaconf import DictConfig, OmegaConf

import torch
from transformers import (
    DataCollatorForLanguageModeling, EarlyStoppingCallback,
    Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
)

from dataset_utils import load_untokenized_dataset, load_or_tokenize_dataset
from model_utils import initialize_model_and_tokenizer, set_random_seeds, format_number


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
            for param in self.trainer.model.parameters():
                param.requires_grad = True
            self.already_unfrozen = True
            print(
                f"\nAll model parameters unfrozen after global step {state.global_step}",
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


def _get_focus_suffix(args: DictConfig) -> str:
    """
    Build FOCUS path suffix encoding vocab size, num samples, and additional token inheritance.

    Args:
        args: Hydra configuration object

    Returns:
        Suffix string like "vocab16k_samples25k" or "vocab16k_samples25k_no_additional"
    """
    vocab_str = format_number(args.focus.vocab_size)
    samples_str = format_number(args.focus.num_samples)
    suffix = f"vocab{vocab_str}_samples{samples_str}"

    if not args.focus.get('inherit_additional_special_tokens', True):
        suffix += "_no_additional"

    return suffix


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

    focus_suffix = _get_focus_suffix(args)
    return f"tokenizers/{args.dataset.language}/{focus_suffix}"


def _get_tokenized_path(args: DictConfig) -> str:
    """
    Compute tokenized dataset path based on configuration.

    Args:
        args: Hydra configuration object

    Returns:
        Path to tokenized dataset directory
    """
    if args.focus.enabled:
        focus_suffix = _get_focus_suffix(args)
        return f"{args.dataset.cache_dir}/tokenized_focus_{focus_suffix}"
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
        focus_suffix = _get_focus_suffix(args)
        base_path = f"{args.output_dir}/{args.dataset.language}/focus_{focus_suffix}_{training_config}"
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
            print(f"  Removing {tokenizer_path}", file=sys.stderr)
            shutil.rmtree(tokenizer_path)
        if os.path.exists(output_dir):
            print(f"  Removing {output_dir}", file=sys.stderr)
            shutil.rmtree(output_dir)

    elif fresh_tokenizer:
        # Clear tokenizer and downstream artifacts
        print("fresh_tokenizer=true: Clearing tokenizer, tokenized dataset, and model", file=sys.stderr)
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"  Removing {tokenizer_path}", file=sys.stderr)
            shutil.rmtree(tokenizer_path)
        if os.path.exists(tokenized_path):
            print(f"  Removing {tokenized_path}", file=sys.stderr)
            shutil.rmtree(tokenized_path)
        if os.path.exists(output_dir):
            print(f"  Removing {output_dir}", file=sys.stderr)
            shutil.rmtree(output_dir)

        # Also clear FOCUS training data if applicable
        if args.focus.enabled:
            focus_suffix = _get_focus_suffix(args)
            # FOCUS data is stored in the cache_dir of the dataset it was sampled from
            if args.focus.dataset is not None:
                # Using separate FOCUS dataset
                focus_data_dir = f"{args.focus.dataset.cache_dir}/focus_{focus_suffix}"
            else:
                # Using training dataset
                focus_data_dir = f"{args.dataset.cache_dir}/focus_{focus_suffix}"

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

    # Load or download untokenized dataset first (needed for FOCUS or standard training)
    untokenized_path = load_untokenized_dataset(
        dataset_config=args.dataset,
        cache_dir=args.dataset.cache_dir,
        dev_size=args.training.dev_size
    )

    # Initialize model and tokenizer (with optional FOCUS)
    model, tokenizer, tokenized_path = initialize_model_and_tokenizer(args)

    # Determine output directory for checkpoints
    output_dir = _get_output_dir(args)

    # Tokenize dataset with appropriate tokenizer
    dataset = load_or_tokenize_dataset(
        untokenized_path=untokenized_path,
        tokenized_path=tokenized_path,
        tokenizer=tokenizer,
        max_length=args.training.max_length,
        dev_size=args.training.dev_size
    )

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
        warmup_steps=args.training.warmup_steps,
        max_grad_norm=args.training.max_grad_norm,
        gradient_checkpointing=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=eval_dataset
    )

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

    # start training
    trainer.train()

    # save the best model (loaded by trainer at end) to a known location
    best_checkpoint_path = os.path.join(output_dir, 'best-checkpoint')
    trainer.save_model(best_checkpoint_path)
    trainer.save_state()
    print(f"Best model saved to: {best_checkpoint_path}", file=sys.stderr)

    # evaluate model
    trainer.evaluate()

if __name__ == "__main__":
    lapt()
