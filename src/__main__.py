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


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def lapt(args: DictConfig):
    set_random_seeds(args.seed)

    # Load or download untokenized dataset first (needed for FOCUS or standard training)
    untokenized_path = load_untokenized_dataset(
        dataset_config=args.dataset,
        cache_dir=args.dataset.cache_dir,
        dev_size=args.training.dev_size
    )

    # Initialize model and tokenizer (with optional FOCUS)
    model, tokenizer, tokenized_path = initialize_model_and_tokenizer(args)

    # Determine output directory for checkpoints
    if args.model_name:
        # Codename override - use base_dir/codename
        output_dir = f"{args.output_dir}/{args.model_name}"
    else:
        # Auto-generate path from training conditions
        training_config = args.training.name.replace('_', '-')

        if args.focus.enabled:
            vocab_str = format_number(args.focus.vocab_size)
            samples_str = format_number(args.focus.num_samples)
            output_dir = f"{args.output_dir}/{args.dataset.language}/focus_vocab{vocab_str}_samples{samples_str}_{training_config}"
        else:
            output_dir = f"{args.output_dir}/{args.dataset.language}/{training_config}"

    # Tokenize dataset with appropriate tokenizer
    dataset = load_or_tokenize_dataset(
        untokenized_path=untokenized_path,
        tokenized_path=tokenized_path,
        tokenizer=tokenizer,
        max_length=args.training.max_length,
        dev_size=args.training.dev_size
    )

    # Prepare eval datasets - either single 'test' split or dict of per-language dev splits
    dev_splits = [key for key in dataset.keys() if key.startswith('dev_')]
    if dev_splits:
        # Multinomial sampling case: multiple per-language dev sets
        eval_dataset = {key: dataset[key] for key in dev_splits}
        print(f"Using {len(eval_dataset)} per-language dev sets for evaluation: {', '.join(dev_splits)}", file=sys.stderr)
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
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.training.early_stopping_patience
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
