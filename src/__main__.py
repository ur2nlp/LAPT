import hydra
import os
import sys

from omegaconf import DictConfig

import torch
from transformers import (
    DataCollatorForLanguageModeling, EarlyStoppingCallback,
    Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
)

from dataset_utils import load_or_download_untokenized_dataset, load_or_tokenize_dataset
from model_utils import initialize_model_and_tokenizer, set_random_seeds


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
        

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def lapt(args: DictConfig):
    set_random_seeds(args.seed)

    # Load or download untokenized dataset first (needed for FOCUS or standard training)
    untokenized_path = load_or_download_untokenized_dataset(
        dataset_path=args.dataset_path,
        language_code=args.language_code
    )

    # Initialize model and tokenizer (with optional FOCUS)
    model, tokenizer, tokenized_path = initialize_model_and_tokenizer(args)

    # Tokenize dataset with appropriate tokenizer
    dataset = load_or_tokenize_dataset(
        untokenized_path=untokenized_path,
        tokenized_path=tokenized_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        dev_size=args.dev_size
    )

    # for sanity, make sure all parameters require gradients initially;
    # this is mostly in response to new embeddings not having grads, but might as
    # well make sure everything is trainable at first
    for p in model.parameters():
        p.requires_grad = True

    # move model to GPU if available
    device = torch.device('cuda')
    model.to(device)

    # initialize trainer class with training configs
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
        per_device_eval_batch_size=args.training.eval_batch_size,
        eval_steps=args.training.eval_steps,
        save_steps=args.training.save_steps,
        save_total_limit=args.training.save_total_limit,
        load_best_model_at_end=True,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        lr_scheduler_type=args.training.lr_scheduler_type,
        warmup_ratio=float(args.training.warmup_ratio),
        warmup_steps=args.training.warmup_steps,
        max_grad_norm=args.training.max_grad_norm
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    broken_loss_callback = DetectBrokenLossCallback(trainer)
    trainer.add_callback(broken_loss_callback)

    if getattr(args, 'early_stopping_patience', False):
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        trainer.add_callback(early_stopping_callback)

    # start training
    trainer.train()

    best_checkpoint_path = trainer.state.best_model_checkpoint
    print(f"Best checkpoint: {best_checkpoint_path}", file=sys.stderr)

    best_checkpoint_path = os.path.join(args.checkpoints_directory, 'best-checkpoint')
    trainer.save_model(best_checkpoint_path)
    trainer.save_state()

    # evaluate model
    trainer.evaluate()

if __name__ == "__main__":
    lapt()
