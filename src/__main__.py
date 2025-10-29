import hydra
import numpy as np
import os
import random
import sys

from omegaconf import DictConfig

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, EarlyStoppingCallback,
    Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
)

from dataset_utils import load_or_download_untokenized_dataset, load_or_tokenize_dataset
from focus_utils import (
    apply_focus_initialization,
    prepare_focus_training_data,
    train_new_tokenizer
)


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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Load or download untokenized dataset first (needed for FOCUS or standard training)
    untokenized_path = load_or_download_untokenized_dataset(
        dataset_path=args.dataset_path,
        language_code=args.language_code
    )

    # FOCUS workflow: prepare new tokenizer and embeddings
    if args.focus.enabled:
        print("=" * 60, file=sys.stderr)
        print("FOCUS MODE ENABLED", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # Prepare JSONL training data for FOCUS
        jsonl_path = prepare_focus_training_data(
            dataset_path=args.dataset_path,
            num_samples=args.focus.num_samples,
            output_jsonl_path=args.focus.training_data_output,
            seed=args.seed
        )

        # Load existing tokenizer or train a new one
        if args.focus.tokenizer_path:
            print(f"Loading tokenizer from {args.focus.tokenizer_path}", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(args.focus.tokenizer_path)
        else:
            tokenizer = train_new_tokenizer(
                jsonl_path=jsonl_path,
                base_tokenizer_name=args.hf_model,
                vocab_size=args.focus.vocab_size,
                output_path=args.focus.tokenizer_output_dir
            )

        # Load model and apply FOCUS
        print(f"Loading model: {args.hf_model}", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model)
        source_tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

        new_input_embeddings, new_output_embeddings = apply_focus_initialization(
            source_model=model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=tokenizer,
            training_data_path=jsonl_path
        )

        # Resize model vocabulary and replace embeddings
        model.resize_token_embeddings(len(tokenizer))

        # Set new input embeddings
        new_input_embedding_layer = torch.nn.Embedding.from_pretrained(
            new_input_embeddings,
            padding_idx=tokenizer.pad_token_id
        )
        model.set_input_embeddings(new_input_embedding_layer)

        # Set new output embeddings if model doesn't tie weights
        if hasattr(model.config, 'tie_word_embeddings') and not model.config.tie_word_embeddings:
            if new_output_embeddings is not None:
                model.get_output_embeddings().weight.data = new_output_embeddings
        else:
            # Tie weights for models that use tied embeddings
            model.tie_weights()

        del source_tokenizer
        print("FOCUS initialization complete", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # Determine tokenized dataset path for FOCUS (separate from standard tokenized data)
        tokenized_path = args.dataset_path + "/tokenized_focus"
    else:
        # Standard workflow: use pretrained tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model)
        tokenized_path = args.dataset_path + "/tokenized"

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
