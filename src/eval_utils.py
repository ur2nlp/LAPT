"""
Evaluation utilities for measuring prediction diversity and detecting model collapse.

This module provides metrics for diagnosing pathological model behavior, particularly
degenerate generation where models collapse to repeatedly predicting a small set of
high-frequency tokens. Also provides bits-per-character (BPC) computation for
tokenizer-agnostic evaluation.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sacrebleu.metrics import CHRF
from transformers import TrainerCallback
from typing import Dict


def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess logits before accumulation to reduce memory usage.

    This function is called by HuggingFace Trainer after each eval batch,
    before accumulating predictions. By converting full logits
    (batch, seq, vocab_size) to argmax token IDs (batch, seq), we reduce
    memory usage by ~256x for models with large vocabularies like XGLM.

    Without this, Trainer accumulates full logits from all eval batches in GPU
    memory, causing OOM errors with large vocabularies (256K tokens).

    Args:
        logits: (batch_size, seq_len, vocab_size) prediction logits from model
        labels: (batch_size, seq_len) ground truth labels (unused, but required by interface)

    Returns:
        (batch_size, seq_len) tensor of predicted token IDs
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_ttr_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute type-token ratio (TTR) to measure prediction diversity.

    This function measures whether the model is collapsing to a small set of
    majority-class predictions, which can happen after embedding reinitialization
    (e.g., with FOCUS). TTR (unique tokens / total tokens) captures vocabulary
    diversity within each sequence.

    Args:
        eval_pred: EvalPrediction object with:
            - predictions: (batch_size, seq_len) token IDs (already argmaxed by preprocess_logits_for_metrics)
            - label_ids: (batch_size, seq_len) with -100 for padding

    Returns:
        Dictionary with:
            - ttr-seq: Average unique tokens / seq_len per sequence

    Note:
        This function expects predictions to already be argmaxed token IDs, not logits.
        Use preprocess_logits_for_metrics with Trainer to ensure this.
    """
    predictions, labels = eval_pred

    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        pred_tokens = predictions.cpu().numpy()
    else:
        pred_tokens = predictions

    # Predictions should already be argmaxed by preprocess_logits_for_metrics
    # Shape: (batch_size, seq_len)
    if len(pred_tokens.shape) == 3:
        raise ValueError(
            "compute_ttr_metrics received 3D predictions (logits), but expects "
            "2D token IDs. Make sure to use preprocess_logits_for_metrics with Trainer."
        )

    # Create mask for non-padding positions (labels == -100 indicates padding)
    if labels is not None:
        mask = labels != -100
    else:
        # No padding information, use all positions
        mask = np.ones_like(pred_tokens, dtype=bool)

    # Compute TTR (type-token ratio) within each sequence, then average
    ttr_per_seq = []
    for i in range(len(pred_tokens)):
        seq_tokens = pred_tokens[i][mask[i]]
        seq_len = len(seq_tokens)
        if seq_len > 0:
            seq_unique = len(np.unique(seq_tokens))
            ttr_per_seq.append(seq_unique / seq_len)

    avg_ttr = np.mean(ttr_per_seq) if ttr_per_seq else 0.0

    return {
        'ttr-seq': avg_ttr,
    }


def compute_chars_per_token(
    eval_dataset,
    tokenizer,
) -> dict[str, float]:
    """Compute the average characters per loss-contributing token for each eval split.

    This ratio enables converting per-token eval loss (nats) to bits-per-character (BPC),
    which is comparable across tokenizers.

    For CLM, the model predicts token[i+1] from token[i], so the number of loss-contributing
    tokens per example is (non-masked labels) - 1.

    Args:
        eval_dataset: Either a single Dataset or a dict of Datasets (keyed by split name).
        tokenizer: The tokenizer used to encode the dataset, needed for decoding back to text.

    Returns:
        Mapping from metric prefix to chars_per_token ratio.
        For a single dataset: {"eval": 4.2}
        For a dict: {"eval_got": 4.2, "eval_ang": 3.8, ...}
    """
    if isinstance(eval_dataset, dict):
        splits = eval_dataset
    else:
        splits = {"__single__": eval_dataset}

    ratios = {}
    for split_name, split_ds in splits.items():
        total_chars = 0
        total_tokens = 0

        input_ids_list = split_ds["input_ids"]
        has_labels = "labels" in split_ds.column_names if hasattr(split_ds, "column_names") else False
        labels_list = split_ds["labels"] if has_labels else None

        for i in range(len(split_ds)):
            input_ids = input_ids_list[i]

            if labels_list is not None:
                labels = labels_list[i]
                # Decode only the response tokens (non-masked positions) so that
                # instruction prefix characters don't inflate chars_per_token.
                response_ids = [input_ids[j] for j, label in enumerate(labels) if label != -100]
                decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
                non_masked = len(response_ids)
            else:
                decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
                non_masked = len(input_ids)

            total_chars += len(decoded)

            # CLM shift: model predicts next token, so loss tokens = non_masked - 1
            loss_tokens = max(non_masked - 1, 0)
            total_tokens += loss_tokens

        if total_tokens == 0:
            ratio = 0.0
        else:
            ratio = total_chars / total_tokens

        if split_name == "__single__":
            ratios["eval"] = ratio
        else:
            ratios[f"eval_{split_name}"] = ratio

    return ratios


class BPCCallback(TrainerCallback):
    """Trainer callback that injects bits-per-character (BPC) into eval metrics.

    BPC = eval_loss / (chars_per_token * ln(2))

    This normalizes per-token cross-entropy loss by character count, enabling fair
    comparison across different tokenizers.

    Args:
        chars_per_token_ratios: Mapping from metric prefix (e.g. "eval", "eval_got")
            to the chars_per_token ratio for that split.
    """

    def __init__(self, chars_per_token_ratios: dict[str, float]):
        self.chars_per_token_ratios = chars_per_token_ratios

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        bpc_values = {}
        for prefix, chars_per_token in self.chars_per_token_ratios.items():
            loss_key = f"{prefix}_loss"
            bpc_key = f"{prefix}_bpc"
            if loss_key in metrics and chars_per_token > 0:
                bpc = metrics[loss_key] / (chars_per_token * math.log(2))
                metrics[bpc_key] = bpc
                bpc_values[bpc_key] = bpc

        # Trainer logs metrics before on_evaluate fires, so patch the most
        # recent log_history entry so BPC appears in saved trainer state
        if bpc_values and state.log_history:
            state.log_history[-1].update(bpc_values)


def load_instruction_prompts(file_path: str) -> tuple[list[str], list[str]]:
    """Load prompts and reference responses from an instruction JSONL file.

    Each line is a JSON object with 'prompt' and 'response' fields, the same
    format consumed by ``tools/chrf_eval.py`` and the instruction-tuning
    collator.

    Args:
        file_path: Path to the JSONL instruction file.

    Returns:
        Tuple of (prompts, references) lists, parallel.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is missing 'prompt' or 'response', or the file is
            empty.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"chrF eval data file not found: {file_path}")

    prompts = []
    references = []
    with path.open(encoding='utf-8') as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if 'prompt' not in obj or 'response' not in obj:
                raise ValueError(
                    f"Line {line_num} of {file_path} missing 'prompt' or 'response'."
                )
            prompts.append(obj['prompt'])
            references.append(obj['response'])

    if not prompts:
        raise ValueError(f"No examples loaded from {file_path}")
    return prompts, references


def truncate_at_stop(text: str, stop_strings: list[str]) -> str:
    """Truncate text at the first occurrence of any stop string."""
    cut = len(text)
    for stop in stop_strings:
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


def generate_greedy_batched(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    max_prompt_length: int,
    batch_size: int,
    stop_strings: list[str],
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> list[str]:
    """Greedily generate a continuation for each prompt using an in-memory model.

    This is the training-loop counterpart to ``tools/chrf_eval.py``'s pipeline
    generation: it drives the already-loaded ``model`` directly (no reload, no
    ``pipeline`` wrapper) so it can be called from a Trainer callback. Decoding
    is greedy for run-to-run reproducibility.

    Only the newly generated text is returned (the prompt tokens are sliced off),
    stripped and truncated at any stop string.

    Args:
        model: A causal-LM model (the unwrapped ``trainer.model``).
        tokenizer: The matching tokenizer.
        prompts: Prompts to continue.
        max_new_tokens: Maximum number of new tokens per generation.
        max_prompt_length: Truncate encoded prompts to this many tokens.
        batch_size: Number of prompts to generate in parallel.
        stop_strings: Substrings at which to truncate each generated response.
        repetition_penalty: Repetition penalty (1.0 = off). Match this to the
            offline ``chrf_eval.py`` protocol for comparable scores.
        no_repeat_ngram_size: No-repeat n-gram size (0 = off).

    Returns:
        List of generated response strings, parallel to ``prompts``.
    """
    device = next(model.parameters()).device

    # Decoder-only batched generation requires left padding so that generation
    # continues from the true end of each (right-aligned) prompt.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # gradient_checkpointing runs force use_cache=False; generation needs the KV
    # cache, so enable it for the duration and restore afterward.
    original_use_cache = getattr(model.config, 'use_cache', None)
    model.config.use_cache = True

    responses = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(device)

            # Some tokenizers emit token_type_ids, which decoder-only models like
            # XGLM do not accept; generate() rejects unused model kwargs, so drop it.
            encoded.pop('token_type_ids', None)

            with torch.no_grad():
                # Pass EOS/PAD explicitly from the tokenizer so a stale
                # generation_config (e.g. a base-model EOS id surviving a vocab
                # swap) cannot silently prevent the model from halting.
                generate_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                )
                if repetition_penalty != 1.0:
                    generate_kwargs['repetition_penalty'] = repetition_penalty
                if no_repeat_ngram_size > 0:
                    generate_kwargs['no_repeat_ngram_size'] = no_repeat_ngram_size
                generated = model.generate(**encoded, **generate_kwargs)

            prompt_length = encoded['input_ids'].shape[1]
            new_tokens = generated[:, prompt_length:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for text in decoded:
                responses.append(truncate_at_stop(text.strip(), stop_strings))
    finally:
        tokenizer.padding_side = original_padding_side
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache

    return responses


class GenerationChrfCallback(TrainerCallback):
    """Trainer callback that injects generation chrF into eval metrics.

    On every evaluation, this generates greedy continuations for each configured
    holdout set and scores them against their reference responses with chrF
    (sacrebleu), injecting ``eval_<name>_chrf`` so the surface-adequacy metric
    sits alongside the forward-pass ``eval_<name>_bpc`` (see ``BPCCallback``).

    bpc is a weakly-correct proxy for generation quality (right sign, low
    magnitude, and it misranks eras where a task mix inflates translation bpc
    without hurting output; see ``.claude/gothic/bpc_vs_chrf.md``); chrF closes
    that gap for run selection. Generation is far slower than the bpc forward
    pass, but the holdouts are small (tens of examples), so cost is bounded.

    The metric is *logged* by default, not selected. To make it drive
    best-model selection, set ``metric_for_best_model`` to an
    ``eval_<name>_chrf`` key and ``greater_is_better: true``.

    Args:
        model: The causal-LM model to generate with (unwrapped ``trainer.model``).
        tokenizer: The matching tokenizer.
        chrf_eval_sets: List of per-holdout config dicts. Each requires 'name'
            (matching its bpc holdout so metrics align) and 'path' (an
            instruction JSONL), with optional 'max_new_tokens', 'word_order'
            (0 = plain chrF, 2 = chrF++), 'batch_size', 'max_examples',
            'stop' (list of stop substrings), 'repetition_penalty', and
            'no_repeat_ngram_size'.
        max_prompt_length: Token cap for encoded prompts (typically the training
            max_length).
    """

    def __init__(
        self,
        model,
        tokenizer,
        chrf_eval_sets: list[dict],
        max_prompt_length: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

        # Pre-load prompts/references once so we don't re-read files every eval.
        self.specs = []
        for eval_config in chrf_eval_sets:
            name = eval_config['name']
            prompts, references = load_instruction_prompts(eval_config['path'])
            max_examples = eval_config.get('max_examples')
            if max_examples is not None:
                prompts = prompts[:max_examples]
                references = references[:max_examples]
            references = [reference.strip() for reference in references]
            self.specs.append({
                'name': name,
                'prompts': prompts,
                'references': references,
                'max_new_tokens': eval_config.get('max_new_tokens', 128),
                'word_order': eval_config.get('word_order', 0),
                'batch_size': eval_config.get('batch_size', 16),
                'stop': list(eval_config.get('stop') or []),
                'repetition_penalty': eval_config.get('repetition_penalty', 1.0),
                'no_repeat_ngram_size': eval_config.get('no_repeat_ngram_size', 0),
            })
            print(
                f"  chrF eval '{name}': {len(prompts)} examples from {eval_config['path']}",
                file=sys.stderr,
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        was_training = self.model.training
        self.model.eval()

        chrf_values = {}
        try:
            for spec in self.specs:
                hypotheses = generate_greedy_batched(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompts=spec['prompts'],
                    max_new_tokens=spec['max_new_tokens'],
                    max_prompt_length=self.max_prompt_length,
                    batch_size=spec['batch_size'],
                    stop_strings=spec['stop'],
                    repetition_penalty=spec['repetition_penalty'],
                    no_repeat_ngram_size=spec['no_repeat_ngram_size'],
                )
                chrf = CHRF(word_order=spec['word_order'])
                corpus_result = chrf.corpus_score(hypotheses, [spec['references']])
                chrf_key = f"eval_{spec['name']}_chrf"
                chrf_values[chrf_key] = corpus_result.score
        finally:
            if was_training:
                self.model.train()

        metrics.update(chrf_values)

        # Trainer logs metrics before on_evaluate fires, so patch the most
        # recent log_history entry so chrF appears in the saved trainer state.
        if chrf_values and state.log_history:
            state.log_history[-1].update(chrf_values)
