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


def compute_ttr_metrics(eval_pred) -> dict[str, float]:
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


def count_new_tokens(row_ids, eos_token_id: int) -> tuple[int, bool]:
    """Measure how many tokens a single generation actually produced.

    ``generate`` right-pads every finished sequence out to the length of the
    longest one in the batch, so the raw row length overstates a short
    generation. The true length is the number of tokens up to and including the
    first EOS; if no EOS is present the model never halted and was cut off by
    ``max_new_tokens``. Counting to the first EOS also stays correct when PAD is
    EOS, which is the usual setup here.

    Args:
        row_ids: 1-D tensor of the newly generated token ids for one example
            (the prompt already sliced off).
        eos_token_id: The EOS id that was passed to ``generate``.

    Returns:
        Tuple of (number of tokens generated, whether the generation hit the
        ``max_new_tokens`` cap without emitting EOS).
    """
    eos_positions = (row_ids == eos_token_id).nonzero()
    if eos_positions.numel() == 0:
        return int(row_ids.shape[0]), True
    first_eos_index = int(eos_positions[0].item())
    return first_eos_index + 1, False


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
) -> tuple[list[str], list[int], list[bool]]:
    """Greedily generate a continuation for each prompt using an in-memory model.

    This is the training-loop counterpart to ``tools/chrf_eval.py``'s pipeline
    generation: it drives the already-loaded ``model`` directly (no reload, no
    ``pipeline`` wrapper) so it can be called from a Trainer callback. Decoding
    is greedy for run-to-run reproducibility.

    Only the newly generated text is returned (the prompt tokens are sliced off),
    stripped and truncated at any stop string. Per-example token counts are
    returned alongside it, measured *before* stop-string truncation so that a
    generation which rambled to the cap but happened to emit a stop string early
    is still counted as having rambled.

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
        Tuple of (responses, new-token counts, hit-cap flags), all parallel to
        ``prompts``.
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
    new_token_counts = []
    hit_cap_flags = []
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
            for row_index, text in enumerate(decoded):
                token_count, hit_cap = count_new_tokens(
                    new_tokens[row_index], tokenizer.eos_token_id
                )
                new_token_counts.append(token_count)
                hit_cap_flags.append(hit_cap)
                responses.append(truncate_at_stop(text.strip(), stop_strings))
    finally:
        tokenizer.padding_side = original_padding_side
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache

    return responses, new_token_counts, hit_cap_flags


class GenerationChrfCallback(TrainerCallback):
    """Trainer callback that injects generation chrF into eval metrics.

    On every evaluation, this generates greedy continuations for each configured
    holdout set and scores them against their reference responses with chrF
    (sacrebleu), injecting ``eval_<name>_chrf`` so the surface-adequacy metric
    sits alongside the forward-pass ``eval_<name>_bpc`` (see ``BPCCallback``).

    It also injects ``eval_<name>_hit_cap_frac``: the fraction of generations
    that ran to ``max_new_tokens`` without emitting EOS. This is free (the
    generation already happened) and is a diagnostic, not a quality score -- a
    model that emits EOS immediately scores a perfect 0.0, so never select on
    it. It is only comparable across runs sharing the same ``max_new_tokens``,
    since raising the cap lowers the rate without the model changing.

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
            'stop' (list of stop substrings), 'repetition_penalty',
            'no_repeat_ngram_size', and 'hit_cap_warn_threshold' (fraction of
            non-halting generations above which a stderr warning fires;
            default 0.25).
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

        # Track the last step generation ran, so a multi-dataset eval (which fires
        # on_evaluate once per split) generates each spec only once per cycle.
        self._last_generated_step = -1

        # Holdouts already warned about for a high non-halting rate, so the
        # warning fires on the transition rather than every eval cycle.
        self._warned_hit_cap = set()

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
                'hit_cap_warn_threshold': eval_config.get('hit_cap_warn_threshold', 0.25),
            })
            print(
                f"  chrF eval '{name}': {len(prompts)} examples from {eval_config['path']}",
                file=sys.stderr,
            )

    def _maybe_warn_hit_cap(self, spec: dict, hit_cap_frac: float, global_step: int):
        """Warn to stderr the first time a holdout's non-halting rate is high.

        A high rate means either ``max_new_tokens`` is miscalibrated for this
        model and task, or the model is failing to halt at all -- the latter
        being the fingerprint of a stale ``generation_config`` EOS id surviving
        a vocabulary swap. Either way it is worth catching early rather than
        finding it later in a plot, so it is announced rather than only logged.

        Warns once per holdout, re-arming if the rate falls back below the
        threshold, so a persistent problem does not spam every eval cycle.
        """
        name = spec['name']
        if hit_cap_frac < spec['hit_cap_warn_threshold']:
            self._warned_hit_cap.discard(name)
            return
        if name in self._warned_hit_cap:
            return
        self._warned_hit_cap.add(name)
        print(
            f"WARNING: chrF eval '{name}' at step {global_step}: "
            f"{hit_cap_frac:.1%} of generations hit max_new_tokens "
            f"({spec['max_new_tokens']}) without emitting EOS. Either the cap is "
            f"too low for this task or the model is not halting (check that the "
            f"model's generation_config EOS id matches the tokenizer's).",
            file=sys.stderr,
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        # With a dict eval_dataset, HuggingFace evaluates each split in a separate
        # recursive evaluate() call, so on_evaluate fires once per split at the same
        # global_step. Generate only on the first firing of each cycle; otherwise
        # every chrF spec would be regenerated once per eval dataset, multiplying
        # cost by their count. Gating on the step (not on which split fired) also
        # lets a chrF set have no forward-pass loss/bpc twin among the eval splits.
        if state.global_step == self._last_generated_step:
            return
        self._last_generated_step = state.global_step

        was_training = self.model.training
        self.model.eval()

        chrf_values = {}
        try:
            for spec in self.specs:
                hypotheses, _, hit_cap_flags = generate_greedy_batched(
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

                hit_cap_frac = sum(hit_cap_flags) / len(hit_cap_flags)
                chrf_values[f"eval_{spec['name']}_hit_cap_frac"] = hit_cap_frac
                self._maybe_warn_hit_cap(spec, hit_cap_frac, state.global_step)
        finally:
            if was_training:
                self.model.train()

        metrics.update(chrf_values)

        # Trainer logs metrics before on_evaluate fires, so patch the most
        # recent log_history entry so chrF appears in the saved trainer state.
        if chrf_values and state.log_history:
            state.log_history[-1].update(chrf_values)
