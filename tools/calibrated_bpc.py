"""
Temperature-calibrated bits-per-character for a trained checkpoint.

Rescores a run's held-out eval sets with the logits divided by a temperature
``T`` and reports the bpc at the best ``T`` alongside the raw (``T = 1``) bpc.
This separates two reasons held-out bpc can rise late in training:

* **miscalibration** -- the model's ranking of next tokens is still improving,
  but it has become overconfident, so the few tokens it gets wrong cost a lot of
  loss. One scalar temperature undoes most of this.
* **lost predictive quality** -- the distribution is worse in a way no single
  temperature can repair.

If the calibrated bpc of a late checkpoint is below the raw bpc minimum that the
same run logged earlier, the late rise was (at least mostly) miscalibration, and
an argmax-based metric such as greedy chrF can legitimately keep improving
while raw bpc rises.

The loss is aggregated exactly as the HuggingFace Trainer aggregates
``eval_loss`` (sequential batches, mean over response tokens within each batch,
batch means weighted by batch size), and converted to bpc with the same
``compute_chars_per_token`` ratio the training run used. The ``T = 1`` row
should therefore reproduce the logged ``eval_<name>_bpc`` at the checkpoint's
step, up to bf16 nondeterminism; check that before trusting the comparison.

The temperature is fit on the same holdout it is scored on. With one parameter
and thousands of tokens the in-sample optimism is negligible, but it is not
zero.

Usage:
    python tools/calibrated_bpc.py \
        --model models/gothic_instruct/<run>/best-checkpoint \
        --trainer-state outputs/trainer_states/<run>.json

    # Restrict to one eval set and write machine-readable results:
    python tools/calibrated_bpc.py --model ... \
        --eval-sets got-translation_holdout --output-json outputs/calibration/<run>.json

The eval sets, ``max_length``, and eval batch size default to those in the
``training_config.yaml`` that training saves into ``best-checkpoint``.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from lapt.evaluation import (
    DataCollatorForInstructionTuning,
    compute_chars_per_token,
    load_external_eval_set,
)

DEFAULT_TEMPERATURES = [round(0.8 + 0.05 * index, 2) for index in range(35)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temperature-calibrated bpc for a checkpoint's held-out eval sets.",
    )
    parser.add_argument(
        '--model',
        required=True,
        help="Checkpoint directory (model + tokenizer), e.g. <run>/best-checkpoint.",
    )
    parser.add_argument(
        '--config',
        default=None,
        help="Training config to read eval sets from (default: <model>/training_config.yaml).",
    )
    parser.add_argument(
        '--eval-sets',
        nargs='+',
        default=None,
        help="Names of external eval sets to score (default: all in the config).",
    )
    parser.add_argument(
        '--temperatures',
        nargs='+',
        type=float,
        default=DEFAULT_TEMPERATURES,
        help="Temperature grid; 1.0 is always added (default: 0.80 to 2.50 by 0.05).",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help="Eval batch size (default: training.eval_batch_size, to match the Trainer).",
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help="Score only the first N examples per set (smoke tests; breaks comparability).",
    )
    parser.add_argument(
        '--trainer-state',
        default=None,
        help="Run's trainer_state JSON; adds logged bpc at the best step and the logged minimum.",
    )
    parser.add_argument(
        '--device',
        default='auto',
        help="'auto', 'cpu', 'mps', or a CUDA device such as 'cuda:0'.",
    )
    parser.add_argument(
        '--no-autocast',
        action='store_true',
        help="Disable bf16 autocast on CUDA (training evals ran under bf16 autocast).",
    )
    parser.add_argument(
        '--output-json',
        default=None,
        help="Write per-set results (including the full temperature curve) to this file.",
    )
    return parser.parse_args()


def resolve_device(device_string: str) -> torch.device:
    """Pick a torch device, preferring CUDA, then MPS, then CPU for 'auto'."""
    if device_string != 'auto':
        return torch.device(device_string)
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def score_batch(
    model,
    batch: dict,
    temperatures: list[float],
    device: torch.device,
    use_autocast: bool,
) -> tuple[dict[float, float], int, int, torch.Tensor]:
    """Score one eval batch at every temperature.

    Args:
        model: Causal LM in eval mode.
        batch: Collated batch with 'input_ids', 'attention_mask', 'labels'.
        temperatures: Temperatures to score at.
        device: Device to run on.
        use_autocast: Whether to run the forward pass under bf16 autocast.

    Returns:
        Tuple of (mean NLL in nats per temperature, number of loss tokens,
        number of argmax-correct loss tokens, per-token NLL at T = 1 on CPU).
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            # (batch_size, seq_len, vocab_size)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # causal-LM shift: position t predicts token t + 1
        # (batch_size, seq_len - 1)
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100

        # keep only loss-contributing positions to bound memory across the grid
        # (num_valid, vocab_size)
        valid_logits = logits[:, :-1, :][valid_mask].float()
        # (num_valid,)
        valid_targets = shift_labels[valid_mask]

        num_valid = int(valid_targets.shape[0])
        num_correct = int((valid_logits.argmax(dim=-1) == valid_targets).sum().item())

        mean_nll_by_temperature = {}
        unit_temperature_nll = None
        for temperature in temperatures:
            # (num_valid, vocab_size)
            log_probs = torch.log_softmax(valid_logits / temperature, dim=-1)
            # (num_valid,)
            token_nll = -log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
            mean_nll_by_temperature[temperature] = token_nll.mean().item()
            if temperature == 1.0:
                unit_temperature_nll = token_nll.cpu()

    return mean_nll_by_temperature, num_valid, num_correct, unit_temperature_nll


def score_eval_set(
    model,
    dataset,
    collator,
    temperatures: list[float],
    batch_size: int,
    device: torch.device,
    use_autocast: bool,
) -> dict:
    """Score a tokenized eval set, aggregating loss the way the Trainer does.

    The Trainer repeats each batch's mean loss once per example and averages
    the result, so ``eval_loss`` is a batch-size-weighted mean of per-batch
    token means rather than a pure token mean. This mirrors that so the
    ``T = 1`` result matches the logged number.

    Returns:
        Dict with the Trainer-style loss per temperature, token accuracy, and
        median per-token NLL at ``T = 1``.
    """
    weighted_loss_sums = {temperature: 0.0 for temperature in temperatures}
    total_examples = 0
    total_tokens = 0
    total_correct = 0
    unit_temperature_nll_chunks = []

    for start in range(0, len(dataset), batch_size):
        features = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch = collator(features)
        mean_nll_by_temperature, num_valid, num_correct, unit_temperature_nll = score_batch(
            model=model,
            batch=batch,
            temperatures=temperatures,
            device=device,
            use_autocast=use_autocast,
        )
        if num_valid == 0:
            continue
        for temperature, mean_nll in mean_nll_by_temperature.items():
            weighted_loss_sums[temperature] += mean_nll * len(features)
        total_examples += len(features)
        total_tokens += num_valid
        total_correct += num_correct
        unit_temperature_nll_chunks.append(unit_temperature_nll)

    trainer_loss = {
        temperature: weighted_loss_sum / total_examples
        for temperature, weighted_loss_sum in weighted_loss_sums.items()
    }
    all_unit_temperature_nll = torch.cat(unit_temperature_nll_chunks)
    return {
        'trainer_loss': trainer_loss,
        'token_accuracy': total_correct / total_tokens,
        'median_token_nll': all_unit_temperature_nll.median().item(),
        'num_tokens': total_tokens,
    }


def best_step_from_state(trainer_state: dict) -> int | None:
    """Return the global step of the run's best checkpoint, if recorded."""
    if trainer_state.get('best_global_step') is not None:
        return int(trainer_state['best_global_step'])
    checkpoint_path = trainer_state.get('best_model_checkpoint')
    if checkpoint_path:
        match = re.search(r'checkpoint-(\d+)', checkpoint_path)
        if match:
            return int(match.group(1))
    return None


def logged_bpc_summary(trainer_state: dict, set_name: str) -> dict | None:
    """Pull the logged bpc at the best step and the logged minimum for one eval set."""
    bpc_key = f"eval_{set_name}_bpc"
    bpc_by_step = {
        entry['step']: entry[bpc_key]
        for entry in trainer_state['log_history']
        if bpc_key in entry
    }
    if not bpc_by_step:
        return None
    min_step = min(bpc_by_step, key=bpc_by_step.get)
    best_step = best_step_from_state(trainer_state)
    return {
        'best_step': best_step,
        'logged_bpc_at_best_step': bpc_by_step.get(best_step),
        'logged_min_bpc': bpc_by_step[min_step],
        'logged_min_step': min_step,
    }


def main():
    args = parse_args()

    config_path = Path(args.config) if args.config else Path(args.model) / 'training_config.yaml'
    config = OmegaConf.load(config_path)
    external_eval_sets = OmegaConf.to_container(
        config.external_eval.external_eval_sets, resolve=True,
    )
    if args.eval_sets is not None:
        configured_names = [eval_config['name'] for eval_config in external_eval_sets]
        missing = [name for name in args.eval_sets if name not in configured_names]
        if missing:
            raise ValueError(f"Eval sets {missing} not in config; available: {configured_names}")
        external_eval_sets = [
            eval_config for eval_config in external_eval_sets
            if eval_config['name'] in args.eval_sets
        ]

    batch_size = args.batch_size or config.training.eval_batch_size
    max_length = config.training.max_length
    temperatures = sorted(set(args.temperatures) | {1.0})

    device = resolve_device(args.device)
    use_autocast = device.type == 'cuda' and not args.no_autocast

    trainer_state = None
    if args.trainer_state:
        with open(args.trainer_state, encoding='utf-8') as handle:
            trainer_state = json.load(handle)

    print(f"Loading model from {args.model} on {device}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # match training-time eval: a checkpoint saved after left-padded generation can
    # reload with padding_side='left', and XGLM's position ids ignore the attention
    # mask, so left padding would shift every shorter example's positions
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.eval()
    collator = DataCollatorForInstructionTuning(tokenizer=tokenizer)

    results = {}
    for eval_config in external_eval_sets:
        name = eval_config['name']
        # add_labels matches an instruction-tuning run, where prepare_eval_datasets
        # gives plaintext external sets labels so they share the instruction collator
        dataset = load_external_eval_set(
            eval_config=eval_config,
            tokenizer=tokenizer,
            max_length=max_length,
            add_labels=True,
        )
        if args.max_examples is not None:
            dataset = dataset.select(range(min(args.max_examples, len(dataset))))

        # reuse the training run's conversion so bpc matches the logged values
        chars_per_token = compute_chars_per_token({name: dataset}, tokenizer)[f"eval_{name}"]

        scores = score_eval_set(
            model=model,
            dataset=dataset,
            collator=collator,
            temperatures=temperatures,
            batch_size=batch_size,
            device=device,
            use_autocast=use_autocast,
        )
        bpc_by_temperature = {
            temperature: loss / (chars_per_token * math.log(2))
            for temperature, loss in scores['trainer_loss'].items()
        }
        best_temperature = min(bpc_by_temperature, key=bpc_by_temperature.get)
        if best_temperature in (temperatures[0], temperatures[-1]) and best_temperature != 1.0:
            print(
                f"WARNING: best temperature for '{name}' is at the grid edge "
                f"({best_temperature}); widen --temperatures.",
                file=sys.stderr,
            )

        result = {
            'num_examples': len(dataset),
            'num_tokens': scores['num_tokens'],
            'chars_per_token': chars_per_token,
            'raw_bpc': bpc_by_temperature[1.0],
            'best_temperature': best_temperature,
            'calibrated_bpc': bpc_by_temperature[best_temperature],
            'token_accuracy': scores['token_accuracy'],
            'median_token_nll': scores['median_token_nll'],
            'bpc_by_temperature': {str(t): bpc for t, bpc in bpc_by_temperature.items()},
        }
        if trainer_state is not None:
            result['logged'] = logged_bpc_summary(trainer_state, name)
        results[name] = result

    # report: one block per eval set
    print(f"model: {args.model}")
    for name, result in results.items():
        print(f"\n{name}  ({result['num_examples']} examples, {result['num_tokens']} tokens)")
        print(f"  raw bpc (T=1.00)        {result['raw_bpc']:.4f}")
        print(
            f"  calibrated bpc (T={result['best_temperature']:.2f})  "
            f"{result['calibrated_bpc']:.4f}"
        )
        print(f"  token accuracy          {result['token_accuracy']:.4f}")
        print(f"  median token NLL (nats) {result['median_token_nll']:.4f}")
        logged = result.get('logged')
        if logged is None:
            continue
        if logged['logged_bpc_at_best_step'] is not None:
            print(
                f"  logged bpc @ step {logged['best_step']}  "
                f"{logged['logged_bpc_at_best_step']:.4f}  (should match raw bpc)"
            )
        print(
            f"  logged min bpc @ step {logged['logged_min_step']}  "
            f"{logged['logged_min_bpc']:.4f}"
        )
        margin = logged['logged_min_bpc'] - result['calibrated_bpc']
        print(f"  logged min - calibrated  {margin:+.4f}  (> 0: late rise is calibration)")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as handle:
            json.dump({'model': args.model, 'results': results}, handle, indent=2)
        print(f"\nWrote {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
