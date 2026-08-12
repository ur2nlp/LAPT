"""
Batch chrF evaluation of a trained model over an instruction-formatted dataset.

Reads a JSONL instruction file (the same ``{"prompt": ..., "response": ...}``
format used for training / holdout sets), generates a continuation for each
prompt, and scores the generated text against the reference response with chrF
(sacrebleu). The corpus chrF is printed to stdout; optionally, a per-example
report with a metadata header is written to a file.

Alongside chrF, this reports generation-length behavior: how many new tokens
each generation actually produced, and what fraction ran to the
``--max-tokens`` cap without ever emitting EOS. That non-halting rate is a
cheap behavioral flag (see ``.claude/gothic/generation_eval.md``) and comes
free with generation you are already paying for.

This is the batch, non-interactive counterpart to ``tools/interactive_prompt.py``
and reuses its model-loading helpers. Although chrF is usually applied to bare
translation, it works for any task whose reference is a sentence-length string
(e.g. CoT translation, transliteration): the full generated response is scored
against the full reference.

Usage:
    python tools/chrf_eval.py \
        --model models/got/some-run \
        --data data/gothic_instruct/translation_all-codices_both-scripts_both-directions_test_v1.1.0.jsonl

    # Write the translations + per-example scores to a file, plus length stats:
    python tools/chrf_eval.py --model ... --data ... \
        --output outputs/chrf/run.txt --stats-json outputs/chrf/run.json

    # Sampling instead of greedy, larger batch, chrF++ (word bigrams):
    python tools/chrf_eval.py --model ... --data ... \
        --sample --temperature 0.8 --batch-size 32 --chrf-word-order 2

Decoding defaults to greedy so the score is reproducible across invocations,
unlike interactive_prompt.py which samples by default.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

# Reuse the device/dtype/model-loading machinery. This file lives in tools/, so
# tools/ is on sys.path[0] when run as a script and interactive_prompt imports
# directly. Import it (and thus transformers/torch) before sacrebleu: on macOS,
# importing torch after another libomp-linked library (numpy via sacrebleu) can
# trip an OpenMP duplicate-runtime abort.
from interactive_prompt import load_model, resolve_device, resolve_dtype

import torch
from sacrebleu.metrics import CHRF
from tqdm import tqdm


def load_instruction_jsonl(file_path: str) -> tuple[list[str], list[str]]:
    """Load prompts and reference responses from an instruction JSONL file.

    Each line is a JSON object with 'prompt' and 'response' fields, e.g.
    {"prompt": "Translate to Gothic: ... Response:", "response": " ..."}.

    Args:
        file_path: Path to the JSONL instruction file.

    Returns:
        Tuple of (prompts, references) lists.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

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
    ``max_new_tokens``.

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


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    batch_size: int,
    stop_strings: list[str],
) -> tuple[list[str], list[int], list[bool]]:
    """Generate a continuation for each prompt, batched.

    Drives ``model.generate`` directly rather than going through the
    ``text-generation`` pipeline, so the raw generated token ids are available:
    the pipeline only hands back decoded text, which cannot distinguish a model
    that halted on EOS from one that rambled to ``max_new_tokens`` and was cut
    off. This mirrors ``src/eval_utils.generate_greedy_batched``, the
    training-loop counterpart, which slices the prompt off by token index for
    the same reason.

    Returns the newly generated text only, stripped of surrounding whitespace
    and truncated at any stop string, alongside per-example token counts
    measured *before* stop-string truncation.

    Args:
        model: A causal-LM model.
        tokenizer: The matching tokenizer, already set to left padding.
        prompts: Prompts to continue.
        max_new_tokens: Maximum number of new tokens per generation.
        do_sample: Whether to sample (True) or decode greedily (False).
        temperature: Sampling temperature (used only when do_sample).
        top_p: Nucleus sampling threshold (used only when do_sample).
        repetition_penalty: Repetition penalty (1.0 = off).
        no_repeat_ngram_size: No-repeat n-gram size (0 = off).
        batch_size: Number of prompts to generate in parallel.
        stop_strings: Substrings at which to truncate each generated response.

    Returns:
        Tuple of (responses, new-token counts, hit-cap flags), all parallel to
        prompts.
    """
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        # Pass EOS/PAD explicitly from the tokenizer so a stale generation_config
        # (e.g. a base-model EOS id surviving a vocab swap) cannot silently
        # prevent the model from halting. Mirrors interactive_prompt.py.
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
    )
    if do_sample:
        generate_kwargs['temperature'] = temperature
        generate_kwargs['top_p'] = top_p
    if repetition_penalty != 1.0:
        generate_kwargs['repetition_penalty'] = repetition_penalty
    if no_repeat_ngram_size > 0:
        generate_kwargs['no_repeat_ngram_size'] = no_repeat_ngram_size

    responses = []
    new_token_counts = []
    hit_cap_flags = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating", unit="batch"):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors='pt', padding=True).to(device)

        # Some tokenizers emit token_type_ids, which decoder-only models like
        # XGLM do not accept; generate() rejects unused model kwargs, so drop it.
        encoded.pop('token_type_ids', None)

        with torch.no_grad():
            generated = model.generate(**encoded, **generate_kwargs)

        prompt_length = encoded['input_ids'].shape[1]
        new_tokens = generated[:, prompt_length:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row_index, text in enumerate(decoded):
            token_count, hit_cap = count_new_tokens(new_tokens[row_index], tokenizer.eos_token_id)
            new_token_counts.append(token_count)
            hit_cap_flags.append(hit_cap)
            responses.append(truncate_at_stop(text.strip(), stop_strings))
    return responses, new_token_counts, hit_cap_flags


def summarize_lengths(
    new_token_counts: list[int],
    hit_cap_flags: list[bool],
    max_new_tokens: int,
) -> dict:
    """Summarize how long generations ran and how often they never halted.

    Args:
        new_token_counts: Per-example counts of newly generated tokens.
        hit_cap_flags: Per-example flags for hitting max_new_tokens without EOS.
        max_new_tokens: The cap that was in force.

    Returns:
        Dict of summary statistics, suitable for JSON serialization.
    """
    num_examples = len(new_token_counts)
    num_hit_cap = sum(hit_cap_flags)
    sorted_counts = sorted(new_token_counts)
    median_index = num_examples // 2
    if num_examples % 2 == 1:
        median = float(sorted_counts[median_index])
    else:
        median = (sorted_counts[median_index - 1] + sorted_counts[median_index]) / 2

    return {
        'num_examples': num_examples,
        'max_new_tokens': max_new_tokens,
        'num_hit_cap': num_hit_cap,
        'frac_hit_cap': num_hit_cap / num_examples,
        'mean_new_tokens': sum(new_token_counts) / num_examples,
        'median_new_tokens': median,
        'min_new_tokens': sorted_counts[0],
        'max_observed_new_tokens': sorted_counts[-1],
    }


def write_report(
    output_path: str,
    args: argparse.Namespace,
    device_name: str,
    model_dtype,
    corpus_result,
    signature: str,
    prompts: list[str],
    references: list[str],
    hypotheses: list[str],
    sentence_scores: list[float],
    new_token_counts: list[int],
    hit_cap_flags: list[bool],
    length_summary: dict,
) -> None:
    """Write a per-example chrF report with a metadata header."""
    decoding = "greedy" if not args.sample else (
        f"sampling (temperature={args.temperature}, top_p={args.top_p})"
    )
    header_lines = [
        "# chrF evaluation report",
        f"# generated_at: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"# model: {args.model}",
        f"# data: {args.data}",
        f"# num_examples: {len(prompts)}",
        f"# device: {device_name}",
        f"# dtype: {model_dtype}",
        f"# decoding: {decoding}",
        f"# max_new_tokens: {args.max_tokens}",
        f"# repetition_penalty: {args.repetition_penalty}",
        f"# no_repeat_ngram_size: {args.no_repeat_ngram_size}",
        f"# stop_strings: {args.stop!r}",
        f"# chrF: {corpus_result.format()}",
        f"# signature: {signature}",
        f"# hit_max_new_tokens: {length_summary['num_hit_cap']}/{length_summary['num_examples']}"
        f" ({length_summary['frac_hit_cap']:.1%})",
        f"# new_tokens: mean={length_summary['mean_new_tokens']:.1f}"
        f" median={length_summary['median_new_tokens']:.1f}"
        f" min={length_summary['min_new_tokens']}"
        f" max={length_summary['max_observed_new_tokens']}",
        "#" + "=" * 70,
        "",
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write("\n".join(header_lines))
        for index, (prompt, reference, hypothesis, score, token_count, hit_cap) in enumerate(
            zip(prompts, references, hypotheses, sentence_scores, new_token_counts, hit_cap_flags),
            start=1,
        ):
            cap_marker = " HIT-CAP" if hit_cap else ""
            handle.write(f"[{index}] chrF={score:.2f} new_tokens={token_count}{cap_marker}\n")
            handle.write(f"PROMPT: {prompt}\n")
            handle.write(f"REF:    {reference.strip()}\n")
            handle.write(f"HYP:    {hypothesis}\n")
            handle.write("\n")
    print(f"Wrote report to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Batch chrF evaluation of a model over an instruction JSONL dataset"
    )
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model directory')
    parser.add_argument('--data', type=str, required=True, help='Path to the instruction JSONL file')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Optional path to write a per-example report (with metadata header)'
    )
    parser.add_argument(
        '--stats-json', type=str, default=None,
        help='Optional path to write generation-length statistics as machine-readable JSON'
    )
    parser.add_argument(
        '--max-examples', type=int, default=None,
        help='Evaluate only the first N examples (default: all)'
    )
    parser.add_argument('--batch-size', type=int, default=16, help='Generation batch size (default: 16)')
    parser.add_argument('--max-tokens', type=int, default=128, help='Max new tokens per generation (default: 128)')
    parser.add_argument('--sample', action='store_true', help='Enable sampling (default: greedy decoding)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature (default: 0.7)')
    parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold (default: 0.9)')
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty (1.0 = off)')
    parser.add_argument('--no-repeat-ngram-size', type=int, default=0, help='No-repeat n-gram size (0 = off)')
    parser.add_argument(
        '--stop', action='append', default=None, metavar='STR',
        help='Truncate each generated response at this substring (repeatable, e.g. a next-turn marker)'
    )
    parser.add_argument(
        '--chrf-word-order', type=int, default=0,
        help='Word n-gram order for chrF++ (0 = plain chrF, 2 = standard chrF++; default: 0)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device: "auto" (default), "cuda", "cpu", or "cuda:N"'
    )
    parser.add_argument(
        '--dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16', 'fp32'],
        help='Model dtype: "auto" (bf16 on CUDA, fp32 on CPU), "bf16", "fp16", or "fp32"'
    )
    parser.add_argument('--compile', action='store_true', help='Compile the model with torch.compile')
    args = parser.parse_args()

    stop_strings = args.stop or []

    prompts, references = load_instruction_jsonl(args.data)
    if args.max_examples is not None:
        prompts = prompts[:args.max_examples]
        references = references[:args.max_examples]
    print(f"Loaded {len(prompts)} examples from {args.data}", file=sys.stderr)

    device, device_name = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    print(f"Using device: {device_name}, dtype: {model_dtype}", file=sys.stderr)
    generator = load_model(args.model, device, model_dtype, args.compile)

    # Decoder-only batched generation requires left padding so that generation
    # continues from the true end of each (right-aligned) prompt.
    generator.tokenizer.padding_side = 'left'
    if generator.tokenizer.pad_token_id is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token

    hypotheses, new_token_counts, hit_cap_flags = generate_responses(
        model=generator.model,
        tokenizer=generator.tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        batch_size=args.batch_size,
        stop_strings=stop_strings,
    )

    references_stripped = [reference.strip() for reference in references]

    chrf = CHRF(word_order=args.chrf_word_order)
    corpus_result = chrf.corpus_score(hypotheses, [references_stripped])
    signature = str(chrf.get_signature())
    sentence_scores = [
        chrf.sentence_score(hypothesis, [reference]).score
        for hypothesis, reference in zip(hypotheses, references_stripped)
    ]

    length_summary = summarize_lengths(new_token_counts, hit_cap_flags, args.max_tokens)

    print(f"\ncorpus {corpus_result.format()}")
    print(f"signature: {signature}")
    print(
        f"hit max_new_tokens ({args.max_tokens}): "
        f"{length_summary['num_hit_cap']}/{length_summary['num_examples']} "
        f"({length_summary['frac_hit_cap']:.1%})"
    )
    print(
        f"new tokens: mean={length_summary['mean_new_tokens']:.1f} "
        f"median={length_summary['median_new_tokens']:.1f} "
        f"min={length_summary['min_new_tokens']} "
        f"max={length_summary['max_observed_new_tokens']}"
    )

    if args.stats_json:
        stats = dict(length_summary)
        stats['model'] = args.model
        stats['data'] = args.data
        stats['chrf'] = corpus_result.score
        stats['chrf_signature'] = signature
        stats['new_token_counts'] = new_token_counts
        stats['hit_cap_flags'] = hit_cap_flags
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open('w', encoding='utf-8') as handle:
            json.dump(stats, handle, indent=2)
        print(f"Wrote generation stats to {args.stats_json}", file=sys.stderr)

    if args.output:
        write_report(
            output_path=args.output,
            args=args,
            device_name=device_name,
            model_dtype=model_dtype,
            corpus_result=corpus_result,
            signature=signature,
            prompts=prompts,
            references=references,
            hypotheses=hypotheses,
            sentence_scores=sentence_scores,
            new_token_counts=new_token_counts,
            hit_cap_flags=hit_cap_flags,
            length_summary=length_summary,
        )


if __name__ == '__main__':
    main()
