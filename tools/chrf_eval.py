"""
Batch chrF evaluation of a trained model over an instruction-formatted dataset.

Reads a JSONL instruction file (the same ``{"prompt": ..., "response": ...}``
format used for training / holdout sets), generates a continuation for each
prompt, and scores the generated text against the reference response with chrF
(sacrebleu). The corpus chrF is printed to stdout; optionally, a per-example
report with a metadata header is written to a file.

This is the batch, non-interactive counterpart to ``tools/interactive_prompt.py``
and reuses its model-loading helpers. Although chrF is usually applied to bare
translation, it works for any task whose reference is a sentence-length string
(e.g. CoT translation, transliteration): the full generated response is scored
against the full reference.

Usage:
    python tools/chrf_eval.py \
        --model models/got/some-run \
        --data data/gothic_instruct/translation_all-codices_both-scripts_both-directions_test_v1.1.0.jsonl

    # Write the translations + per-example scores to a file:
    python tools/chrf_eval.py --model ... --data ... --output outputs/chrf/run.txt

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


def generate_responses(
    generator,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    batch_size: int,
    stop_strings: list[str],
) -> list[str]:
    """Generate a continuation for each prompt, batched.

    Returns the newly generated text only (the prompt is stripped by the
    pipeline via ``return_full_text=False``), stripped of surrounding whitespace
    and truncated at any stop string.

    Args:
        generator: A HuggingFace text-generation pipeline.
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
        List of generated response strings, parallel to prompts.
    """
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_full_text=False,
        # Pass EOS/PAD explicitly from the tokenizer so a stale generation_config
        # (e.g. a base-model EOS id surviving a vocab swap) cannot silently
        # prevent the model from halting. Mirrors interactive_prompt.py.
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs['temperature'] = temperature
        generate_kwargs['top_p'] = top_p
    if repetition_penalty != 1.0:
        generate_kwargs['repetition_penalty'] = repetition_penalty
    if no_repeat_ngram_size > 0:
        generate_kwargs['no_repeat_ngram_size'] = no_repeat_ngram_size

    responses = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating", unit="batch"):
        batch = prompts[start:start + batch_size]
        outputs = generator(batch, batch_size=len(batch), **generate_kwargs)
        # For a list input the pipeline returns a list (per prompt) of lists
        # (per returned sequence) of dicts; we requested one sequence each.
        for output in outputs:
            generated = output[0]['generated_text'] if isinstance(output, list) else output['generated_text']
            responses.append(truncate_at_stop(generated.strip(), stop_strings))
    return responses


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
        "#" + "=" * 70,
        "",
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write("\n".join(header_lines))
        for index, (prompt, reference, hypothesis, score) in enumerate(
            zip(prompts, references, hypotheses, sentence_scores), start=1
        ):
            handle.write(f"[{index}] chrF={score:.2f}\n")
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

    hypotheses = generate_responses(
        generator=generator,
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

    print(f"\ncorpus {corpus_result.format()}")
    print(f"signature: {signature}")

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
        )


if __name__ == '__main__':
    main()
