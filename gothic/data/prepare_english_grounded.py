"""Prepare English input-grounded instruction data for the Stage-2 IT mix.

The Gothic translation model tends to ignore the input and emit a memorized
Bible verse. To push the general-instruction substrate toward attending to the
prompt, this script adapts two English *extractive / grounded* sources into the
framework's ``{prompt, response}`` JSONL schema:

- ``dolly``  -- the grounded subset of ``databricks/databricks-dolly-15k``
  (categories ``closed_qa``, ``information_extraction``, ``summarization``),
  each of which carries a ``context`` the answer must be drawn from.
- ``squad``  -- ``rajpurkar/squad_v2`` extractive span QA, including the
  *unanswerable* questions, which are mapped to a canonical decline response so
  the model learns to check the passage before answering.

Output matches the Gothic tasks exactly: single-line prompts ending in
`` Response:`` and responses beginning with a single leading space (see
``gothic/instruction_format.py`` for why prompts stay single-line). Emitted to
``data/english_instruct/``.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

from gothic.instruction_format import flatten_prompt


# Canonical response for SQuAD v2 unanswerable questions. Kept fixed (not
# diversified) so the decline behavior has a single, learnable target.
SQUAD_NO_ANSWER_RESPONSE = "The answer is not in the passage."

# Dolly categories whose examples carry a non-empty ``context`` the answer must
# be grounded in. The remaining categories (open_qa, brainstorming, etc.) are
# free generation and are deliberately excluded.
DOLLY_GROUNDED_CATEGORIES = {
    "closed_qa",
    "information_extraction",
    "summarization",
}

# Prompt phrasings, lightly diversified per example (seeded) so the model does
# not bind the grounded behavior to a single template. Each phrasing embeds the
# instruction and the context passage; the answer must come from the passage.
DOLLY_PHRASINGS = [
    "{instruction}\n\nUse only the following passage to answer:\n{context}",
    "Passage:\n{context}\n\nBased only on the passage above, {instruction_lc}",
    "Read the passage and respond using only its content.\nPassage: {context}\nTask: {instruction}",
    "{instruction}\n\nContext:\n{context}",
]

SQUAD_PHRASINGS = [
    "Answer the question using only the passage. If the passage does not contain "
    "the answer, say so.\nPassage: {context}\nQuestion: {question}",
    "Passage:\n{context}\n\nBased only on the passage above, answer: {question}",
    "{question}\n\nAnswer using only this passage:\n{context}",
    "Read the passage and answer the question from it; if it is not answerable "
    "from the passage, say the answer is not in the passage.\n"
    "Passage: {context}\nQuestion: {question}",
]


def lowercase_first(text: str) -> str:
    """Lowercase the first character so an instruction can follow a clause."""
    if not text:
        return text
    return text[0].lower() + text[1:]


def make_example(prompt_body: str, response_text: str) -> dict[str, str]:
    """Assemble a single ``{prompt, response}`` example in canonical shape.

    Args:
        prompt_body: The fully-rendered prompt before the `` Response:`` cue.
        response_text: The raw target text (newlines and edge whitespace are
            collapsed; a single leading space is then added).

    Returns:
        A dict with single-line ``prompt`` (ending `` Response:``) and
        ``response`` (beginning with one leading space).
    """
    prompt = f"{flatten_prompt(prompt_body)} Response:"
    response = f" {flatten_prompt(response_text)}"
    return {"prompt": prompt, "response": response}


def prepare_dolly(output_path: Path, seed: int) -> int:
    """Adapt the grounded Dolly subset to JSONL. Returns the example count."""
    rng = random.Random(seed)
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_file:
        for example in dataset:
            category = example["category"]
            context = example["context"].strip()
            instruction = example["instruction"].strip()
            response = example["response"].strip()
            if category not in DOLLY_GROUNDED_CATEGORIES:
                continue
            if not context or not response:
                continue
            phrasing = rng.choice(DOLLY_PHRASINGS)
            prompt_body = phrasing.format(
                instruction=instruction,
                instruction_lc=lowercase_first(instruction),
                context=context,
            )
            record = make_example(prompt_body, response)
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def prepare_squad(
    output_path: Path,
    seed: int,
    split: str,
    answerable_only: bool,
) -> int:
    """Adapt a SQuAD v2 split to JSONL. Returns the example count.

    Args:
        output_path: Destination JSONL path.
        seed: Random seed for prompt-phrasing selection.
        split: Which SQuAD v2 split to read (``train`` or ``validation``).
        answerable_only: If True, drop the unanswerable questions. Used for the
            holdout: unanswerable targets are all the identical canonical decline
            string, so their response bpc measures memorization of one phrase
            rather than abstention quality and is not worth tracking.

    Returns:
        The number of examples written.
    """
    rng = random.Random(seed)
    # Load the parquet files directly rather than via the dataset's hub
    # metadata: that metadata references the ``List`` feature type introduced in
    # ``datasets`` 4.x, which the pinned 3.6.0 cannot parse. Parquet inference
    # recovers the same ``answers`` struct from the data itself.
    dataset = load_dataset(
        "parquet",
        data_files=f"hf://datasets/rajpurkar/squad_v2/squad_v2/{split}-*.parquet",
        split="train",
    )
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_file:
        for example in dataset:
            context = example["context"].strip()
            question = example["question"].strip()
            answer_texts = example["answers"]["text"]
            if answer_texts:
                response = answer_texts[0].strip()
            elif answerable_only:
                continue
            else:
                response = SQUAD_NO_ANSWER_RESPONSE
            if not context or not question or not response:
                continue
            phrasing = rng.choice(SQUAD_PHRASINGS)
            prompt_body = phrasing.format(context=context, question=question)
            record = make_example(prompt_body, response)
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        choices=["dolly", "squad"],
        help="Which English grounded source to prepare.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation"],
        help=(
            "Which split to prepare (squad only). 'validation' emits an "
            "answerable-only holdout; 'train' emits the full training file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/english_instruct",
        help="Directory for the emitted JSONL file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for prompt-phrasing selection.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "dolly":
        if args.split != "train":
            parser.error("--source dolly only supports --split train")
        output_path = output_dir / "dolly-grounded_closed-qa-extract-summ_train.jsonl"
        count = prepare_dolly(output_path, args.seed)
    elif args.split == "validation":
        output_path = output_dir / "squad-v2_answerable_holdout.jsonl"
        count = prepare_squad(output_path, args.seed, split="validation", answerable_only=True)
    else:
        output_path = output_dir / "squad-v2_extractive-qa_train.jsonl"
        count = prepare_squad(output_path, args.seed, split="train", answerable_only=False)

    print(f"Wrote {count} examples to {output_path}", file=sys.stdout)


if __name__ == "__main__":
    main()
