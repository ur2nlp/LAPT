"""Text-shaping helpers shared by the corpus source types.

These live here rather than in `lapt.dataset_utils` so that source modules can
use them without importing back into the module that imports *them*, which
would be circular.
"""

import json
import os
from itertools import chain

from datasets import Dataset


def docs_to_lines(examples: dict) -> dict:
    """Convert document-based examples to line-based examples.

    Corpora such as OSCAR arrive as documents containing newlines. Splitting
    each document into its lines gives more granular training examples.

    Args:
        examples: Batch of examples with a `text` field of documents.

    Returns:
        A dict with a `text` field of individual lines, blank lines removed.
    """
    return {
        'text': list(chain(
            *[[line.strip() for line in doc.split('\n') if line.strip()]
              for doc in examples['text']]
        ))
    }


def read_instruction_jsonl(file_path: str) -> tuple[list[str], list[str]]:
    """Read prompts and responses from an instruction JSONL file.

    Each line is a JSON object with `prompt` and `response` fields, e.g.
    `{"prompt": "Translate to Gothic: hello\\nResponse:", "response": " hails"}`.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A tuple of (prompts, responses).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On malformed JSON, a line missing either field, or a file
            with no valid examples.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instruction JSONL file not found: {file_path}")

    prompts = []
    responses = []
    with open(file_path, encoding='utf-8') as jsonl_file:
        for line_num, raw_line in enumerate(jsonl_file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as decode_error:
                raise ValueError(f"Invalid JSON on line {line_num}: {decode_error}")

            if 'prompt' not in obj or 'response' not in obj:
                raise ValueError(
                    f"Line {line_num} missing 'prompt' or 'response' field. "
                    f"Got keys: {list(obj.keys())}"
                )
            prompts.append(obj['prompt'])
            responses.append(obj['response'])

    if not prompts:
        raise ValueError(f"JSONL file {file_path} contains no valid examples")

    return prompts, responses


def collect_from_stream(stream, limit: int) -> Dataset:
    """
    Collect examples from a streaming dataset up to a limit.

    Args:
        stream: An iterable of examples (typically from load_dataset with streaming=True)
        limit: Maximum number of examples to collect

    Returns:
        Dataset containing the collected examples
    """
    samples = []
    for i, example in enumerate(stream):
        if i >= limit:
            break
        samples.append(example)
    return Dataset.from_list(samples)


def docs_to_filtered_lines(
    dataset: Dataset,
    text_column: str = 'text',
    min_words_per_line: int = None,
    split_into_lines: bool = True,
) -> Dataset:
    """
    Convert document-based dataset to (optionally) line-based format with filtering.

    This helper standardizes the transformation pipeline used by HuggingFace dataset loaders.

    Args:
        dataset: Dataset with document text
        text_column: Name of the text column (will be renamed to 'text' if different)
        min_words_per_line: Minimum words per kept example (None to skip filtering).
            Applies per line when splitting, per document otherwise.
        split_into_lines: If True (default), split each document on newlines into
            one example per line. If False, keep each document as a single example
            (newlines preserved), parallel to the instruction-data loaders.

    Returns:
        Dataset with one example per line (or per document when
        ``split_into_lines`` is False).
    """
    # Standardize column name to 'text' if needed
    if text_column != 'text':
        dataset = dataset.rename_column(text_column, 'text')

    # Convert to line-based format (split documents on newlines)
    if split_into_lines:
        original_columns = dataset.column_names
        dataset = dataset.map(
            docs_to_lines,
            batched=True,
            remove_columns=original_columns
        )
    elif set(dataset.column_names) != {'text'}:
        # Drop any extra metadata columns so the schema matches other loaders.
        dataset = dataset.remove_columns(
            [column for column in dataset.column_names if column != 'text']
        )

    # Filter short examples if specified
    if min_words_per_line is not None:
        dataset = dataset.filter(
            lambda x: len(x['text'].split()) >= min_words_per_line
        )

    return dataset
