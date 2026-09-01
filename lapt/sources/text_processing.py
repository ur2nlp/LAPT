"""Text-shaping helpers shared by the corpus source types.

These live here rather than in `lapt.dataset_utils` so that source modules can
use them without importing back into the module that imports *them*, which
would be circular.
"""

import json
import os
from itertools import chain


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
