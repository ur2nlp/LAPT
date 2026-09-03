"""Instruction-tuning source read from a JSONL file of prompt/response pairs."""

import sys

from datasets import Dataset, DatasetDict

from lapt.sources.base import SOURCE_TYPES, SourceDataset
from lapt.sources.factory import field
from lapt.sources.text_processing import read_instruction_jsonl


class InstructionJsonlDataset(SourceDataset):
    """An instruction corpus read from a JSONL file.

    Unlike the plaintext sources, which produce a single `text` column, this
    produces separate `prompt` and `response` columns so that training can mask
    the loss to response tokens only.
    """

    type_name = "instruction_jsonl"

    def __init__(self, cache_dir: str, file_path: str):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory is created in.
            file_path: JSONL file of `{prompt, response}` objects.
        """
        super().__init__(cache_dir)
        self.file_path = file_path

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        The seed is deliberately absent: every example in the file is kept, in
        file order, so nothing here is sampled.
        """
        return {'type': 'instruction_jsonl', 'path': self.file_path}

    def build(self, deps) -> DatasetDict:
        """Read the file into a prompt/response dataset.

        Args:
            deps: Unused; this source has no dependencies.

        Returns:
            A `DatasetDict` with a single `train` split carrying `prompt` and
            `response` columns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: On malformed JSON or a file with no valid examples.
        """
        print(f"Loading instruction data from {self.file_path}", file=sys.stderr)

        prompts, responses = read_instruction_jsonl(self.file_path)

        print(f"Loaded {len(prompts)} instruction examples from JSONL file", file=sys.stderr)

        return DatasetDict({
            'train': Dataset.from_dict({'prompt': prompts, 'response': responses}),
        })

    @classmethod
    def from_config(
        cls,
        cache_dir: str,
        source_config,
        seed: int = 1,
        dev_size: float | None = None,
    ) -> 'InstructionJsonlDataset':
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            source_config: Entry carrying `path`.
            seed: Unused; every example is kept in file order.
            dev_size: Unused; only a mix holds out a dev split.

        Returns:
            The configured source.
        """
        return cls(cache_dir, field(source_config, 'path'))

SOURCE_TYPES.register(InstructionJsonlDataset)
