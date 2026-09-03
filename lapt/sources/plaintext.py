"""Plaintext file source: one training example per non-empty line."""

import os
import sys

from datasets import Dataset, DatasetDict

from lapt.sources.base import SOURCE_TYPES, SourceDataset
from lapt.sources.factory import field


class PlaintextDataset(SourceDataset):
    """A corpus read from a single plaintext file.

    Blank lines are dropped and surrounding whitespace stripped, so the example
    count is the number of non-empty lines rather than the file's line count.
    """

    type_name = "plaintext"

    def __init__(self, cache_dir: str, file_path: str):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory is created in.
            file_path: Plaintext file to read, one example per line.
        """
        super().__init__(cache_dir)
        self.file_path = file_path

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        The seed is deliberately absent: nothing in `build` is random, so a
        different seed must not invalidate this cache.
        """
        return {'type': 'plaintext', 'path': self.file_path}

    def build(self, deps) -> DatasetDict:
        """Read the file into a single-split dataset.

        Args:
            deps: Unused; this source has no dependencies.

        Returns:
            A `DatasetDict` with a single `train` split and a `text` column.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file has no non-empty lines.
        """
        print(f"Loading plaintext data from {self.file_path}", file=sys.stderr)

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Plaintext file not found: {self.file_path}")

        with open(self.file_path, encoding='utf-8') as plaintext_file:
            lines = [line.strip() for line in plaintext_file if line.strip()]

        if not lines:
            raise ValueError(f"Plaintext file {self.file_path} contains no non-empty lines")

        print(f"Loaded {len(lines)} lines from plaintext file", file=sys.stderr)

        return DatasetDict({'train': Dataset.from_dict({'text': lines})})

    @classmethod
    def from_config(
        cls,
        cache_dir: str,
        source_config,
        seed: int = 1,
        dev_size: float | None = None,
    ) -> 'PlaintextDataset':
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            source_config: Entry carrying `path`.
            seed: Unused; nothing here is random.
            dev_size: Unused; only a mix holds out a dev split.

        Returns:
            The configured source.
        """
        return cls(cache_dir, field(source_config, 'path'))

SOURCE_TYPES.register(PlaintextDataset)
