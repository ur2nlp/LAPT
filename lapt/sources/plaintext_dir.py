"""Plaintext directory source: every file matching a glob, read as one corpus."""

import glob
import os
import sys

from datasets import Dataset, DatasetDict

from lapt.sources.base import SOURCE_TYPES
from lapt.sources.factory import field
from lapt_core.dataset_artifacts import DatasetArtifact


class PlaintextDirDataset(DatasetArtifact):
    """Every file in a directory matching a pattern, concatenated in name order.

    Blank lines are dropped and surrounding whitespace stripped, exactly as for
    a single plaintext file; the only difference is how many files are read.

    Unlike the pre-artifact implementation, this does not build a `concat` of
    one plaintext source per file. A directory of shards is one corpus, not a
    composition of independently-addressable sources, and giving each file its
    own child cache meant a second copy of the whole corpus on disk for no
    benefit -- nothing ever referred to an individual shard by id.
    """

    type_name = "plaintext_dir"

    def __init__(self, cache_dir: str, directory: str, pattern: str = '*.txt'):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory is created in.
            directory: Directory to read files from.
            pattern: Glob pattern selecting files, e.g. `*.on.txt`.
        """
        super().__init__(cache_dir)
        self.directory = directory
        self.pattern = pattern

    def _matching_files(self) -> list[str]:
        """Return the matching files in name order.

        Returns:
            Absolute-or-configured paths, sorted so the concatenation order is
            reproducible rather than filesystem-dependent.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If the path is not a directory, or nothing matches.
        """
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory not found: {self.directory}")
        if not os.path.isdir(self.directory):
            raise ValueError(f"Path is not a directory: {self.directory}")

        file_paths = sorted(glob.glob(os.path.join(self.directory, self.pattern)))
        if not file_paths:
            raise ValueError(
                f"No files found matching pattern '{self.pattern}' in {self.directory}"
            )
        return file_paths

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        The resolved file list is recorded, not just the directory and pattern:
        adding or removing a shard changes the corpus while leaving both of
        those identical, and that has to invalidate the cache. File *contents*
        are not tracked, matching `plaintext` -- editing a file in place is
        what `fresh_dataset` is for.

        The seed is deliberately absent: nothing in `build` is random.
        """
        return {
            'type': 'plaintext_dir',
            'directory': self.directory,
            'pattern': self.pattern,
            'files': [os.path.basename(path) for path in self._matching_files()],
        }

    def build(self, deps) -> DatasetDict:
        """Read every matching file into a single-split dataset.

        Args:
            deps: Unused; this source has no dependencies.

        Returns:
            A `DatasetDict` with a single `train` split and a `text` column.

        Raises:
            ValueError: If no file has any non-empty line.
        """
        file_paths = self._matching_files()
        print(
            f"Found {len(file_paths)} files matching '{self.pattern}' in {self.directory}",
            file=sys.stderr,
        )

        lines = []
        for file_path in file_paths:
            with open(file_path, encoding='utf-8') as text_file:
                file_lines = [line.strip() for line in text_file if line.strip()]
            print(f"  {os.path.basename(file_path)}: {len(file_lines)} lines", file=sys.stderr)
            lines.extend(file_lines)

        if not lines:
            raise ValueError(
                f"Files matching '{self.pattern}' in {self.directory} "
                "contain no non-empty lines"
            )

        print(f"Loaded {len(lines)} lines from {len(file_paths)} files", file=sys.stderr)
        return DatasetDict({'train': Dataset.from_dict({'text': lines})})

    @classmethod
    def from_config(
        cls,
        cache_dir: str,
        source_config,
        seed: int = 1,
        dev_size: float | None = None,
    ) -> 'PlaintextDirDataset':
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            source_config: Entry carrying `directory` and optionally `pattern`.
            seed: Unused; nothing here is random.
            dev_size: Unused; only a mix holds out a dev split.

        Returns:
            The configured source.
        """
        return cls(
            cache_dir,
            field(source_config, 'directory'),
            field(source_config, 'pattern', '*.txt'),
        )


SOURCE_TYPES.register(PlaintextDirDataset)
