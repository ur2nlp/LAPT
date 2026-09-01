"""Generic HuggingFace corpus source, optionally subsampled to a target size."""

import random
import sys

from datasets import Dataset, DatasetDict, load_dataset

from lapt.sources.base import SOURCE_TYPES, SourceDataset
from lapt.sources.text_processing import collect_from_stream, docs_to_filtered_lines


class HuggingFaceDataset(SourceDataset):
    """A corpus drawn from any HuggingFace dataset.

    Two shapes, chosen by `max_samples`. Without it, the whole split is
    downloaded and converted. With it, documents are streamed and oversampled
    for diversity, then subsampled at random to exactly `max_samples` examples
    — which is the only randomness in this source, and the reason its cache is
    keyed on the seed when and only when `max_samples` is set.
    """

    type_name = "huggingface"

    def __init__(
        self,
        cache_dir: str,
        name: str,
        config: str | None = None,
        split: str = 'train',
        text_column: str = 'text',
        max_samples: int | None = None,
        min_words_per_line: int | None = None,
        oversampling_factor: int = 3,
        split_into_lines: bool = True,
        seed: int = 1,
    ):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory is created in.
            name: HuggingFace dataset name, e.g. `allenai/c4`.
            config: Dataset configuration or subset, e.g. `en`.
            split: Which split to load.
            text_column: Column holding the text; renamed to `text`.
            max_samples: Cap on the number of examples, triggering streaming
                and a random subsample. An example is a line when
                `split_into_lines`, otherwise a whole document.
            min_words_per_line: Drop examples with fewer space-separated words
                than this, e.g. to remove section titles.
            oversampling_factor: Download this many times more documents than
                estimated necessary, for diversity, before subsampling.
            split_into_lines: Split each document on newlines into one example
                per line, the historical behavior. When False, keep each
                document whole with newlines preserved.
            seed: Seed for the subsample. Recorded only when `max_samples` is
                set, since it has no effect otherwise.
        """
        super().__init__(cache_dir)
        self.hf_name = name
        self.hf_config = config
        self.split = split
        self.text_column = text_column
        self.max_samples = max_samples
        self.min_words_per_line = min_words_per_line
        self.oversampling_factor = oversampling_factor
        self.split_into_lines = split_into_lines
        self.seed = seed

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        The seed appears only when `max_samples` is set. Recording it
        unconditionally would invalidate a whole-split cache whenever the seed
        changed, even though nothing about that cache depends on it.
        """
        tracked = {
            'type': 'huggingface',
            'name': self.hf_name,
            'config': self.hf_config,
            'split': self.split,
            'text_column': self.text_column,
            'max_samples': self.max_samples,
            'min_words_per_line': self.min_words_per_line,
            'oversampling_factor': self.oversampling_factor,
            'split_into_lines': self.split_into_lines,
        }
        if self.max_samples is not None:
            tracked['seed'] = self.seed
        return tracked

    @property
    def _example_unit(self) -> str:
        """Word for one example, for logging."""
        return "lines" if self.split_into_lines else "documents"

    def _estimate_documents_needed(self) -> int:
        """Estimate how many documents yield `max_samples` examples.

        Splitting documents into lines multiplies the example count by an
        unknown factor, so sample a small batch and measure it. Without
        splitting, one document is one example and no estimate is needed.

        Returns:
            Number of documents to download, including oversampling headroom.

        Raises:
            ValueError: If the sampled batch yields no examples at all, which
                means the filters cannot be satisfied by this dataset.
        """
        if not self.split_into_lines:
            return self.max_samples * self.oversampling_factor

        stream = load_dataset(self.hf_name, self.hf_config, split=self.split, streaming=True)

        estimation_sample_size = min(1000, self.max_samples // 10)
        print(
            f"  Phase 1: Sampling {estimation_sample_size} documents to estimate lines/doc",
            file=sys.stderr,
        )

        estimation_dataset = collect_from_stream(stream, estimation_sample_size)
        num_estimation_docs = len(estimation_dataset)
        estimation_dataset = docs_to_filtered_lines(
            estimation_dataset,
            self.text_column,
            self.min_words_per_line,
            self.split_into_lines,
        )

        lines_per_doc = (
            len(estimation_dataset) / num_estimation_docs if num_estimation_docs else 1
        )
        print(
            f"  Estimated {lines_per_doc:.1f} lines per document (after all filters)",
            file=sys.stderr,
        )

        if lines_per_doc == 0:
            raise ValueError(
                f"Estimation phase found 0 lines per document after filtering. "
                f"This suggests min_words_per_line={self.min_words_per_line} is too strict, "
                f"or the dataset has no suitable content."
            )

        return int((self.max_samples / lines_per_doc) * self.oversampling_factor)

    def _download(self) -> Dataset:
        """Fetch the documents, streaming only when capped by `max_samples`.

        Returns:
            The raw documents, before conversion to the target schema.
        """
        if not self.max_samples:
            return load_dataset(self.hf_name, self.hf_config, split=self.split)

        docs_needed = self._estimate_documents_needed()
        print(f"  Downloading {docs_needed} documents total", file=sys.stderr)

        # restart the stream rather than resume it: the estimation pass above
        # consumed an unknown prefix, and combining the two is more fragile
        # than paying for the re-read
        stream = load_dataset(self.hf_name, self.hf_config, split=self.split, streaming=True)
        dataset = collect_from_stream(stream, docs_needed)
        print(f"  Downloaded {len(dataset)} documents", file=sys.stderr)
        return dataset

    def _filter_short_examples(self, dataset: Dataset) -> Dataset:
        """Drop examples below `min_words_per_line`, reporting the loss.

        Args:
            dataset: Dataset with a `text` column.

        Returns:
            The filtered dataset, or the original if no minimum is set.
        """
        if self.min_words_per_line is None:
            return dataset

        original_size = len(dataset)
        dataset = dataset.filter(
            lambda example: len(example['text'].split()) >= self.min_words_per_line
        )
        filtered_size = len(dataset)
        print(
            f"  Filtered {original_size - filtered_size} {self._example_unit} with "
            f"< {self.min_words_per_line} words "
            f"({filtered_size} {self._example_unit} remaining)",
            file=sys.stderr,
        )

        if self.max_samples and filtered_size < self.max_samples:
            print(
                f"Warning: After filtering, only {filtered_size} {self._example_unit} remain, "
                f"but {self.max_samples} requested. Consider increasing oversampling_factor "
                f"(current: {self.oversampling_factor}) or reducing min_words_per_line.",
                file=sys.stderr,
            )

        return dataset

    def _subsample(self, dataset: Dataset) -> Dataset:
        """Reduce the dataset to exactly `max_samples` examples, at random.

        Oversampling deliberately downloads more than needed for document
        diversity; this is where the excess is discarded. Selected indices are
        sorted, so the result keeps the corpus's own order.

        Args:
            dataset: Dataset with a `text` column.

        Returns:
            The subsampled dataset, or the original if no cap applies.
        """
        if not self.max_samples:
            return dataset

        if len(dataset) < self.max_samples:
            print(
                f"  Note: Got {len(dataset)} {self._example_unit}, which is less than "
                f"requested {self.max_samples}",
                file=sys.stderr,
            )
            return dataset

        if len(dataset) == self.max_samples:
            return dataset

        print(
            f"  Randomly sampling {self.max_samples} {self._example_unit} from "
            f"{len(dataset)} available {self._example_unit}",
            file=sys.stderr,
        )
        indices = random.sample(range(len(dataset)), self.max_samples)
        return dataset.select(sorted(indices))

    def build(self, deps) -> DatasetDict:
        """Download, reshape, filter, and subsample the corpus.

        Args:
            deps: Unused; this source has no dependencies.

        Returns:
            A `DatasetDict` with a single `train` split carrying a `text` column.
        """
        print(f"Downloading and preparing HuggingFace dataset: {self.hf_name}", file=sys.stderr)
        if self.hf_config:
            print(f"  Config: {self.hf_config}", file=sys.stderr)
        print(f"  Split: {self.split}", file=sys.stderr)
        if self.max_samples:
            print(f"  Max samples ({self._example_unit}): {self.max_samples}", file=sys.stderr)
            print(f"  Oversampling factor: {self.oversampling_factor}x", file=sys.stderr)

        dataset = self._download()

        # min_words_per_line is applied separately below so the filtering is
        # reported; passing it here would drop the examples silently
        dataset = docs_to_filtered_lines(
            dataset,
            self.text_column,
            min_words_per_line=None,
            split_into_lines=self.split_into_lines,
        )
        if self.split_into_lines:
            print(f"  Converted to {len(dataset)} lines from documents", file=sys.stderr)
        else:
            print(f"  Kept {len(dataset)} documents (no line splitting)", file=sys.stderr)

        dataset = self._filter_short_examples(dataset)
        dataset = self._subsample(dataset)

        return DatasetDict({'train': dataset})


SOURCE_TYPES.register(HuggingFaceDataset)
