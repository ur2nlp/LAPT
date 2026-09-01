"""OSCAR corpus source, split from documents into lines."""

import sys

from datasets import DatasetDict, load_dataset

from lapt.sources.base import SOURCE_TYPES, SourceDataset
from lapt.sources.text_processing import docs_to_lines


class OscarDataset(SourceDataset):
    """A corpus drawn from one language of `oscar-corpus/OSCAR-2201`.

    Requires a HuggingFace token, since OSCAR is a gated dataset.
    """

    type_name = "oscar"

    def __init__(self, cache_dir: str, language_code: str):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory is created in.
            language_code: OSCAR language code, e.g. `en`.
        """
        super().__init__(cache_dir)
        self.language_code = language_code

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        The seed is deliberately absent: the whole language split is taken, so
        nothing here is sampled.
        """
        return {'type': 'oscar', 'language': self.language_code}

    def build(self, deps) -> DatasetDict:
        """Download the language split and convert documents to lines.

        Args:
            deps: Unused; this source has no dependencies.

        Returns:
            A `DatasetDict` whose splits carry a single `text` column.
        """
        print("Downloading and preparing OSCAR dataset", file=sys.stderr)
        dataset = load_dataset(
            "oscar-corpus/OSCAR-2201",
            token=True,
            language=self.language_code,
        )
        return dataset.map(
            docs_to_lines,
            batched=True,
            remove_columns=dataset['train'].column_names,  # type: ignore
        )


SOURCE_TYPES.register(OscarDataset)
