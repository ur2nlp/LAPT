"""Regex substitution applied to an already-resolved source.

Substitutions are type-agnostic: they run over every string column of whatever
a source produced, so one pattern list works for a plaintext corpus, a
HuggingFace download, or an instruction set with `prompt` and `response`
columns. That is why this is a wrapper rather than a registered source type —
it composes with any of them, and appears in a config as a `substitutions`
field rather than as a `type`.
"""

import os
import re
import sys

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, ListConfig, OmegaConf

from lapt.sources.base import SourceDataset
from lapt_core.artifacts import config_digest


def parse_substitutions(raw) -> list[tuple[str, str]]:
    """Normalize a config's `substitutions` field into (pattern, replacement) pairs.

    Args:
        raw: The raw value from the dataset config: a list of
            `{pattern, replacement}` mappings, a `ListConfig`, or None.
            `replacement` defaults to the empty string.

    Returns:
        Pairs in declaration order, empty when nothing is configured.

    Raises:
        ValueError: If an entry omits `pattern`.
        re.error: If a pattern will not compile. Raised here so a typo fails
            immediately rather than part-way through a dataset map.
    """
    if not raw:
        return []
    if isinstance(raw, (ListConfig, DictConfig)):
        raw = OmegaConf.to_container(raw, resolve=True)

    substitutions = []
    for item in raw:
        if 'pattern' not in item:
            raise ValueError(f"Each substitution must specify a 'pattern' (got {item!r}).")
        pattern = item['pattern']
        replacement = item.get('replacement', '')
        re.compile(pattern)
        substitutions.append((pattern, replacement))
    return substitutions


class SubstitutedDataset(SourceDataset):
    """A source with regex substitutions applied to every string column.

    Caches beside the source it wraps, as `{base}_sub_{digest}`, so the raw
    corpus survives and several substitution sets over one source can coexist.

    Deliberately not registered in `SOURCE_TYPES`: this is not a dataset type a
    config can name, it is a transformation any source may carry.
    """

    def __init__(self, base: SourceDataset, substitutions: list[tuple[str, str]]):
        """Initialize the wrapper.

        Args:
            base: The source to transform, unresolved.
            substitutions: Ordered (pattern, replacement) pairs.

        Raises:
            ValueError: If `substitutions` is empty, since the wrapper would
                then cache a second identical copy of the base.
        """
        super().__init__(os.path.dirname(base.path))
        if not substitutions:
            raise ValueError(
                "SubstitutedDataset requires at least one substitution; "
                "use the base source directly when there are none"
            )
        self.base = base
        self.substitutions = substitutions

    def _normalized(self) -> list[dict]:
        """Return the substitutions in their recorded mapping form."""
        return [
            {'pattern': pattern, 'replacement': replacement}
            for pattern, replacement in self.substitutions
        ]

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        The seed is deliberately absent: substitution is deterministic.
        """
        return {
            'type': 'substituted',
            'base': os.path.basename(self.base.path),
            'substitutions': self._normalized(),
        }

    @property
    def path(self) -> str:
        """Cache directory, a sibling of the base named for the substitutions.

        Keyed on the substitutions alone rather than the whole config, because
        the other two tracked fields cannot vary independently of the prefix:
        `base` is the prefix, and `type` is constant. So the path and the record
        beside it still cannot describe different things.

        Not a free choice either way — this digest is embedded in the names of
        the tokenized caches derived from it, so changing what it covers would
        orphan them.
        """
        return f"{self.base.path}_sub_{config_digest(self._normalized())}"

    def _substitute_split(self, split: Dataset) -> Dataset:
        """Apply every substitution to each string column of one split.

        Args:
            split: The split to transform.

        Returns:
            The transformed split.
        """
        string_columns = [
            name
            for name, feature in split.features.items()
            if getattr(feature, 'dtype', None) == 'string'
        ]
        compiled = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.substitutions
        ]

        def substitute_batch(examples):
            for column in string_columns:
                substituted = []
                for value in examples[column]:
                    for pattern, replacement in compiled:
                        value = pattern.sub(replacement, value)
                    substituted.append(value)
                examples[column] = substituted
            return examples

        return split.map(substitute_batch, batched=True)

    def build(self, deps) -> DatasetDict:
        """Resolve the base source and transform every split of it.

        Args:
            deps: Unused; the base is held directly rather than injected,
                since it is chosen by the configuration this wraps.

        Returns:
            The transformed dataset, with the same splits and columns.
        """
        print(
            f"Applying {len(self.substitutions)} regex substitution(s) to {self.base.path}",
            file=sys.stderr,
        )
        base_data = self.base.resolve()
        return DatasetDict({
            split_name: self._substitute_split(split)
            for split_name, split in base_data.items()
        })
