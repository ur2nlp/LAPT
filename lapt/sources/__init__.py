"""Concrete untokenized corpus sources, keyed by a dataset config's `type`.

Importing this package registers every source type in `SOURCE_TYPES`, so
`SOURCE_TYPES.create(dataset_type, ...)` resolves any type the configuration
system accepts. Registration is an explicit call in each type's module rather
than a side effect of subclassing, so a name collision is reported where it
happens and test subclasses do not pollute the table.
"""

from lapt.sources.base import SOURCE_TYPES, SourceDataset
from lapt.sources.concat import ConcatDataset
from lapt.sources.factory import build_source
from lapt.sources.huggingface import HuggingFaceDataset
from lapt.sources.instruction_hf import InstructionHFDataset
from lapt.sources.instruction_jsonl import InstructionJsonlDataset
from lapt.sources.multinomial import MultinomialDataset
from lapt.sources.oscar import OscarDataset
from lapt.sources.plaintext import PlaintextDataset

__all__ = [
    'SOURCE_TYPES',
    'ConcatDataset',
    'HuggingFaceDataset',
    'InstructionHFDataset',
    'InstructionJsonlDataset',
    'MultinomialDataset',
    'OscarDataset',
    'PlaintextDataset',
    'SourceDataset',
    'build_source',
]
