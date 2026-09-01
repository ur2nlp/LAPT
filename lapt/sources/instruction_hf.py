"""Instruction-tuning source read from a chat-formatted HuggingFace dataset."""

import random
import sys

from datasets import Dataset, DatasetDict, load_dataset

from lapt.sources.base import SOURCE_TYPES, SourceDataset
from lapt.sources.factory import field


class InstructionHFDataset(SourceDataset):
    """An instruction corpus drawn from a chat-formatted HuggingFace dataset.

    Expects a column of message lists in the OpenAI Chat Completions
    convention used by no_robots, Tulu, UltraChat and OpenHermes. Only
    examples shaped as exactly `[user, assistant]` are kept — multi-turn and
    system-prompted examples are dropped to match the single-turn
    prompt/response schema the rest of the pipeline assumes.

    Each kept example is flattened to `prompt` and `response` columns, so
    downstream tokenization treats it identically to an `instruction_jsonl`
    source.
    """

    type_name = "instruction_hf"

    def __init__(
        self,
        cache_dir: str,
        name: str,
        config: str | None = None,
        split: str = 'train',
        messages_column: str = 'messages',
        prompt_template: str = '{user} Response:',
        response_template: str = ' {assistant}',
        max_samples: int | None = None,
        seed: int = 1,
    ):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory is created in.
            name: HuggingFace dataset name, e.g. `HuggingFaceH4/no_robots`.
            config: Dataset configuration or subset.
            split: Which split to load.
            messages_column: Column holding the list of `{role, content}` dicts.
            prompt_template: Format string with a `{user}` placeholder,
                producing the `prompt` column.
            response_template: Format string with an `{assistant}` placeholder,
                producing the `response` column.
            max_samples: Cap on the number of examples, applied as a random
                subsample after filtering.
            seed: Seed for that subsample. Recorded only when `max_samples` is
                set, since it has no effect otherwise.
        """
        super().__init__(cache_dir)
        self.hf_name = name
        self.hf_config = config
        self.split = split
        self.messages_column = messages_column
        self.prompt_template = prompt_template
        self.response_template = response_template
        self.max_samples = max_samples
        self.seed = seed

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        As with `HuggingFaceDataset`, the seed appears only when `max_samples`
        is set; without a cap every surviving example is kept, in order.
        """
        tracked = {
            'type': 'instruction_hf',
            'name': self.hf_name,
            'config': self.hf_config,
            'split': self.split,
            'messages_column': self.messages_column,
            'prompt_template': self.prompt_template,
            'response_template': self.response_template,
            'max_samples': self.max_samples,
        }
        if self.max_samples is not None:
            tracked['seed'] = self.seed
        return tracked

    def _is_single_turn_pair(self, example: dict) -> bool:
        """Whether an example is exactly one user turn and one assistant turn."""
        messages = example[self.messages_column]
        return (
            len(messages) == 2
            and messages[0]['role'] == 'user'
            and messages[1]['role'] == 'assistant'
        )

    def _to_prompt_response(self, example: dict) -> dict:
        """Flatten one message pair into prompt and response columns."""
        user = example[self.messages_column][0]['content']
        assistant = example[self.messages_column][1]['content']
        return {
            'prompt': self.prompt_template.format(user=user),
            'response': self.response_template.format(assistant=assistant),
        }

    def _subsample(self, dataset: Dataset) -> Dataset:
        """Reduce the dataset to `max_samples` examples, at random.

        Selected indices are sorted, so the result keeps the corpus's own order.

        Args:
            dataset: The filtered dataset.

        Returns:
            The subsampled dataset, or the original if no cap applies.
        """
        if self.max_samples is None or len(dataset) <= self.max_samples:
            return dataset

        print(
            f"  Subsampling to {self.max_samples} examples from {len(dataset)}",
            file=sys.stderr,
        )
        indices = random.sample(range(len(dataset)), self.max_samples)
        return dataset.select(sorted(indices))

    def build(self, deps) -> DatasetDict:
        """Download, filter to single-turn pairs, subsample, and flatten.

        Args:
            deps: Unused; this source has no dependencies.

        Returns:
            A `DatasetDict` with a single `train` split carrying `prompt` and
            `response` columns.

        Raises:
            ValueError: If no example survives the single-turn filter, which
                usually means `messages_column` names the wrong column.
        """
        print(
            f"Downloading instruction dataset from HuggingFace: {self.hf_name}",
            file=sys.stderr,
        )
        if self.hf_config:
            print(f"  Config: {self.hf_config}", file=sys.stderr)
        print(f"  Split: {self.split}", file=sys.stderr)

        raw = load_dataset(self.hf_name, self.hf_config, split=self.split)

        count_before = len(raw)
        raw = raw.filter(self._is_single_turn_pair)
        count_after = len(raw)
        print(
            f"  Kept {count_after}/{count_before} examples after filtering to single-turn "
            f"[user, assistant] pairs",
            file=sys.stderr,
        )

        if count_after == 0:
            raise ValueError(
                f"No examples remained after filtering. Check that "
                f"'{self.messages_column}' contains "
                f"[{{'role': 'user', ...}}, {{'role': 'assistant', ...}}] pairs."
            )

        raw = self._subsample(raw)

        return DatasetDict({
            'train': raw.map(self._to_prompt_response, remove_columns=raw.column_names),
        })

    @classmethod
    def from_config(cls, cache_dir: str, source_config, seed: int = 1) -> 'InstructionHFDataset':
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            source_config: Entry carrying at least `name`.
            seed: Global random seed, recorded when `max_samples` is set.

        Returns:
            The configured source.
        """
        return cls(
            cache_dir,
            field(source_config, 'name'),
            config=field(source_config, 'config'),
            split=field(source_config, 'split', 'train'),
            messages_column=field(source_config, 'messages_column', 'messages'),
            prompt_template=field(source_config, 'prompt_template', '{user} Response:'),
            response_template=field(source_config, 'response_template', ' {assistant}'),
            max_samples=field(source_config, 'max_samples'),
            seed=seed,
        )

SOURCE_TYPES.register(InstructionHFDataset)
