"""Composite source that samples several sources into one temperature-scaled mix."""

import os
import sys

from datasets import DatasetDict, concatenate_datasets
from omegaconf import DictConfig, OmegaConf

from lapt.artifact_configs import multinomial_mix_slug
from lapt.sources.base import SOURCE_TYPES, SourceDataset
from lapt.sources.concat import source_id
from lapt.sources.factory import build_source, field
from lapt.sources.sampling import compute_sampling_probs, exhaust_first_sample

SKIP_DEV_SPLIT = -1


class MultinomialDataset(SourceDataset):
    """Several sources sampled into one mix, with per-source dev splits.

    Each source is split into train and dev *before* upsampling, so a repeated
    training example cannot also appear in dev. Train splits are then sampled to
    a target count according to alpha-weighted probabilities; dev splits are
    kept whole, at their natural proportions.

    Unlike the other sources, this one caches under a mix-keyed subdirectory
    rather than directly under `cache_dir` — see `path`. That is what lets a
    sweep over alpha or sample count keep its previous mixes, while the
    per-source caches stay shared at the parent level.
    """

    type_name = "multinomial"

    def __init__(
        self,
        cache_dir: str,
        sources: list,
        alpha: float | None,
        total_samples: int,
        dev_size: float,
        seed: int = 1,
    ):
        """Initialize the mix.

        Args:
            cache_dir: Parent directory holding both the mix subdirectory and
                the shared per-source caches.
            sources: Configuration entries for the sources to sample from, each
                optionally carrying `sampling_prob`, `upsampling_factor`, or a
                `dev_size` override.
            alpha: Temperature for reweighting unpinned sources; below 1
                upsamples the smaller ones. Optional when it cannot affect the
                result.
            total_samples: Target size of the training split, dev excluded.
            dev_size: Default fraction of each source held out, or -1 to skip.
            seed: Global random seed, which selects the sampled examples.

        Raises:
            ValueError: On an empty source list, a non-positive `total_samples`
                or `alpha`, or a `dev_size` that is neither a fraction nor -1.
        """
        super().__init__(cache_dir)
        if not sources:
            raise ValueError("Cannot sample from datasets: sources list is empty")
        if total_samples <= 0:
            raise ValueError(f"total_samples must be positive, got {total_samples}")
        if alpha is not None and alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if dev_size is None:
            raise ValueError("dev_size must be provided for multinomial sampling")
        if dev_size == 0:
            raise ValueError(
                "dev_size=0 is ambiguous. Use dev_size=-1 to explicitly skip dev split, "
                "or use a value > 0 for fractional split size."
            )
        if dev_size != SKIP_DEV_SPLIT and not (0 < dev_size < 1):
            raise ValueError(
                f"Multinomial sampling requires fractional dev_size (0 < dev_size < 1), "
                f"got {dev_size}. Use dev_size=-1 to skip dev split (e.g., when using "
                "external dev sets). Fixed-size dev sets are not supported for "
                "multinomial sampling."
            )

        self.sources = sources
        self.alpha = alpha
        self.total_samples = total_samples
        self.dev_size = dev_size
        self.seed = seed

    @classmethod
    def from_config(cls, cache_dir: str, source_config, seed: int = 1) -> 'MultinomialDataset':
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Parent directory for the mix and the source caches.
            source_config: Entry carrying `sources`, `total_samples`, and
                `dev_size`, optionally `alpha`.
            seed: Global random seed.

        Returns:
            The configured mix.
        """
        return cls(
            cache_dir,
            field(source_config, 'sources'),
            field(source_config, 'alpha'),
            field(source_config, 'total_samples'),
            field(source_config, 'dev_size'),
            seed=seed,
        )

    @property
    def skip_dev_split(self) -> bool:
        """Whether the global setting disables dev splits entirely."""
        return self.dev_size == SKIP_DEV_SPLIT

    def _normalized_sources(self) -> list[dict]:
        """Return the source configurations as plain, fully-resolved dicts."""
        return [
            OmegaConf.to_container(DictConfig(source), resolve=True)
            for source in self.sources
        ]

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        The seed appears unconditionally, unlike in the path, because it does
        change which examples are sampled. `path` omits it at the default only
        so that directories written before seed-keying stay addressed.
        """
        return {
            'type': 'multinomial',
            'sources': self._normalized_sources(),
            'alpha': self.alpha,
            'total_samples': self.total_samples,
            'dev_size': self.dev_size,
            'seed': self.seed,
        }

    @property
    def mix_dir(self) -> str:
        """Directory holding this mix and everything else keyed on it.

        Tokenized dev splits, the training plan, and sampled subsets live
        beside `untokenized` here, all keyed on the same mix.
        """
        return os.path.join(self.root, multinomial_mix_slug(self.config()))

    @property
    def path(self) -> str:
        """Cache directory, nested inside the mix directory.

        Overridden rather than using `path_includes_digest`, which would append
        the digest to this artifact's own name and so orphan the sibling
        artifacts keyed on the same mix.
        """
        return os.path.join(self.mix_dir, self.name)

    def _split_source(self, index: int, source_config) -> tuple[str, object, object]:
        """Resolve one source and hold out its dev split.

        Splitting before upsampling is what keeps a repeated training example
        out of dev.

        Args:
            index: Position in the source list, for naming and messages.
            source_config: The source's configuration entry.

        Returns:
            A tuple of (source id, train split, dev split or None).

        Raises:
            ValueError: On a per-source `dev_size` of 0 or a negative value
                other than -1.
        """
        child_id = source_id(source_config, fallback=f"source_{index}")
        child = build_source(os.path.join(self.root, child_id), source_config, self.seed)
        full_data = child.resolve()['train']

        source_dev_size = field(source_config, 'dev_size', self.dev_size)
        if source_dev_size == 0:
            raise ValueError(
                f"Source {index}: dev_size=0 is ambiguous. "
                "Use dev_size=-1 to explicitly skip dev split."
            )
        if source_dev_size != SKIP_DEV_SPLIT and source_dev_size < 0:
            raise ValueError(
                f"Source {index}: dev_size must be positive or -1 to skip, "
                f"got {source_dev_size}."
            )

        if source_dev_size == SKIP_DEV_SPLIT:
            train_data, dev_data = full_data, None
        else:
            split = full_data.train_test_split(test_size=source_dev_size, seed=1)
            train_data, dev_data = split['train'], split['test']

        has_override = field(source_config, 'dev_size') is not None
        label = (
            f"dev_size={source_dev_size}" if has_override
            else f"global dev_size={source_dev_size}"
        )
        if dev_data is not None:
            print(
                f"  Source {index} ({child_id}): {len(train_data)} train, "
                f"{len(dev_data)} dev examples ({label})",
                file=sys.stderr,
            )
        else:
            print(
                f"  Source {index} ({child_id}): {len(train_data)} examples "
                f"(no dev split, {label})",
                file=sys.stderr,
            )

        return child_id, train_data, dev_data

    def _samples_per_source(self, train_sizes: list[int]) -> list[int]:
        """Turn sampling probabilities into integer sample counts.

        Args:
            train_sizes: Size of each source's training split.

        Returns:
            One count per source, summing to `total_samples`.
        """
        probs = compute_sampling_probs(self.sources, train_sizes, self.alpha)
        counts = [int(prob * self.total_samples) for prob in probs]

        # hand out the rounding remainder one at a time so the counts sum exactly
        for index in range(self.total_samples - sum(counts)):
            counts[index % len(self.sources)] += 1

        print("Train sampling distribution:", file=sys.stderr)
        for index, count in enumerate(counts):
            percentage = 100 * count / self.total_samples
            pinned = field(self.sources[index], 'sampling_prob') is not None
            print(
                f"  {source_id(self.sources[index], fallback=f'source_{index}')}: "
                f"{count} samples ({percentage:.2f}%){' (pinned)' if pinned else ''}",
                file=sys.stderr,
            )
        return counts

    def build(self, deps) -> DatasetDict:
        """Split every source, sample the training data, and assemble the mix.

        Args:
            deps: Unused; sources are resolved here, since the set of them is
                only known from the configuration.

        Returns:
            A `DatasetDict` with a `train` split and one dev split per source
            that has one, named for that source.

        Raises:
            ValueError: If every source's training split is empty.
        """
        print(
            f"Multinomial sampling from {len(self.sources)} sources with alpha={self.alpha}",
            file=sys.stderr,
        )
        print(f"Mix cache directory: {self.mix_dir}", file=sys.stderr)
        if self.skip_dev_split:
            print("No dev split (dev_size=-1, using all data for training)", file=sys.stderr)
        else:
            print(
                f"Dev split: {self.dev_size:.1%} of each source (before upsampling)",
                file=sys.stderr,
            )

        train_datasets = []
        dev_splits = {}
        for index, source_config in enumerate(self.sources):
            child_id, train_data, dev_data = self._split_source(index, source_config)
            train_datasets.append(train_data)
            if dev_data is not None:
                dev_splits[child_id] = dev_data

        train_sizes = [len(dataset) for dataset in train_datasets]
        if all(size == 0 for size in train_sizes):
            raise ValueError("Cannot sample: all source datasets are empty")

        selected = [
            dataset.select(exhaust_first_sample(len(dataset), count))
            for dataset, count in zip(train_datasets, self._samples_per_source(train_sizes))
        ]

        train = concatenate_datasets(selected).shuffle(seed=1)

        print(f"  Train: {len(train)} examples (upsampled)", file=sys.stderr)
        if dev_splits:
            print(
                f"  Dev splits: {', '.join(dev_splits)} "
                f"({sum(len(d) for d in dev_splits.values())} examples total, "
                "natural proportions)",
                file=sys.stderr,
            )

        return DatasetDict({'train': train, **dev_splits})


SOURCE_TYPES.register(MultinomialDataset)
