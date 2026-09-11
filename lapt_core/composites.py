"""Composite dataset artifacts: several sources combined into one.

Concatenating corpora, and sampling among them with temperature scaling, need
only that a source is a collection of examples -- not that an example is a line
of text. A multilingual speech corpus mixes exactly the way a multilingual text
corpus does, so these live here rather than in any one project.

These are the only artifacts whose dependencies are dynamic. The child set is
read off `sources` at resolve time rather than declared in `depends_on`, so
children are constructed and resolved inside `build()` rather than injected by
an `ArtifactGraph`. The pipeline graph and the source tree are different
structures that meet at one node; "everything is an ArtifactGraph" is the
tempting wrong turn.

A composite cannot know how a project turns a configuration entry into a child
-- that is the project's registry, its type names, and any wrapper it applies --
so it is handed a `child_factory` instead. The seam is deliberately narrow:

    (cache_dir, source_config, seed) -> DatasetArtifact

`seed` is included because a child may subsample and must not silently fall
back to global RNG state. A dev-split size is deliberately *excluded*: a
composite's children must not hold out their own dev sets, because the mix does
that itself, before upsampling, so that a repeated training example cannot also
appear in dev.

`sources` arrive as plain dicts. Unwrapping a configuration object -- a Hydra
`DictConfig`, say -- is the caller's job, done once at the boundary, which is
what keeps `omegaconf` out of this module.
"""

import os
import sys
from collections.abc import Callable

from datasets import DatasetDict, concatenate_datasets

from lapt_core.dataset_artifacts import DatasetArtifact
from lapt_core.mixing import (
    compute_sampling_probs,
    exhaust_first_sample,
    field,
    mix_slug,
    source_id,
)

ChildFactory = Callable[[str, dict, int], DatasetArtifact]

SKIP_DEV_SPLIT = -1


class ConcatArtifact(DatasetArtifact):
    """Several sources concatenated into one training split.

    Children are resolved through the same registry as any other source, so a
    child may itself be composite. Each child caches under its own
    subdirectory of this source's cache directory, keyed by its id — which is
    what lets two mixes referencing the same source id share one copy of it.

    The child set is read from the configuration at build time rather than
    declared statically, so these dependencies are deliberately resolved inside
    `build` rather than through an `ArtifactGraph`.
    """

    def __init__(
        self,
        cache_dir: str,
        sources: list[dict],
        *,
        child_factory: ChildFactory,
        parent_id: str | None = None,
        seed: int = 1,
    ):
        """Initialize the composite.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in, and
                the parent of each child's own cache directory.
            sources: Configuration entries, as plain dicts.
            child_factory: Turns `(cache_dir, source_config, seed)` into an
                unresolved child artifact. See the module docstring.
            parent_id: This source's own id, used to name unnamed children.
            seed: Global random seed, passed to children that subsample.

        Raises:
            ValueError: If `sources` is empty.
        """
        super().__init__(cache_dir)
        if not sources:
            raise ValueError("Cannot concatenate datasets: sources list is empty")
        self.sources = sources
        self.child_factory = child_factory
        self.parent_id = parent_id
        self.seed = seed

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        The children's full configurations are recorded, so a change anywhere
        in the tree invalidates this cache. The seed is deliberately absent:
        concatenation preserves order and samples nothing, and any child that
        does sample records the seed itself.
        """
        return {
            'type': 'concat',
            'sources': [dict(source) for source in self.sources],
        }

    def children(self) -> list[tuple[str, DatasetArtifact]]:
        """Build the child sources, unresolved, paired with their ids.

        Returns:
            A list of `(source_id, source)` in configuration order.
        """
        built = []
        for index, source_config in enumerate(self.sources):
            default_id = f"{self.parent_id}_{index}" if self.parent_id else f"source_{index}"
            child_id = source_id(source_config, fallback=default_id)
            child = self.child_factory(
                os.path.join(self.root, child_id), source_config, self.seed
            )
            built.append((child_id, child))
        return built

    def build(self, deps) -> DatasetDict:
        """Resolve each child and concatenate their training splits.

        Args:
            deps: Unused; children are resolved here rather than injected,
                since the child set is only known from the configuration.

        Returns:
            A `DatasetDict` with a single `train` split.
        """
        print(f"Concatenating {len(self.sources)} dataset sources", file=sys.stderr)

        to_concat = []
        for index, (child_id, child) in enumerate(self.children()):
            child_data = child.resolve()
            to_concat.append(child_data['train'])
            print(
                f"  Source {index} ({child_id}): {len(child_data['train'])} examples",
                file=sys.stderr,
            )

        concatenated = concatenate_datasets(to_concat)
        print(f"  Concatenated to {len(concatenated)} total examples", file=sys.stderr)
        return DatasetDict({'train': concatenated})


class MultinomialArtifact(DatasetArtifact):
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

    def __init__(
        self,
        cache_dir: str,
        sources: list,
        alpha: float | None,
        total_samples: int,
        dev_size: float,
        *,
        child_factory: ChildFactory,
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
            child_factory: Turns `(cache_dir, source_config, seed)` into an
                unresolved child artifact. See the module docstring.
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
        self.child_factory = child_factory
        self.alpha = alpha
        self.total_samples = total_samples
        self.dev_size = dev_size
        self.seed = seed

    @property
    def skip_dev_split(self) -> bool:
        """Whether the global setting disables dev splits entirely."""
        return self.dev_size == SKIP_DEV_SPLIT

    def _normalized_sources(self) -> list[dict]:
        """Return the source configurations as plain, fully-resolved dicts."""
        return [dict(source) for source in self.sources]

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
        return os.path.join(self.root, mix_slug(self.config()))

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
        child = self.child_factory(
            os.path.join(self.root, child_id), source_config, self.seed
        )
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
            split = full_data.train_test_split(test_size=source_dev_size, seed=self.seed)
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

        train = concatenate_datasets(selected).shuffle(seed=self.seed)

        print(f"  Train: {len(train)} examples (upsampled)", file=sys.stderr)
        if dev_splits:
            print(
                f"  Dev splits: {', '.join(dev_splits)} "
                f"({sum(len(d) for d in dev_splits.values())} examples total, "
                "natural proportions)",
                file=sys.stderr,
            )

        return DatasetDict({'train': train, **dev_splits})
