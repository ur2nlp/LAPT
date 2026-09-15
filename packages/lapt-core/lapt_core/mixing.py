"""Combining several dataset sources into one, independent of what they hold.

A pipeline that trains on more than one corpus needs to answer the same three
questions regardless of domain: how does a configuration entry name a source,
what share of the training budget does each source get, and where does the
resulting mix cache. None of those depend on whether an example is a line of
text or a second of audio -- they need only that a source is a collection of
examples -- so they live here rather than in any one project.

Deliberately stdlib-only, like `lapt_core.artifacts`: nothing here loads a
dataset, so nothing here should make importing it expensive. The composite
artifacts that *do* load datasets belong beside `DatasetArtifact`.

Also deliberately free of `omegaconf`. `field` reads plain dicts and attribute
objects alike, which covers a Hydra `DictConfig` without importing one;
projects that want a `DictConfig` fully normalized should convert it at the
boundary, where the configuration arrives, rather than repeatedly in here.
"""

import random
import sys
from typing import Any

from lapt_core.artifacts import config_digest, format_number

DEFAULT_SEED = 1


def field(source_config: Any, name: str, default: Any = None) -> Any:
    """Read one field from a source configuration.

    Accepts both mappings and attribute-style objects, since sources arrive as
    either depending on whether they came from a config framework or from a
    parent's `sources` list.

    Args:
        source_config: The configuration entry.
        name: Field to read.
        default: Value to return when the field is absent.

    Returns:
        The field's value, or `default`.
    """
    if isinstance(source_config, dict):
        value = source_config.get(name, default)
    else:
        value = getattr(source_config, name, default)
    return default if value is None and default is not None else value


def source_id(source_config: Any, fallback: str | None = None) -> str:
    """Return a source's cache identifier.

    Checks `id`, then the deprecated `language`, then the fallback.

    Args:
        source_config: The configuration entry.
        fallback: Identifier to use when the entry names none.

    Returns:
        The identifier.
    """
    identifier = field(source_config, 'id')
    if not identifier:
        identifier = field(source_config, 'language')
        if identifier:
            print(
                f"Warning: 'language' field for source identification is deprecated, "
                f"use 'id' instead (found language='{identifier}')",
                file=sys.stderr,
            )
    return identifier or fallback


def compute_sampling_probs(
    sources: list,
    train_sizes: list[int],
    alpha: float | None,
) -> list[float]:
    """
    Compute per-source sampling probabilities, respecting pinned sampling_prob values.

    Sources with an explicit `sampling_prob` or `upsampling_factor field get that probability
    directly. The remaining probability budget is distributed among unpinned sources using
    alpha-based temperature scaling: p_i = (size_i)^alpha / Z, scaled to fill the budget.

    Alpha may be None when it has nothing to do: when every source is pinned, or when
    exactly one source is unpinned and therefore takes the whole remaining budget
    regardless of the exponent. It is required whenever two or more sources are unpinned.

    Args:
        sources: List of source config entries (may carry 'sampling_prob')
        train_sizes: Number of training examples per source (after dev split)
        alpha: Temperature parameter for unpinned source reweighting. May be None only
            if it cannot affect the result.

    Returns:
        List of sampling probabilities (one per source, sums to 1.0)
    """
    num_sources = len(sources)
    total_size = sum(train_sizes)
    pinned_probs = {}
    for idx, source in enumerate(sources):
        prob = field(source, 'sampling_prob')
        upsampling_factor = field(source, 'upsampling_factor')
        if upsampling_factor is not None and prob is None:
            pinned_probs[idx] = train_sizes[idx] * upsampling_factor / total_size
        if prob is not None:
            if prob <= 0 or prob >= 1.0:
                source_name = source_id(source, fallback=f"source_{idx}")
                raise ValueError(
                    f"Source '{source_name}': sampling_prob must be between 0 and 1 exclusive, "
                    f"got {prob}"
                )
            pinned_probs[idx] = prob

    pinned_total = sum(pinned_probs.values())

    # If every source is pinned, they must sum to exactly 1.0
    if len(pinned_probs) == num_sources:
        if abs(pinned_total - 1.0) > 1e-9:
            raise ValueError(
                f"All sources have sampling_prob but they sum to {pinned_total:.6f}, not 1.0"
            )
        return [pinned_probs[i] for i in range(num_sources)]

    # With unpinned sources present, pinned probs must leave room for them
    if pinned_total >= 1.0:
        raise ValueError(
            f"Sum of pinned sampling_prob values is {pinned_total:.4f}, "
            "must be less than 1.0 to leave budget for remaining sources"
        )

    # Distribute remaining budget among unpinned sources using alpha-based weighting
    remaining_budget = 1.0 - pinned_total
    unpinned_indices = [i for i in range(num_sources) if i not in pinned_probs]

    unpinned_sizes = [train_sizes[i] for i in unpinned_indices]
    if all(s == 0 for s in unpinned_sizes):
        raise ValueError("Cannot compute sampling probabilities: all unpinned sources are empty")

    # a lone unpinned source takes the whole remaining budget: its weight normalizes
    # to 1.0 for any exponent, so alpha is not needed to resolve the mixture
    if len(unpinned_indices) == 1:
        lone_probs = dict(pinned_probs)
        lone_probs[unpinned_indices[0]] = remaining_budget
        return [lone_probs[i] for i in range(num_sources)]

    if alpha is None:
        unpinned_ids = [
            source_id(sources[i], fallback=f"source_{i}")
            for i in unpinned_indices
        ]
        raise ValueError(
            f"alpha is required when two or more sources are unpinned "
            f"({', '.join(unpinned_ids)}): it sets how the remaining probability "
            "budget is split between them"
        )

    weights = [size ** alpha for size in unpinned_sizes]
    total_weight = sum(weights)
    unpinned_probs = {
        idx: (weights[j] / total_weight) * remaining_budget
        for j, idx in enumerate(unpinned_indices)
    }

    return [pinned_probs.get(i, unpinned_probs.get(i)) for i in range(num_sources)]


def exhaust_first_sample(dataset_size: int, num_samples: int) -> list[int]:
    """
    Generate sample indices using exhaust-first strategy.

    When num_samples > dataset_size, includes ALL examples once before any
    duplication. This maximizes coverage of unique examples, which is critical
    for low-resource datasets.

    Args:
        dataset_size: Number of examples in the dataset
        num_samples: Number of samples to draw

    Returns:
        List of indices (may contain duplicates if num_samples > dataset_size)
    """
    if num_samples <= dataset_size:
        # Sample without replacement
        return random.sample(range(dataset_size), num_samples)
    else:
        # Include ALL examples once, then sample remainder with replacement
        all_indices = list(range(dataset_size))
        num_additional = num_samples - dataset_size
        additional_indices = random.choices(range(dataset_size), k=num_additional)
        indices = all_indices + additional_indices
        random.shuffle(indices)  # Shuffle to mix exhaustive + repeated samples
        return indices


def mix_slug(dataset_config: dict) -> str:
    """
    Build a deterministic subdirectory name for a sampled mix of sources.

    The upsampled training split produced by multinomial sampling depends on
    alpha, total_samples, dev_size, per-source sampling_prob /
    upsampling_factor / dev_size overrides, and the seed, but NOT on the
    underlying source datasets (which live in parent-level subdirectories and
    can be shared across mixes). Caching mix-dependent artifacts inside {cache_dir}/{slug}/
    instead of directly under {cache_dir}/ means sweeping alpha or sample
    counts no longer clobbers the previous mix and source caches are
    transparently shared.

    Args:
        dataset_config: Dict with at least 'total_samples' and 'sources'.
            'alpha' is optional, since it is omissible for mixes where it
            cannot affect the sampling probabilities. 'seed' is optional and
            defaults to DEFAULT_SEED.

    Returns:
        Slug like "mix_a0.5_s5m_ab12cd34", or "mix_s5m_ab12cd34" without alpha.
    """
    alpha = dataset_config.get('alpha')
    total_samples = dataset_config['total_samples']

    mix_keys = {
        'alpha': alpha,
        'total_samples': total_samples,
        'dev_size': dataset_config.get('dev_size'),
        'sources': [
            {
                'id': source.get('id') or source.get('language'),
                'sampling_prob': source.get('sampling_prob'),
                'upsampling_factor': source.get('upsampling_factor'),
                'dev_size': source.get('dev_size'),
                'substitutions': source.get('substitutions'),
            }
            for source in dataset_config.get('sources', [])
        ],
    }
    # a non-default seed changes which examples are sampled and repeated, so
    # mixes that differ only by seed must not share a directory. the key is
    # omitted at the default so that slugs predating seed-keying are unchanged
    # -- every mix built before this was built at DEFAULT_SEED, so the omission
    # records a fact rather than papering over one. the seed is recorded in the
    # config unconditionally either way, so validation is unaffected.
    seed = dataset_config.get('seed', DEFAULT_SEED)
    if seed != DEFAULT_SEED:
        mix_keys['seed'] = seed

    digest = config_digest(mix_keys)

    # omit the alpha segment when the config has no alpha, rather than writing
    # "aNone" into the directory name; slugs for configs that do set it are unchanged
    alpha_part = f"a{alpha}_" if alpha is not None else ""
    return f"mix_{alpha_part}s{format_number(total_samples)}_{digest}"
