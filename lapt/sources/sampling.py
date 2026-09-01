"""Sampling arithmetic for multinomial dataset mixes.

Separated from the source class so the probability and index computations can
be tested directly, without building a dataset to exercise them.
"""

import random

from omegaconf import DictConfig

from lapt.sources.concat import source_id


def compute_sampling_probs(
    sources: list[dict],
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
        sources: List of source config dicts (may contain 'sampling_prob' field)
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
        prob = source.get('sampling_prob')
        upsampling_factor = source.get('upsampling_factor')
        if upsampling_factor is not None and prob is None:
            pinned_probs[idx] = train_sizes[idx] * upsampling_factor / total_size
        if prob is not None:
            if prob <= 0 or prob >= 1.0:
                source_name = source_id(DictConfig(source), fallback=f"source_{idx}")
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
            source_id(DictConfig(sources[i]), fallback=f"source_{i}")
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

    Helper for _load_multinomial_dataset. When num_samples > dataset_size,
    includes ALL examples once before any duplication. This maximizes coverage
    of unique examples, which is critical for low-resource datasets.

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
