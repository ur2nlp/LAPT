"""Sampling arithmetic for multinomial dataset mixes.

Now a re-export: the arithmetic is domain-neutral -- it needs only that a
source is a collection of examples -- so it lives in `lapt_core.mixing`
alongside the rest of the mixing vocabulary. This module stays so that
`lapt.sources.sampling` keeps working for existing callers and tests.
"""

from lapt_core.mixing import compute_sampling_probs, exhaust_first_sample

__all__ = ['compute_sampling_probs', 'exhaust_first_sample']
