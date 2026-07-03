"""Character trigram model for generating plausible Gothic non-words.

The non-word is the training signal for the OOV-robustness augmentation: it must
look like a real, unseen Gothic word so the model learns to hedge on genuine
unfamiliarity rather than on spelling weirdness (see
`.claude/gothic/oov_robustness_augmentation.md` § "load-bearing caveat").

Trigrams (not bigrams) because Gothic romanization is digraph-heavy — ``ai``,
``au``, ``ei``, ``iu``, ``hw``, and clusters around ``þ`` — so second-order
context captures far more of the phonotactics for a negligible cost at this
corpus size.

The model operates on the *roman* form; callers transliterate to Gothic script
via ``gothic.orthography.transliterate_latin_to_gothic``.
"""

import random
from collections.abc import Callable
from collections import defaultdict


# Word-boundary markers. Padding a word with two start markers lets the first
# real character be sampled from a trigram, and the end marker lets word length
# be learned rather than fixed.
START = "^"
END = "$"


class TrigramNonwordModel:
    """A smoothed character-trigram model over romanized Gothic words.

    Attributes:
        temperature: Sampling temperature; lower is more phonotactically typical.
        top_p: Nucleus-sampling cutoff applied after temperature.
        known_words: Roman word forms to reject if regenerated, so every emitted
            non-word is genuinely unseen.
    """

    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 1,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self._rng = random.Random(seed)
        # (char, char) -> {next_char: count}
        self._counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._alphabet: set[str] = set()
        self.known_words: set[str] = set()

    def fit(self, words: list[str]) -> "TrigramNonwordModel":
        """Accumulate trigram counts from a list of romanized words.

        Fitting shapes phonotactics only; it does **not** populate
        ``known_words``. The rejection vocabulary is set separately by the caller,
        because the model is fit on pseudo-*stems* but must reject collisions
        against real *words* (stem + grafted affix). See
        `.claude/gothic/oov_robustness_augmentation.md`.

        Args:
            words: Lowercased romanized forms (stems, in the stem-model pipeline).

        Returns:
            self, to allow chaining.
        """
        for word in words:
            if not word:
                continue
            self._alphabet.update(word)
            padded = START + START + word + END
            for index in range(len(padded) - 2):
                context = (padded[index], padded[index + 1])
                next_char = padded[index + 2]
                self._counts[context][next_char] += 1
        return self

    def _sample_next(self, context: tuple[str, str]) -> str:
        """Sample the next character given the two-character context.

        Falls back to a lower-order (single-character) context, then to a uniform
        draw over the alphabet, so an unseen context never dead-ends.
        """
        distribution = self._counts.get(context)
        if not distribution:
            # back off to contexts sharing the most recent character
            backoff: dict[str, int] = defaultdict(int)
            for (_, second), nexts in self._counts.items():
                if second == context[1]:
                    for char, count in nexts.items():
                        backoff[char] += count
            distribution = backoff or {char: 1 for char in self._alphabet}

        chars = list(distribution.keys())
        # temperature-scaled counts, then nucleus filtering
        weights = [count ** (1.0 / self.temperature) for count in distribution.values()]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]

        ranked = sorted(zip(chars, probabilities), key=lambda pair: pair[1], reverse=True)
        cumulative = 0.0
        nucleus: list[tuple[str, float]] = []
        for char, probability in ranked:
            nucleus.append((char, probability))
            cumulative += probability
            if cumulative >= self.top_p:
                break

        nucleus_chars = [char for char, _ in nucleus]
        nucleus_weights = [probability for _, probability in nucleus]
        return self._rng.choices(nucleus_chars, weights=nucleus_weights, k=1)[0]

    def generate(
        self,
        min_length: int = 3,
        max_length: int = 14,
        prefix: str | None = None,
        suffix: str | None = None,
        accept: Callable[[str], bool] | None = None,
        reject_known: bool = True,
        max_attempts: int = 50,
    ) -> str | None:
        """Generate a single non-word.

        The model generates a stem body; an optional ``prefix`` and ``suffix`` are
        grafted on either side (the stem model itself is trained on affix-stripped
        pseudo-stems, so grafting restores morphology the body lacks).

        Args:
            min_length: Reject candidates shorter than this.
            max_length: A candidate whose body reaches this length *without* the
                model emitting an end-of-word is rejected (rather than truncated
                mid-word, which would create an implausible ending) and resampled.
            prefix: If given, prepend this string to the generated body.
            suffix: If given, append this string to the generated body.
            accept: Optional predicate on the full candidate; a candidate is only
                returned if it passes (used for junction validation when grafting
                an affix). Shares the ``max_attempts`` budget.
            reject_known: Whether to reject candidates present in ``known_words``.
                Set False only when *measuring* the real-word collision rate for
                sampling-parameter tuning.
            max_attempts: How many candidates to try before giving up.

        Returns:
            A romanized non-word (not in ``known_words`` when ``reject_known``) and
            passing ``accept``, or None if no valid candidate was found within
            ``max_attempts``.
        """
        prefix = prefix or ""
        suffix = suffix or ""
        body_max = max_length - len(prefix) - len(suffix)
        for _ in range(max_attempts):
            context = (START, START)
            characters: list[str] = []
            reached_end = False
            while len(characters) < body_max:
                next_char = self._sample_next(context)
                if next_char == END:
                    reached_end = True
                    break
                characters.append(next_char)
                context = (context[1], next_char)

            if not reached_end or not characters:
                continue

            candidate = prefix + "".join(characters) + suffix
            if len(candidate) < min_length:
                continue
            if reject_known and candidate in self.known_words:
                continue
            if accept is not None and not accept(candidate):
                continue
            return candidate
        return None
