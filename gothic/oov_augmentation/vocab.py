"""Build romanized Gothic word vocabularies for the OOV augmentation.

Two vocabularies are needed and they are deliberately different (see
`.claude/gothic/oov_robustness_augmentation.md`):

* **training vocab** — the words the trigram stem model is fit on and the words
  affix scoring runs over. This should be the *train* split only, so the model's
  phonotactics come from data it has actually seen.
* **rejection vocab** — the *train* words a generated non-word must not collide
  with. The hazard is asymmetric: colliding with a **train** word is bad (the
  model knows it, so a hedge target on it teaches a false "I don't recognize
  X" — the over-hedging failure mode). Colliding with a **test** word is benign
  or even useful: it is real but unseen by the trained model, so hedging on it is
  honest, and no holdout meaning is leaked (the hedge is attached inside a
  different training sentence with that sentence's gloss). So the rejection set is
  train-only by default; ``include_test`` exists only for experiments.

Source is the prepared monolingual codices (roman + Gothic-script interleaved).
Only roman lines are read — Gothic-script lines are skipped — so the vocabulary
matches the romanization convention used elsewhere in the pipeline. (Note:
``data/gotica/gotica.txt`` writes hwair as the digraph ``hv`` and carries verse
references, so it is intentionally *not* used here.)

TODO: the Gothic-script line-skipping is a stopgap tied to this file's
interleaved both-scripts layout. Prefer a roman-only monolingual source (or an
explicit script field) so this module doesn't have to infer script per line.
See `.claude/TODO.md`.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Gothic Unicode block; a line containing any of these is a Gothic-script line.
_GOTHIC_SCRIPT = re.compile(r"[\U00010330-\U0001034F]")

# Everything that is not a romanized Gothic letter is stripped from a token.
# Romanization uses ASCII plus thorn and hwair.
_NON_LETTER = re.compile(r"[^a-zþƕ]")

MONOLINGUAL_DIR = Path("data/gothic_prepared")
TRAIN_FILE = MONOLINGUAL_DIR / "monolingual_all-codices_both-scripts_train.txt"
TEST_FILE = MONOLINGUAL_DIR / "monolingual_all-codices_both-scripts_test.txt"


def clean_token(token: str) -> str:
    """Lowercase a token and strip non-letter characters.

    Args:
        token: A whitespace-delimited token from a roman line.

    Returns:
        The cleaned romanized word, possibly empty.
    """
    return _NON_LETTER.sub("", token.lower())


def iter_roman_words(paths: list[Path]):
    """Yield cleaned romanized word forms from the roman lines of given files.

    Gothic-script lines and empty tokens are skipped.

    Args:
        paths: Text files with roman/Gothic-script lines.

    Yields:
        Cleaned romanized word forms, with repeats (so callers can count).
    """
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if _GOTHIC_SCRIPT.search(line):
                    continue
                for token in line.split():
                    word = clean_token(token)
                    if word:
                        yield word


def load_vocab(paths: list[Path]) -> Counter:
    """Return a Counter mapping romanized word type to token frequency.

    Args:
        paths: Text files to read roman words from.

    Returns:
        A Counter of word type -> token count.
    """
    return Counter(iter_roman_words(paths))


def load_training_vocab() -> Counter:
    """Return the train-split vocabulary (for fitting the model / affix scoring)."""
    return load_vocab([TRAIN_FILE])


def load_rejection_vocab(include_test: bool = False) -> Counter:
    """Return the vocabulary used to reject real-word collisions.

    Defaults to the train split only. Pass ``include_test=True`` to also reject
    genuinely-unseen test words (broader, but crosses the train/test line).

    Args:
        include_test: Whether to add the test split to the rejection set.
    """
    paths = [TRAIN_FILE, TEST_FILE] if include_test else [TRAIN_FILE]
    return load_vocab(paths)


def content_vocab(
    vocab: Counter,
    min_length: int = 3,
    top_k: int = 50,
) -> Counter:
    """Drop function-word-like entries from a vocabulary.

    The filter is deliberately statistical (no curated stoplist) to keep the
    recipe portable to languages with no linguistic resources — at the cost of
    occasionally dropping a very frequent *content* word (e.g. ``qaþ``, ``frauja``
    in this Biblical corpus). That loss is negligible to a stem model of thousands
    of types, whereas a surviving function word pollutes it with short, functiony
    phonotactics. See `.claude/gothic/oov_robustness_augmentation.md`.

    Two crude signals, combined:
      * length floor — drop words shorter than ``min_length`` (most Gothic
        function words are 1-2 characters);
      * frequency rank — drop the ``top_k`` most frequent types (catches the
        longer demonstratives/conjunctions the length floor misses).

    Args:
        vocab: A word type -> token count Counter.
        min_length: Keep only words with at least this many characters.
        top_k: Drop the this-many most frequent types.

    Returns:
        A new Counter with function-word-like entries removed.
    """
    most_frequent = {word for word, _ in vocab.most_common(top_k)}
    kept = Counter()
    for word, count in vocab.items():
        if len(word) < min_length:
            continue
        if word in most_frequent:
            continue
        kept[word] = count
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["train", "rejection"],
        default="train",
        help="Which vocabulary to build (both default to the train split).",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="For the rejection split, also include the test split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write sorted word types here (one per line). Default: stdout counts.",
    )
    args = parser.parse_args()

    if args.split == "train":
        vocab = load_training_vocab()
    else:
        vocab = load_rejection_vocab(include_test=args.include_test)
    print(f"{args.split}: {sum(vocab.values())} tokens, {len(vocab)} types", file=sys.stderr)

    if args.output is not None:
        args.output.write_text("\n".join(sorted(vocab)) + "\n")
    else:
        for word, count in vocab.most_common():
            print(f"{count}\t{word}")


if __name__ == "__main__":
    main()
