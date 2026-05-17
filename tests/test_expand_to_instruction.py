"""
Tests for gothic.word_spotting.expand_to_instruction.

Coverage:
- blank_gothic_word: token match, punctuation preservation, no-match
- make_example: each projection's response and prompt content
- expand_entry: script fan-out, cloze skipping, discrimination distractor source
- forward projection backward compatibility
"""

import random

import pytest

from gothic.word_spotting import expand_to_instruction as expand


@pytest.fixture
def sample_entry():
    """A word-spotting entry with two alignments, both scripts populated."""
    return {
        "english_sentence": "You are the light of the world.",
        "gothic_sentence_roman": "jus sijuþ liuhaþ þis fairhvaus.",
        "gothic_sentence_gothic": "𐌾𐌿𐍃 𐍃𐌹𐌾𐌿𐌸 𐌻𐌹𐌿𐌷𐌰𐌸 𐌸𐌹𐍃 𐍆𐌰𐌹𐍂𐌷𐍅𐌰𐌿𐍃.",
        "alignments": [
            {
                "target_word": "light",
                "gothic_word_roman": "liuhaþ",
                "gothic_word_gothic": "𐌻𐌹𐌿𐌷𐌰𐌸",
            },
            {
                "target_word": "world",
                "gothic_word_roman": "fairhvaus",
                "gothic_word_gothic": "𐍆𐌰𐌹𐍂𐌷𐍅𐌰𐌿𐍃",
            },
        ],
    }


# --- blank_gothic_word ---------------------------------------------------


def test_blank_gothic_word_basic():
    blanked = expand.blank_gothic_word("jus sijuþ liuhaþ þis fairhvaus.", "liuhaþ")
    assert blanked == "jus sijuþ ____ þis fairhvaus."


def test_blank_gothic_word_preserves_trailing_punctuation():
    """A word followed by a period keeps the period on the blank."""
    blanked = expand.blank_gothic_word("jus sijuþ liuhaþ þis fairhvaus.", "fairhvaus")
    assert blanked == "jus sijuþ liuhaþ þis ____."


def test_blank_gothic_word_only_first_occurrence():
    blanked = expand.blank_gothic_word("sa waurd sa waurd.", "sa")
    assert blanked == "____ waurd sa waurd."


def test_blank_gothic_word_not_found_returns_none():
    assert expand.blank_gothic_word("jus sijuþ liuhaþ.", "missing") is None


def test_blank_gothic_word_no_partial_substring_match():
    """A word that is only a substring of a token is not blanked."""
    assert expand.blank_gothic_word("liuhaþei rules.", "liuhaþ") is None


# --- make_example: per-projection ---------------------------------------


def test_forward_example(sample_entry):
    rng = random.Random(0)
    example = expand.make_example(
        "forward", sample_entry["alignments"][0], sample_entry, "roman", rng, ["world"]
    )
    assert example["response"] == " liuhaþ"
    assert "light" in example["prompt"]
    assert example["prompt"].endswith("Response:")


def test_reverse_example_answers_english(sample_entry):
    rng = random.Random(0)
    example = expand.make_example(
        "reverse", sample_entry["alignments"][0], sample_entry, "roman", rng, ["world"]
    )
    assert example["response"] == " light"
    assert "liuhaþ" in example["prompt"]


def test_cloze_example_blanks_sentence(sample_entry):
    rng = random.Random(0)
    example = expand.make_example(
        "cloze", sample_entry["alignments"][0], sample_entry, "roman", rng, ["world"]
    )
    assert example["response"] == " liuhaþ"
    assert "____" in example["prompt"]
    # The answer word must not leak into the blanked prompt.
    assert "liuhaþ" not in example["prompt"]


def test_cloze_example_returns_none_when_word_absent(sample_entry):
    rng = random.Random(0)
    bad_alignment = {
        "target_word": "light",
        "gothic_word_roman": "notinsentence",
        "gothic_word_gothic": "notinsentence",
    }
    example = expand.make_example(
        "cloze", bad_alignment, sample_entry, "roman", rng, ["world"]
    )
    assert example is None


def test_discrimination_example_contains_both_options(sample_entry):
    rng = random.Random(0)
    example = expand.make_example(
        "discrimination",
        sample_entry["alignments"][0],
        sample_entry,
        "roman",
        rng,
        ["world"],
    )
    assert example["response"] == " light"
    assert "light" in example["prompt"]
    assert "world" in example["prompt"]


def test_gothic_script_uses_gothic_fields(sample_entry):
    rng = random.Random(0)
    example = expand.make_example(
        "forward", sample_entry["alignments"][0], sample_entry, "gothic", rng, ["world"]
    )
    assert example["response"] == " 𐌻𐌹𐌿𐌷𐌰𐌸"


# --- expand_entry --------------------------------------------------------


def test_expand_entry_both_scripts_all_projections(sample_entry):
    rng = random.Random(42)
    examples, skipped = expand.expand_entry(
        sample_entry,
        expand.PROJECTIONS,
        ["roman", "gothic"],
        pick_one_script=False,
        rng=rng,
        global_pool=["light", "world"],
    )
    # 2 alignments x 4 projections x 2 scripts = 16, no skips expected here.
    assert len(examples) == 16
    assert skipped == 0


def test_expand_entry_forward_only_backward_compatible(sample_entry):
    """forward + both scripts reproduces the original two-examples-per-alignment."""
    rng = random.Random(42)
    examples, skipped = expand.expand_entry(
        sample_entry,
        ["forward"],
        ["roman", "gothic"],
        pick_one_script=False,
        rng=rng,
        global_pool=["light", "world"],
    )
    assert len(examples) == 4
    responses = sorted(e["response"] for e in examples)
    assert responses == sorted([" liuhaþ", " 𐌻𐌹𐌿𐌷𐌰𐌸", " fairhvaus", " 𐍆𐌰𐌹𐍂𐌷𐍅𐌰𐌿𐍃"])


def test_expand_entry_cloze_skip_counted():
    """An alignment whose Gothic word is absent skips its cloze example."""
    entry = {
        "english_sentence": "The word.",
        "gothic_sentence_roman": "sa waurd.",
        "gothic_sentence_gothic": "𐍃𐌰 𐍅𐌰𐌿𐍂𐌳.",
        "alignments": [
            {
                "target_word": "word",
                "gothic_word_roman": "absent",
                "gothic_word_gothic": "absent",
            }
        ],
    }
    rng = random.Random(0)
    examples, skipped = expand.expand_entry(
        entry,
        ["cloze"],
        ["roman", "gothic"],
        pick_one_script=False,
        rng=rng,
        global_pool=["word"],
    )
    assert examples == []
    assert skipped == 2


def test_discrimination_uses_global_pool_when_no_sibling():
    """A single-alignment entry draws its distractor from the global pool."""
    entry = {
        "english_sentence": "The light.",
        "gothic_sentence_roman": "sa liuhaþ.",
        "gothic_sentence_gothic": "𐍃𐌰 𐌻𐌹𐌿𐌷𐌰𐌸.",
        "alignments": [
            {
                "target_word": "light",
                "gothic_word_roman": "liuhaþ",
                "gothic_word_gothic": "𐌻𐌹𐌿𐌷𐌰𐌸",
            }
        ],
    }
    rng = random.Random(0)
    examples, skipped = expand.expand_entry(
        entry,
        ["discrimination"],
        ["roman"],
        pick_one_script=False,
        rng=rng,
        global_pool=["light", "kingdom"],
    )
    assert len(examples) == 1
    assert skipped == 0
    # "kingdom" is the only non-self word in the global pool.
    assert "kingdom" in examples[0]["prompt"]
