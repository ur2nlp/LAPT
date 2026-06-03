"""
Tests for gothic.word_spotting.expand_to_cot.

Coverage:
- trainable_alignments: status filtering, missing-status fallback
- token_position: match, punctuation/case handling, miss sentinel
- gloss_pair: direction-dependent (source_word, translation) orientation
- render_response: join styles, gloss ordering, full translation present
- make_cot_example: both directions, min_words gating, response well-formed
- expand_entry: direction/script fan-out and variants
"""

import random

import pytest

from gothic.word_spotting import expand_to_cot as cot


@pytest.fixture
def sample_entry():
    """A verse with three trainable alignments, both scripts populated."""
    return {
        "sentence_id": "train_0000",
        "english_sentence": "The farmer sows the word.",
        "gothic_sentence_roman": "sa saijands waurd saijiþ.",
        "gothic_sentence_gothic": "𐍃𐌰 𐍃𐌰𐌹𐌾𐌰𐌽𐌳𐍃 𐍅𐌰𐌿𐍂𐌳 𐍃𐌰𐌹𐌾𐌹𐌸.",
        "alignments": [
            {
                "status": "verified_correct",
                "target_word": "farmer",
                "gothic_word_roman": "saijands",
                "gothic_word_gothic": "𐍃𐌰𐌹𐌾𐌰𐌽𐌳𐍃",
            },
            {
                "status": "verified_correct",
                "target_word": "sows",
                "gothic_word_roman": "saijiþ",
                "gothic_word_gothic": "𐍃𐌰𐌹𐌾𐌹𐌸",
            },
            {
                "status": "verified_correct",
                "target_word": "word",
                "gothic_word_roman": "waurd",
                "gothic_word_gothic": "𐍅𐌰𐌿𐍂𐌳",
            },
        ],
    }


# --- trainable_alignments ------------------------------------------------


def test_trainable_alignments_filters_untrustworthy():
    entry = {
        "alignments": [
            {"status": "verified_correct", "target_word": "a"},
            {"status": "unverified", "target_word": "b"},
            {"status": "kept_edited", "target_word": "c"},
            {"status": "rejected", "target_word": "d"},
        ]
    }
    kept = cot.trainable_alignments(entry)
    assert [a["target_word"] for a in kept] == ["a", "c"]


def test_trainable_alignments_missing_status_is_kept():
    """Older finalized files without a status field are treated as trainable."""
    entry = {"alignments": [{"target_word": "a"}, {"target_word": "b"}]}
    assert len(cot.trainable_alignments(entry)) == 2


# --- token_position ------------------------------------------------------


def test_token_position_strips_punctuation():
    assert cot.token_position("sa saijands waurd saijiþ.", "saijiþ", False) == 3


def test_token_position_case_insensitive_english():
    assert cot.token_position("The farmer sows.", "the", True) == 0


def test_token_position_miss_returns_sentinel():
    assert cot.token_position("sa waurd.", "missing", False) == 2


# --- gloss_pair ----------------------------------------------------------


def test_gloss_pair_got2eng_orientation():
    alignment = {"target_word": "word", "gothic_word_roman": "waurd"}
    assert cot.gloss_pair(alignment, "got2eng", "gothic_word_roman") == (
        "waurd",
        "word",
    )


def test_gloss_pair_eng2got_orientation():
    alignment = {"target_word": "word", "gothic_word_roman": "waurd"}
    assert cot.gloss_pair(alignment, "eng2got", "gothic_word_roman") == (
        "word",
        "waurd",
    )


# --- render_response -----------------------------------------------------


def test_render_response_contains_full_translation_and_all_glosses():
    items = [("waurd", "word"), ("saijands", "farmer")]
    rng = random.Random(0)
    response = cot.render_response(items, "The farmer sows the word.", "got2eng", rng)
    assert "The farmer sows the word." in response
    assert "waurd" in response and "saijands" in response
    assert "word" in response and "farmer" in response


def test_render_response_single_item_has_no_dangling_and():
    items = [("waurd", "word")]
    rng = random.Random(0)
    response = cot.render_response(items, "the word.", "got2eng", rng)
    assert ", and " not in response


# --- make_cot_example ----------------------------------------------------


def test_make_cot_example_got2eng_prompt_has_gothic_source(sample_entry):
    rng = random.Random(1)
    example = cot.make_cot_example(sample_entry, "got2eng", "roman", rng, 1)
    assert "Gothic: sa saijands waurd saijiþ." in example["prompt"]
    assert example["prompt"].endswith("Response:")
    assert example["response"].startswith(" ")
    assert "The farmer sows the word." in example["response"]


def test_make_cot_example_eng2got_prompt_has_english_source(sample_entry):
    rng = random.Random(1)
    example = cot.make_cot_example(sample_entry, "eng2got", "gothic", rng, 1)
    assert "English: The farmer sows the word." in example["prompt"]
    assert "𐍃𐌰 𐍃𐌰𐌹𐌾𐌰𐌽𐌳𐍃 𐍅𐌰𐌿𐍂𐌳 𐍃𐌰𐌹𐌾𐌹𐌸." in example["response"]


def test_make_cot_example_respects_min_words(sample_entry):
    """A verse with fewer trainable alignments than min_words yields None."""
    rng = random.Random(1)
    assert cot.make_cot_example(sample_entry, "got2eng", "roman", rng, 4) is None


def test_make_cot_example_glosses_ordered_by_source_position(sample_entry):
    """With all three words glossed, they appear in Gothic reading order."""
    # force count == len(alignments) by seeding so randint picks the max; assert
    # ordering directly instead of relying on the seed.
    rng = random.Random(1)
    example = cot.make_cot_example(sample_entry, "got2eng", "roman", rng, 3)
    response = example["response"]
    # saijands (idx 1) precedes waurd (idx 2) precedes saijiþ (idx 3)
    assert response.index("saijands") < response.index("waurd") < response.index("saijiþ")


# --- expand_entry --------------------------------------------------------


def test_expand_entry_fans_out_directions_and_scripts(sample_entry):
    rng = random.Random(1)
    examples, skipped = cot.expand_entry(
        sample_entry,
        directions=["got2eng", "eng2got"],
        scripts=["roman", "gothic"],
        pick_one_script=False,
        variants=1,
        min_words=1,
        rng=rng,
    )
    assert len(examples) == 4
    assert skipped == 0


def test_expand_entry_variants_multiply(sample_entry):
    rng = random.Random(1)
    examples, _ = cot.expand_entry(
        sample_entry,
        directions=["got2eng"],
        scripts=["roman"],
        pick_one_script=False,
        variants=3,
        min_words=1,
        rng=rng,
    )
    assert len(examples) == 3
