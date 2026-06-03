"""
Tests for gothic.dictionary_lookup.expand_to_instruction.

Coverage:
- forward emits one row per (gloss x form); response is the Gothic form
- variant spellings emitted in both directions (forward targets, reverse inputs)
- reverse gloss modes: sample (one), each (one per gloss), join (comma-joined)
- reconstructed headwords excluded by default, included with the flag
- entries with no glosses (cross-reference-only) produce nothing
- gothic script transliterates the form
- quote diversification: --quote-prob bounds and the nesting guard
"""

import random

from gothic.dictionary_lookup import expand_to_instruction as ex


def make_entry(headword_normalized, glosses, variant_forms=None, reconstructed=False):
    """Build a minimal parsed-glossary entry dict for the expander."""
    return {
        "headword": headword_normalized,
        "headword_segmented": headword_normalized,
        "headword_normalized": headword_normalized,
        "variant_forms": variant_forms or [],
        "pos": None,
        "reconstructed": reconstructed,
        "glosses": glosses,
        "cross_reference": [],
    }


def run(entry, **overrides):
    """Run expand_entry with test-friendly defaults (Roman, no quoting)."""
    kwargs = dict(
        directions=["forward", "reverse"],
        scripts=["roman"],
        pick_one_script=False,
        reverse_gloss_mode="sample",
        variants=1,
        include_reconstructed=False,
        quote_prob=0.0,
        rng=random.Random(1),
    )
    kwargs.update(overrides)
    return ex.expand_entry(entry, **kwargs)


def test_forward_one_row_per_gloss_and_form():
    entry = make_entry("aba", ["man", "husband"])
    rows = run(entry, directions=["forward"])
    assert len(rows) == 2
    assert all(row["response"] == " aba" for row in rows)
    prompts = " ".join(row["prompt"] for row in rows)
    assert "man" in prompts and "husband" in prompts


def test_variant_forms_emitted_both_directions():
    entry = make_entry("apaustaulus", ["apostle"], variant_forms=["apaulstulus"])
    forward = run(entry, directions=["forward"])
    assert {row["response"] for row in forward} == {" apaustaulus", " apaulstulus"}

    reverse = run(entry, directions=["reverse"], reverse_gloss_mode="each")
    assert all(row["response"] == " apostle" for row in reverse)
    prompts = " ".join(row["prompt"] for row in reverse)
    assert "apaustaulus" in prompts and "apaulstulus" in prompts


def test_reverse_each_mode_one_row_per_gloss():
    entry = make_entry("saiands", ["one sowing", "sower"])
    rows = run(entry, directions=["reverse"], reverse_gloss_mode="each")
    assert {row["response"] for row in rows} == {" one sowing", " sower"}


def test_reverse_join_mode_single_row():
    entry = make_entry("saiands", ["one sowing", "sower"])
    rows = run(entry, directions=["reverse"], reverse_gloss_mode="join")
    assert len(rows) == 1
    assert rows[0]["response"] == " one sowing, sower"


def test_reverse_sample_mode_single_row():
    entry = make_entry("saiands", ["one sowing", "sower"])
    rows = run(entry, directions=["reverse"], reverse_gloss_mode="sample")
    assert len(rows) == 1
    assert rows[0]["response"] in (" one sowing", " sower")


def test_reconstructed_excluded_by_default_included_with_flag():
    entry = make_entry("afdojan", ["to fatigue"], reconstructed=True)
    assert run(entry) == []
    included = run(entry, directions=["forward"], include_reconstructed=True)
    assert len(included) == 1
    assert included[0]["response"] == " afdojan"


def test_entry_without_glosses_produces_nothing():
    entry = make_entry("gaumei", [])
    assert run(entry) == []


def test_gothic_script_transliterates_form():
    entry = make_entry("aba", ["man"])
    rows = run(entry, directions=["forward"], scripts=["gothic"])
    assert rows[0]["response"] == " 𐌰𐌱𐌰"


def test_script_label_disambiguates_output_script():
    entry = make_entry("aba", ["man"])
    roman = run(entry, directions=["forward"], scripts=["roman"])
    assert all("romanized Gothic" in row["prompt"] for row in roman)
    assert all(row["response"] == " aba" for row in roman)

    gothic = run(entry, directions=["forward"], scripts=["gothic"])
    # Native-script answers use the unmarked "Gothic" label, never "romanized".
    assert all("romanized" not in row["prompt"] for row in gothic)
    assert all("Gothic" in row["prompt"] for row in gothic)
    assert all(row["response"] == " 𐌰𐌱𐌰" for row in gothic)


def test_quote_prob_bounds():
    entry = make_entry("aba", ["man"])
    always = run(entry, directions=["forward"], quote_prob=1.0)
    assert all('"' in row["prompt"] for row in always)
    never = run(entry, directions=["forward"], quote_prob=0.0)
    assert all('"' not in row["prompt"] for row in never)


def test_maybe_quote_nesting_guard():
    assert ex.maybe_quote("apostle", random.Random(1), 1.0) == '"apostle"'
    assert ex.maybe_quote("apostle", random.Random(1), 0.0) == "apostle"
    # Text already containing a quote is left untouched even at prob 1.0.
    assert ex.maybe_quote('he said "hi"', random.Random(1), 1.0) == 'he said "hi"'
