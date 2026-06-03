"""
Tests for gothic.dictionary_lookup.parse_wright_glossary.

Coverage:
- end-to-end parse of a GLP-style HTML fixture into structured entries
- gloss splitting on commas/semicolons; trailing period stripped
- cognate/etymology run cut from the gloss (both with and without a page-ref em)
- page-reference <em> excluded from the gloss
- "see ..." cross-references captured (hyphen-stripped) and removed from glosses
- inline Gothic example (<strong> mid-entry) dropped, not treated as a headword
- inter-entry page markers (<em>NNa</em>) ignored
- character images mapped to Unicode: thorn -> þ, hwair -> ƕ, vowel macrons
- unmapped (Greek) images dropped and reported
- headword orthography: faithful / segmented (hyphens kept) / normalized (solid)
- reconstructed (*) flag
- helper functions: split_glosses, clean_headword, clean_pos
"""

from gothic.dictionary_lookup import parse_wright_glossary as pw


# Minimal HTML mirroring the Germanic Lexicon Project markup of Wright's
# glossary: <br>-delimited entries, <em> for POS and page refs, untagged cognate
# runs, character images as <IMG ... chars/NAME.png>, and page markers.
SAMPLE_HTML = """
<hr>
<br><br><em>1a</em><br><br>
<strong>aba,</strong>
<em>wm.</em>
man, husband,
<em>206, 208 note, 353.</em>
O.Icel. afe. Gr. <IMG SRC="http://x/chars/alpha.png">.
<br>
<strong>abrs,</strong>
<em>aj.</em>
strong, violent, great, mighty.
O.Icel. afar.
<br>
<strong>af-airzjan,</strong>
<em>wv. I,</em>
to deceive, lead astray;
see <strong> airzeis, airzjan.</strong>
<br>
<strong>afar-sabbatus,</strong>
<em>sm.</em>
the day after the Sabbath;
<strong><IMG SRC="http://x/chars/thorn.png">is dagis afar-sabbat<IMG SRC="http://x/chars/e-long.png">,</strong>
on the first day of the week,
<em>356</em>
<br>
<strong>af-d&aacute;u<IMG SRC="http://x/chars/thorn.png">jan,</strong>
<em>wv. I,</em>
to kill, put to death,
<em>402.</em>
<br>
<strong>af-<IMG SRC="http://x/chars/hw.png">apjan,</strong>
<em>wv. I,</em>
to choke, quench.
<br>
<br><br><em>2a</em><br><br>
<strong>*us-anan,</strong>
<em>sv. VI,</em>
to expire.
<br>
<strong>riqis, riqiz,</strong>
<em>sn.</em>
darkness.
<br>
<strong>gaumjan,</strong>
<em>wv. I,</em>
to perceive, see, behold, observe,
<em>402.</em>
<br>
<strong>saiands,</strong>
<em>pres.</em>
part. one sowing, sower.
<br>
<strong>gaumei,</strong>
<em>sf.</em>
See <strong> gaumjan.</strong>
<br>
<strong>balgs,</strong>
<em>sm.</em>
wine-skin, pl. nom. -eis, gen. -e,
<em>206.</em>
<br>
"""


def parse_sample():
    """Run the parser pipeline on SAMPLE_HTML; return (by_headword, unmapped)."""
    parser = pw.WrightHTMLParser()
    parser.feed(SAMPLE_HTML)
    entries = []
    for entry_tokens in pw.segment_entries(parser.tokens):
        entry = pw.parse_entry(entry_tokens)
        if entry is not None and (entry.glosses or entry.cross_reference):
            entries.append(entry)
    by_headword = {entry.headword: entry for entry in entries}
    return by_headword, parser.unmapped_images


def test_expected_entry_set():
    by_headword, _ = parse_sample()
    # Page markers (1a, 2a) and the inline example are not entries.
    assert set(by_headword) == {
        "aba",
        "abrs",
        "af-airzjan",
        "afar-sabbatus",
        "af-dáuþjan",
        "af-ƕapjan",
        "us-anan",
        "riqis",
        "gaumjan",
        "saiands",
        "gaumei",
        "balgs",
    }


def test_basic_entry_glosses_and_pos():
    by_headword, _ = parse_sample()
    aba = by_headword["aba"]
    assert aba.pos == "wm."
    assert aba.glosses == ["man", "husband"]
    # Page-ref em and cognate run are excluded.
    assert aba.cross_reference == []


def test_cognate_cut_without_page_ref():
    by_headword, _ = parse_sample()
    abrs = by_headword["abrs"]
    # "O.Icel. afar." follows the gloss in the same untagged run and is cut.
    assert abrs.glosses == ["strong", "violent", "great", "mighty"]


def test_cross_reference_captured_and_removed_from_glosses():
    by_headword, _ = parse_sample()
    entry = by_headword["af-airzjan"]
    assert entry.glosses == ["to deceive", "lead astray"]
    assert entry.cross_reference == ["airzeis", "airzjan"]


def test_inline_example_dropped():
    by_headword, _ = parse_sample()
    entry = by_headword["afar-sabbatus"]
    assert entry.glosses == ["the day after the Sabbath"]
    assert entry.cross_reference == []


def test_thorn_and_macron_image_mapping_and_normalization():
    by_headword, _ = parse_sample()
    entry = by_headword["af-dáuþjan"]
    assert entry.headword == "af-dáuþjan"
    assert entry.headword_segmented == "af-dauþjan"
    assert entry.headword_normalized == "afdauþjan"
    assert entry.glosses == ["to kill", "put to death"]


def test_hwair_image_mapping_and_normalization():
    by_headword, _ = parse_sample()
    entry = by_headword["af-ƕapjan"]
    assert entry.headword == "af-ƕapjan"
    assert entry.headword_segmented == "af-hwapjan"
    assert entry.headword_normalized == "afhwapjan"


def test_reconstructed_flag_and_hyphen_stripping():
    by_headword, _ = parse_sample()
    entry = by_headword["us-anan"]
    assert entry.reconstructed is True
    assert entry.headword == "us-anan"
    assert entry.headword_normalized == "usanan"


def test_unmapped_greek_image_reported_and_dropped():
    by_headword, unmapped = parse_sample()
    assert "alpha" in unmapped
    # The Greek glyph lived in the cognate run, so no gloss contains an artifact.
    for entry in by_headword.values():
        for gloss in entry.glosses:
            assert gloss.strip()


def test_variant_spellings_split_into_list():
    by_headword, _ = parse_sample()
    entry = by_headword["riqis"]
    assert entry.headword == "riqis"
    assert entry.headword_normalized == "riqis"
    assert entry.variant_forms == ["riqiz"]
    assert entry.glosses == ["darkness"]


def test_participle_part_prefix_stripped():
    by_headword, _ = parse_sample()
    entry = by_headword["saiands"]
    assert entry.glosses == ["one sowing", "sower"]


def test_verb_see_gloss_retained():
    # "see" is a real gloss (the verb), not a cross-reference marker here.
    by_headword, _ = parse_sample()
    entry = by_headword["gaumjan"]
    assert entry.glosses == ["to perceive", "see", "behold", "observe"]
    assert entry.cross_reference == []


def test_capitalized_see_cross_reference():
    by_headword, _ = parse_sample()
    entry = by_headword["gaumei"]
    assert entry.glosses == []
    assert entry.cross_reference == ["gaumjan"]


def test_inflectional_paradigm_note_dropped():
    by_headword, _ = parse_sample()
    entry = by_headword["balgs"]
    assert entry.glosses == ["wine-skin"]


def test_split_glosses_parentheticals_verb_see_and_morph_notes():
    assert pw.split_glosses("catch (with the hand), seize") == [
        "catch (with the hand)",
        "seize",
    ]
    # "see" is kept here (cross-ref handling lives in collect_gloss).
    assert pw.split_glosses("to deceive; see") == ["to deceive", "see"]
    assert pw.split_glosses("man, husband.") == ["man", "husband"]
    # Inflectional paradigm notes are dropped.
    assert pw.split_glosses("apostle, pl. nom. -eis, gen. -e") == ["apostle"]


def test_clean_headword_strips_trailing_comma():
    assert pw.clean_headword("aba,") == "aba"
    assert pw.clean_headword("  af-airzjan,  ") == "af-airzjan"


def test_clean_pos_strips_trailing_comma():
    assert pw.clean_pos("wv. I,") == "wv. I"
    assert pw.clean_pos("prep. c. dat.") == "prep. c. dat."
