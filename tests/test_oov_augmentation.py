"""
Tests for the gothic.oov_augmentation package.

Coverage (all hermetic — no corpus file reads; corpus-backed loaders like
build_stem_model / load_training_vocab are exercised end-to-end elsewhere):
- vocab: clean_token normalization, content_vocab function-word filter
- affixes: weighted-MI ranking, length-1 degeneracy, elbow, nested dedup,
  guarded stripping (single + both edges), attested junction chars
- nonword: prefix/suffix grafting, known-word rejection, accept predicate,
  length-guard failure
- augment: whole-word blanking, hedge-response assembly (glossed + bare),
  prompt styles, make_hedge_example / expand_entry (generate_nonword stubbed)
- diversity: char n-gram NLL, distinct-ngram ratio, quantile
"""

import random

import pytest

from gothic.oov_augmentation import affixes, augment, diversity, nonword, vocab
from gothic.oov_augmentation.affixes import AffixScore


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


# --- vocab ---------------------------------------------------------------


def test_clean_token_lowercases_and_strips_punctuation():
    assert vocab.clean_token("Saijands,") == "saijands"
    assert vocab.clean_token("·waurd·") == "waurd"


def test_content_vocab_drops_short_and_frequent():
    from collections import Counter

    counts = Counter(
        {
            "jah": 100,  # frequent function word (dropped by top_k)
            "sa": 90,  # also short (dropped by both filters)
            "saijands": 5,
            "waurd": 4,
            "in": 80,  # length 2 (dropped by length floor)
        }
    )
    kept = vocab.content_vocab(counts, min_length=3, top_k=1)
    assert set(kept) == {"saijands", "waurd"}


# --- affixes: scoring ----------------------------------------------------


# Six "-ans" words plus contrastive endings, so the -ans characters are not
# 100%-frequent at their positions (otherwise the position-unigram null predicts
# them perfectly and the over-representation ratio degenerates to 1.0).
_AFFIX_WORDS = [
    "saijans", "gibans", "nimans", "haitans", "waldans", "baurans",
    "stoma", "frauja", "razda", "himins", "dauþus", "fisks",
]


def test_score_affixes_sorted_by_weighted_mi_descending():
    scores = affixes.score_affixes(_AFFIX_WORDS, side="suffix", max_length=4, min_count=5)
    weighted = [score.weighted_mi for score in scores]
    assert weighted == sorted(weighted, reverse=True)


def test_score_affixes_surfaces_shared_suffix():
    scores = affixes.score_affixes(_AFFIX_WORDS, side="suffix", max_length=4, min_count=5)
    ans = next((s for s in scores if s.affix == "ans"), None)
    assert ans is not None
    assert ans.observed == 6
    # co-occurs at the word edge far more than independent per-position rates.
    assert ans.ratio > 1.0
    assert ans.weighted_mi > 0.0


def test_score_affixes_length_one_has_zero_weighted_mi():
    words = ["saijans", "gibans", "nimans", "haitans", "waldans", "baurans"]
    scores = affixes.score_affixes(words, side="suffix", max_length=1, min_count=1)
    # a single edge character cannot be over-represented relative to its own
    # position frequency, so ratio == 1 and weighted MI == 0 by construction.
    assert scores
    assert all(score.weighted_mi == 0.0 for score in scores)


# --- affixes: elbow ------------------------------------------------------


def test_find_elbow_small_inputs_return_all():
    assert affixes.find_elbow([]) == 0
    assert affixes.find_elbow([1.0]) == 1
    assert affixes.find_elbow([2.0, 1.0]) == 2


def test_find_elbow_linear_curve_has_no_interior_bend():
    # a straight descending line sits on its own chord everywhere.
    assert affixes.find_elbow([5.0, 4.0, 3.0, 2.0, 1.0]) == 1


def test_find_elbow_detects_dominance_step():
    # three dominant values then a floor: the sharpest bend is at the drop.
    assert affixes.find_elbow([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0]) == 4


# --- affixes: selection --------------------------------------------------


def _score(affix, side, weighted_mi):
    """Build a minimal AffixScore carrying only the fields select_affixes reads."""
    return AffixScore(
        affix=affix,
        side=side,
        length=len(affix),
        observed=int(weighted_mi),
        expected=1.0,
        ratio=2.0,
        weighted_mi=weighted_mi,
    )


def test_select_affixes_top_n_limits_count():
    scores = [_score(a, "suffix", w) for a, w in [("ans", 9), ("jan", 7), ("ins", 5)]]
    assert affixes.select_affixes(scores, top_n=2) == ["ans", "jan"]


def test_select_affixes_dedup_nested_keeps_stronger_prefix():
    # "fr" is edge-nested in the higher-scoring "fra" -> dropped; "ga" survives.
    scores = [
        _score("fra", "prefix", 8),
        _score("ga", "prefix", 6),
        _score("fr", "prefix", 4),
    ]
    kept = affixes.select_affixes(scores, top_n=None, dedup_nested=True)
    assert kept == ["fra", "ga"]


# --- affixes: stripping / junctions --------------------------------------


def test_strip_affix_respects_min_residual():
    # "saijands" -> strip the longer "-ands" (stem "saij", len 4 >= 3).
    stem, affix = affixes.strip_affix("saijands", ["ands", "nds"], "suffix", min_residual=3)
    assert (stem, affix) == ("saij", "ands")


def test_strip_affix_keeps_affix_when_stem_would_be_too_short():
    # "bans" minus "ans" leaves "b" (len 1 < 3): nothing stripped.
    assert affixes.strip_affix("bans", ["ans"], "suffix", min_residual=3) == ("bans", "")


def test_strip_both_reconstructs_word():
    prefix, stem, suffix = affixes.strip_both(
        "gasaijans", ["ga"], ["ans"], min_residual=3
    )
    assert (prefix, stem, suffix) == ("ga", "saij", "ans")
    assert prefix + stem + suffix == "gasaijans"


def test_attested_junction_chars_collects_adjacent_stem_chars():
    junction = affixes.attested_junction_chars(
        ["saijans", "gibans"], ["ans"], side="suffix", min_residual=3
    )
    assert junction == {"ans": {"j", "b"}}


# --- nonword generation --------------------------------------------------


def _single_word_model(word):
    """A model fit on one word, so generation is deterministic."""
    return nonword.TrigramNonwordModel(temperature=1.0, top_p=1.0, seed=1).fit([word])


def test_generate_produces_the_only_fitted_word():
    model = _single_word_model("abc")
    assert model.generate(min_length=3, reject_known=False) == "abc"


def test_generate_rejects_known_word():
    model = _single_word_model("abc")
    model.known_words = {"abc"}
    # the only reachable output collides with a known word -> no candidate.
    assert model.generate(min_length=3, reject_known=True) is None


def test_generate_honors_accept_predicate():
    model = _single_word_model("abc")
    assert model.generate(reject_known=False, accept=lambda word: False) is None
    assert model.generate(reject_known=False, accept=lambda word: True) == "abc"


def test_generate_grafts_prefix_and_suffix():
    model = _single_word_model("abc")
    grafted = model.generate(prefix="ga", suffix="ns", reject_known=False)
    assert grafted == "gaabcns"


def test_generate_returns_none_when_affixes_exceed_max_length():
    model = _single_word_model("abc")
    # body budget = max_length - len(prefix) - len(suffix) < 0 -> unsatisfiable.
    assert model.generate(prefix="aaaa", suffix="bbbb", max_length=5) is None


# --- augment: blanking / capitalization ----------------------------------


def test_blank_english_span_blanks_first_whole_word_case_insensitively():
    blanked = augment.blank_english_span("The farmer sows the word.", "FARMER")
    assert blanked == "The _____ sows the word."


def test_blank_english_span_respects_word_boundaries():
    # "sow" must not match inside "sows".
    unchanged = augment.blank_english_span("The farmer sows the word.", "sow")
    assert augment.BLANK not in unchanged


def test_capitalize_first():
    assert augment.capitalize_first("the word") == "The word"
    assert augment.capitalize_first("") == ""


# --- augment: response rendering -----------------------------------------


def test_render_hedge_response_bare_hedge_when_no_glosses():
    response = augment.render_hedge_response(
        [], 'I don\'t recognize the word "X"', "The _____ sows."
    )
    assert response.startswith("I don't recognize")
    assert "The _____ sows." in response
    assert ", but " not in response and "However" not in response


def test_render_hedge_response_includes_gloss_hedge_and_blanked_conclusion():
    response = augment.render_hedge_response(
        [("waurd", "word")],
        'I don\'t recognize the word "X"',
        "The _____ sows the word.",
    )
    assert "waurd" in response and "word" in response
    assert '"X"' in response
    assert "The _____ sows the word." in response
    assert ", but " in response


def test_render_hedge_response_is_deterministic_across_calls():
    """The response side consumes no RNG (unified format, v2.2.0)."""
    arguments = (
        [("waurd", "word"), ("manna", "man")],
        'I don\'t recognize the word "X"',
        "The _____ sows the word.",
    )
    responses = {augment.render_hedge_response(*arguments) for _ in range(10)}
    assert len(responses) == 1


# --- augment: prompt styles ----------------------------------------------


def test_build_prompt_plain_ends_in_response_cue():
    rng = random.Random(0)
    prompt = augment.build_prompt("sa gasand waurd.", "plain", rng)
    assert prompt.rstrip().endswith("Response:")
    assert "sa gasand waurd." in prompt


def test_build_prompt_cot_wraps_gothic_source():
    rng = random.Random(0)
    prompt = augment.build_prompt("sa gasand waurd.", "cot", rng)
    assert "Gothic: sa gasand waurd." in prompt
    assert prompt.rstrip().endswith("Response:")


# --- augment: example / entry assembly (generate_nonword stubbed) ---------


@pytest.fixture
def stub_nonword(monkeypatch):
    """Force a fixed, transliterable non-word so examples are deterministic."""
    monkeypatch.setattr(
        augment, "generate_nonword", lambda model, word, reject_known=True: "gasand"
    )


def test_make_hedge_example_roman_wellformed(sample_entry, stub_nonword):
    rng = random.Random(1)
    target = sample_entry["alignments"][0]  # saijands / farmer
    example = augment.make_hedge_example(
        sample_entry,
        target,
        sample_entry["alignments"],
        stem_model=None,
        rng=rng,
        script_key="roman",
        prompt_style="plain",
        min_gloss=1,
    )
    assert example is not None
    assert example["prompt"].rstrip().endswith("Response:")
    assert example["response"].startswith(" ")
    # the non-word replaced the source word in both the prompt and the hedge.
    assert "gasand" in example["prompt"]
    assert "saijands" not in example["prompt"]
    assert "gasand" in example["response"]
    # the aligned English span is blanked, not translated.
    assert augment.BLANK in example["response"]
    assert "farmer" not in example["response"]


def test_expand_entry_fans_out_scripts_and_styles(sample_entry, stub_nonword):
    rng = random.Random(1)
    examples = augment.expand_entry(
        sample_entry,
        stem_model=None,
        rng=rng,
        scripts=["roman", "gothic"],
        prompt_styles=["plain", "cot"],
        max_per_verse=None,
        min_gloss=1,
    )
    # 3 trainable words x 2 scripts x 2 prompt styles.
    assert len(examples) == 12


def test_expand_entry_max_per_verse_caps_corrupted_words(sample_entry, stub_nonword):
    rng = random.Random(1)
    examples = augment.expand_entry(
        sample_entry,
        stem_model=None,
        rng=rng,
        scripts=["roman"],
        prompt_styles=["plain"],
        max_per_verse=2,
        min_gloss=1,
    )
    assert len(examples) == 2


def test_expand_entry_no_trainable_alignments_returns_empty(stub_nonword):
    rng = random.Random(1)
    entry = {
        "english_sentence": "the son of Levi,",
        "gothic_sentence_roman": "sunaus Laiwweis,",
        "gothic_sentence_gothic": "x",
        "alignments": [],
    }
    assert augment.expand_entry(
        entry, None, rng, ["roman"], ["plain"], None, 1
    ) == []


# --- diversity -----------------------------------------------------------


def test_char_ngram_per_char_nll_is_positive_finite():
    model = diversity.CharNGramModel(order=3).fit(["saijan", "gibolan", "waurdjan"])
    nll = model.per_char_nll("saijan")
    assert nll > 0.0 and nll != float("inf")


def test_distinct_ngram_ratio():
    # "abcabc" trigrams: abc, bca, cab, abc -> 3 distinct / 4 total.
    assert diversity.distinct_ngram_ratio(["abcabc"], n=3) == 0.75


def test_quantile_nearest_rank():
    assert diversity.quantile([5, 1, 3, 2, 4], 0.5) == 3
    assert diversity.quantile([5, 1, 3, 2, 4], 0.9) == 5
