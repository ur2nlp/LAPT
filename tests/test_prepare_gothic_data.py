"""
Tests for gothic.data.prepare_gothic_data.build_instruction_prompt.

Coverage:
- prompt always ends with the ' Response:' delimiter
- {target} substitution for the English -> Gothic direction
- input quote-wrapping coin flip (wrap below threshold, bare above)
- nested-quote guard: inputs already containing a quote are never wrapped
- template drawn from the requested task's list
- reproducibility under a seeded RNG
"""

import random

from gothic.data import prepare_gothic_data as prep


class StubRandom:
    """Deterministic stand-in for random.Random with controllable outputs."""

    def __init__(self, choice_index: int = 0, random_value: float = 0.0):
        self.choice_index = choice_index
        self.random_value = random_value

    def choice(self, seq):
        return seq[self.choice_index]

    def random(self):
        return self.random_value


def test_prompt_ends_with_response_delimiter():
    rng = StubRandom(choice_index=0, random_value=0.9)
    prompt = prep.build_instruction_prompt("to_english", "𐌵𐌰𐌸", rng)
    assert prompt.endswith(" Response:")


def test_to_target_substitutes_label():
    rng = StubRandom(choice_index=0, random_value=0.9)
    prompt = prep.build_instruction_prompt(
        "to_target", "the word", rng, target_label="Romanized Gothic"
    )
    assert "Romanized Gothic" in prompt
    assert "{target}" not in prompt


def test_quote_wrap_below_threshold():
    rng = StubRandom(choice_index=0, random_value=0.0)
    prompt = prep.build_instruction_prompt("to_gothic_script", "dog", rng)
    assert '"dog"' in prompt


def test_no_quote_wrap_above_threshold():
    rng = StubRandom(choice_index=0, random_value=0.9)
    prompt = prep.build_instruction_prompt("to_gothic_script", "dog", rng)
    assert '"dog"' not in prompt
    assert "dog" in prompt


def test_guard_skips_wrap_when_input_has_quote():
    """An input that already contains a quote is never wrapped, even when the
    coin flip would otherwise wrap it (random_value below threshold)."""
    rng = StubRandom(choice_index=0, random_value=0.0)
    text = 'He said, "go".'
    prompt = prep.build_instruction_prompt("to_english", text, rng)
    input_region = prompt[: prompt.rfind(" Response:")].split(": ", 1)[1]
    assert input_region == text
    assert not (input_region.startswith('"') and input_region.endswith('"'))


def test_template_drawn_from_task_list():
    rng = StubRandom(choice_index=1, random_value=0.9)
    prompt = prep.build_instruction_prompt("to_latin_script", "𐌵𐌰𐌸", rng)
    expected = prep.INSTRUCTION_TEMPLATES["to_latin_script"][1].format(
        input="𐌵𐌰𐌸", target=None
    )
    assert prompt == f"{expected} Response:"


def test_reproducible_under_seeded_rng():
    inputs = ["dog", "the word", "a narrative", "light"]
    rng_a = random.Random(1)
    rng_b = random.Random(1)
    prompts_a = [prep.build_instruction_prompt("to_english", text, rng_a) for text in inputs]
    prompts_b = [prep.build_instruction_prompt("to_english", text, rng_b) for text in inputs]
    assert prompts_a == prompts_b
