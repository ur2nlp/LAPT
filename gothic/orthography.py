#!/usr/bin/env python3
"""
Orthography utilities for Gothic.

Provides two related families of operations:

1. Bidirectional transliteration between Latin romanization and Gothic Unicode
   script (``transliterate_latin_to_gothic`` / ``transliterate_gothic_to_latin``).
2. Normalization of romanized dictionary orthography (macrons, acute accents,
   the ƕ digraph) to the surface conventions used in the gotica corpus
   (``normalize_orthography``).

Usage as module:
    from gothic.orthography import transliterate_latin_to_gothic, normalize_orthography

    gothic_text = transliterate_latin_to_gothic("saihwan")
    latin_text = transliterate_gothic_to_latin("𐍃𐌰𐌹𐍈𐌰𐌽")
    surface = normalize_orthography("dwalmōn")  # -> "dwalmon"

Usage as CLI:
    python orthography.py input.txt output.txt --direction to-gothic
    python orthography.py input.txt output.txt --direction to-latin
"""

import re


def get_latin_gothic_mappings():
    """
    Returns Latin-to-Gothic character mappings.

    Maps Latin characters to Gothic Unicode. The digraph 'hw' maps to 𐍈 (hwair).
    Note: 'th' is NOT a digraph - when it appears it's t+h separately (e.g., athaitands).
    The character þ represents Gothic 𐌸 (thorn).

    Returns:
        dict: Mapping from Latin characters/digraphs to Gothic Unicode characters
    """
    return {
        # Digraphs - must be processed before single characters
        "hw": "𐍈",  # hwair
        "hv": "𐍈",  # hwair (alternative transcription)
        "Hw": "𐍈",
        "Hv": "𐍈",
        "HW": "𐍈",
        "HV": "𐍈",

        # Special character for thorn
        "þ": "𐌸",
        "Þ": "𐌸",

        # Single characters (alphabetical order)
        "a": "𐌰",
        "b": "𐌱",
        "d": "𐌳",
        "e": "𐌴",
        "f": "𐍆",
        "g": "𐌲",
        "h": "𐌷",
        "i": "𐌹",
        "j": "𐌾",
        "k": "𐌺",
        "l": "𐌻",
        "m": "𐌼",
        "n": "𐌽",
        "o": "𐍉",
        "p": "𐍀",
        "q": "𐌵",
        "r": "𐍂",
        "s": "𐍃",
        "t": "𐍄",
        "u": "𐌿",
        "v": "𐍅",  # v and w represent the same sound (wynn/uuinne)
        "w": "𐍅",
        "x": "𐍇",
        "z": "𐌶",

        # Uppercase variants
        "A": "𐌰",
        "B": "𐌱",
        "D": "𐌳",
        "E": "𐌴",
        "F": "𐍆",
        "G": "𐌲",
        "H": "𐌷",
        "I": "𐌹",
        "J": "𐌾",
        "K": "𐌺",
        "L": "𐌻",
        "M": "𐌼",
        "N": "𐌽",
        "O": "𐍉",
        "P": "𐍀",
        "Q": "𐌵",
        "R": "𐍂",
        "S": "𐍃",
        "T": "𐍄",
        "U": "𐌿",
        "V": "𐍅",  # v and w represent the same sound (wynn/uuinne)
        "W": "𐍅",
        "X": "𐍇",
        "Z": "𐌶",
    }


def transliterate_latin_to_gothic(text):
    """
    Transliterate Latin romanization to Gothic Unicode script.

    Handles the digraph hw → 𐍈 and special character þ → 𐌸.
    Case-insensitive for single characters (both map to Gothic lowercase).
    Note: 'th' is not treated as a digraph (e.g., 'athaitands' → 𐌰𐍄𐌷𐌰𐌹𐍄𐌰𐌽𐌳𐍃).

    Args:
        text: String in Latin romanization

    Returns:
        String in Gothic Unicode script

    Example:
        >>> transliterate_latin_to_gothic("saihwan þata")
        '𐍃𐌰𐌹𐍈𐌰𐌽 𐌸𐌰𐍄𐌰'
    """
    latin_to_gothic = get_latin_gothic_mappings()

    # Sort by length descending to handle digraphs before single characters
    for latin, gothic in sorted(latin_to_gothic.items(), key=lambda x: -len(x[0])):
        text = text.replace(latin, gothic)

    return text


def transliterate_gothic_to_latin(text):
    """
    Transliterate Gothic Unicode script to Latin romanization.

    Converts Gothic characters to lowercase Latin equivalents.
    The digraph 𐍈 → hw and special character 𐌸 → þ.

    Args:
        text: String in Gothic Unicode script

    Returns:
        String in Latin romanization (lowercase)

    Example:
        >>> transliterate_gothic_to_latin("𐍃𐌰𐌹𐍈𐌰𐌽 𐌸𐌰𐍄𐌰")
        'saihwan þata'
    """
    latin_to_gothic = get_latin_gothic_mappings()

    # Create reverse mapping (Gothic → Latin lowercase only)
    # Preferred forms for ambiguous mappings: w over v, hw over hv (gotica convention)
    preferred_reverse = {"𐍅": "w", "𐍈": "hw"}
    gothic_to_latin = {}
    for latin, gothic in latin_to_gothic.items():
        if gothic in preferred_reverse:
            gothic_to_latin[gothic] = preferred_reverse[gothic]
        elif gothic not in gothic_to_latin:
            gothic_to_latin[gothic] = latin.lower()

    # Sort by length descending (though all Gothic chars are single code points)
    for gothic, latin in sorted(gothic_to_latin.items(), key=lambda x: -len(x[0])):
        text = text.replace(gothic, latin)

    return text


# Normalize romanized dictionary orthography to match gotica surface forms.
# Macrons mark vowel length, acute accents mark diphthong components (aí=ai, aú=au),
# ƕ is the Unicode digraph character for hw, ↑ is a cross-reference artifact.
ORTHOGRAPHY_MAP = str.maketrans(
    {
        "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
        "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U",
        "á": "a", "í": "i", "ú": "u",
        "à": "a", "ì": "i",
        "ä": "a",
        "↑": "",
    }
)


def normalize_orthography(text: str) -> str:
    """Normalize romanized Gothic dictionary orthography to gotica conventions.

    Strips macrons (ā→a), acute accents (aí→ai, aú→au), grave accents,
    diaeresis, cross-reference artifacts (↑), and expands ƕ→hw.

    Args:
        text: Romanized Gothic text using dictionary orthographic conventions.

    Returns:
        The text normalized to the surface orthography of the gotica corpus.
    """
    text = text.translate(ORTHOGRAPHY_MAP)
    # ƕ → hw (must be done as string replacement, not char-to-char)
    text = text.replace("ƕ", "hw").replace("Ƕ", "Hw")
    return text


# A parenthesized run of pure ASCII digits — the decimal-value gloss of Gothic
# letter-numerals in the genealogy tables (e.g. "·t· ·l· ·g· (333)"). The
# preceding whitespace is consumed so removal does not leave a double space.
_ARABIC_NUMERAL_PAREN = re.compile(r"\s*\(\d+\)")

# A trailing fragment-continuation dash left at the end of a genealogy line.
_TRAILING_DASH = re.compile(r"\s*-\s*$")

# Rare editorial diacritics attested in the gotica corpus, mapped to their base
# letter. Two independent reasons to strip them:
#   1. They are generation attractors — û (a long-vowel editor's mark, only þû/jû)
#      and ï (the hiatus/word-initial diaeresis, saïas/gaïddjedun) are exceptional
#      enough that the model latches onto them and emits them spuriously.
#   2. transliterate_latin_to_gothic has no mapping for them, so any that survive
#      leak *untransliterated* into the Gothic-script side (raw Latin letters
#      inside a Gothic-script string). Normalizing here, before transliteration,
#      fixes both the Roman surface and that leak.
# Extend this map if further such diacritics appear in the source.
_EDITORIAL_DIACRITICS = str.maketrans(
    {
        "û": "u", "Û": "U",
        "ï": "i", "Ï": "I",
    }
)


def clean_gothic_artifacts(text: str) -> str:
    """Strip editorial annotation artifacts from a Gothic surface string.

    The gotica source marks editorial interventions that are noise for modeling
    and for word-spotting alignment. This removes them:

    - **Restoration parentheses** wrap letters the editor supplied where the
      manuscript is damaged, e.g. ``wa(i)ros``, ``(su)niwe``, ``(jah)``, and
      parenthesized Gothic numerals ``(·k·)``. The parentheses are dropped and
      the enclosed text kept (it is the intended reading). No Gothic-side
      parenthetical spans whitespace, so dropping the parenthesis characters is
      exactly an unwrap — unlike English translation text, which *does* contain
      genuine whitespace-spanning parentheticals and must not be passed here.
    - **Arabic-numeral glosses** ``(2222)``, ``(654)`` give the decimal value of
      the preceding Gothic letter-numerals; the whole token is removed.
    - **Enclitic tildes** ``~`` mark the boundary of an assimilated enclitic
      (Streitberg notation), e.g. ``qaþuþ~þan`` = qaþ+uh+þan. The manuscript
      writes a single word, so the tilde is dropped (join).
    - **Rare editorial diacritics** ``û`` and ``ï`` are stripped to their base
      letter (``û``→``u``, ``ï``→``i``); see ``_EDITORIAL_DIACRITICS``.
    - A trailing fragment-continuation dash (`` -``) is stripped.

    Genuine Gothic letter-numerals (``·b·``, ``·x·``, ``·{90}·``) are preserved.

    Args:
        text: A Gothic string in either Roman or Gothic script.

    Returns:
        The cleaned string.
    """
    # Remove decimal-gloss tokens first, before the blanket parenthesis strip
    # would turn "(333)" into a bare "333".
    text = _ARABIC_NUMERAL_PAREN.sub("", text)
    # Unwrap all remaining (whitespace-free, non-numeric) restoration parentheses.
    text = text.replace("(", "").replace(")", "")
    # Join assimilated enclitics written across a tilde.
    text = text.replace("~", "")
    # Normalize rare editorial diacritics to their base letter. Must run before
    # transliteration so they never leak untransliterated into Gothic script.
    text = text.translate(_EDITORIAL_DIACRITICS)
    # Drop a trailing fragment-continuation dash and collapse leftover whitespace.
    text = _TRAILING_DASH.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transliterate between Latin romanization and Gothic Unicode script"
    )
    parser.add_argument(
        "input_file",
        help="Input file path"
    )
    parser.add_argument(
        "output_file",
        help="Output file path"
    )
    parser.add_argument(
        "--direction",
        choices=["to-gothic", "to-latin"],
        required=True,
        help="Direction of transliteration"
    )

    args = parser.parse_args()

    # Read input file
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Transliterate
    if args.direction == "to-gothic":
        result = transliterate_latin_to_gothic(text)
        print(f"Transliterated {args.input_file} from Latin to Gothic")
    else:
        result = transliterate_gothic_to_latin(text)
        print(f"Transliterated {args.input_file} from Gothic to Latin")

    # Write output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(result)

    print(f"Wrote output to {args.output_file}")
