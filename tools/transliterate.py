#!/usr/bin/env python3
"""
Transliteration utilities for Gothic script.

Provides bidirectional transliteration between Latin romanization and Gothic Unicode
characters. Can be used as a module or run standalone to transliterate files.

Usage as module:
    from transliterate import transliterate_latin_to_gothic, transliterate_gothic_to_latin

    gothic_text = transliterate_latin_to_gothic("saihwan")
    latin_text = transliterate_gothic_to_latin("𐍃𐌰𐌹𐍈𐌰𐌽")

Usage as CLI:
    python transliterate.py input.txt output.txt --direction to-gothic
    python transliterate.py input.txt output.txt --direction to-latin
"""

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
        # Digraph - must be processed before single characters
        "hw": "𐍈",  # hwair
        "Hw": "𐍈",
        "HW": "𐍈",

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
    gothic_to_latin = {}
    for latin, gothic in latin_to_gothic.items():
        if gothic not in gothic_to_latin:
            # Use lowercase version for reverse mapping
            gothic_to_latin[gothic] = latin.lower()

    # Sort by length descending (though all Gothic chars are single code points)
    for gothic, latin in sorted(gothic_to_latin.items(), key=lambda x: -len(x[0])):
        text = text.replace(gothic, latin)

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
