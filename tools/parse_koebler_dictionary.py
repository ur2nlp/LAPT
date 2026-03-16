#!/usr/bin/env python3
"""
Parse the Koebler Gothic-English dictionary from HTML to structured JSON.

Converts the Word-generated HTML into a clean JSON array of dictionary entries,
each containing the English headword and one or more Gothic sub-entries with
citation form, frequency, POS, morphology, and glosses.

Input: data/gothic_dictionaries/koeblergerhard_dictionary_utf8.html
Output: data/gothic_dictionaries/koebler_gothic_english.json
"""

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser


@dataclass
class GothicSubEntry:
    citation_form: str
    frequency: int | None = None
    reconstructed: bool = False
    uncertain: bool = False
    morphology: str | None = None
    pos: str | None = None
    german: str | None = None
    english_glosses: list[str] = field(default_factory=list)


@dataclass
class DictionaryEntry:
    english: str
    gothic_entries: list[GothicSubEntry] = field(default_factory=list)


# Segments are (text, role) tuples where role is one of:
# "headword", "bold", "italic", "standard"
Segment = tuple[str, str]


class KoeblerHTMLParser(HTMLParser):
    """Extract annotated text segments from each <p> entry in the dictionary."""

    def __init__(self):
        super().__init__()
        self.paragraphs: list[list[Segment]] = []
        self.current_segments: list[Segment] | None = None
        self.in_p = False
        self.in_headword_span = False
        self.in_bold = False
        self.in_italic = False
        self.tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self.tag_stack.append(tag)
        if tag == "p":
            self.in_p = True
            self.current_segments = []
        elif tag == "span":
            attr_dict = dict(attrs)
            if attr_dict.get("class") == "Wrterbuchkopfzeile":
                self.in_headword_span = True
        elif tag == "b":
            self.in_bold = True
        elif tag == "i":
            self.in_italic = True

    def handle_endtag(self, tag: str):
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        if tag == "p":
            if self.current_segments:
                self.paragraphs.append(self.current_segments)
            self.current_segments = None
            self.in_p = False
            self.in_headword_span = False
            self.in_bold = False
            self.in_italic = False
        elif tag == "span":
            # Check if we're leaving the headword span
            # Heuristic: headword span is always innermost, so any span close
            # while in_headword_span turns it off
            if self.in_headword_span:
                self.in_headword_span = False
        elif tag == "b":
            self.in_bold = False
        elif tag == "i":
            self.in_italic = False

    def handle_data(self, data: str):
        if not self.in_p or self.current_segments is None:
            return
        if self.in_headword_span:
            role = "headword"
        elif self.in_italic:
            role = "italic"
        elif self.in_bold:
            role = "bold"
        else:
            role = "standard"
        self.current_segments.append((data, role))

    def handle_entityref(self, name: str):
        char = html.unescape(f"&{name};")
        self.handle_data(char)

    def handle_charref(self, name: str):
        char = html.unescape(f"&#{name};")
        self.handle_data(char)


def parse_headword(segments: list[Segment]) -> str:
    """Extract the full English headword/phrase from paragraph segments.

    Handles three patterns:
    1. Direct: headword text followed by ": got."
    2. Suffix: headword + bold " --" + phrase (e.g., "approach -- approach (V.)")
    3. Prefix: headword "--" + bold phrase + rest (e.g., "-- apportion from")
    """
    headword_parts = [text for text, role in segments if role == "headword"]
    headword_text = "".join(headword_parts).strip()

    bold_parts = [text for text, role in segments if role == "bold"]
    bold_text = "".join(bold_parts).strip()

    # Collect standard text before "got." to capture sub-entry phrases
    standard_before_got = []
    for text, role in segments:
        if role == "standard":
            if "got." in text:
                standard_before_got.append(text[:text.index("got.")])
                break
            standard_before_got.append(text)
    pre_got_text = re.sub(r"\s+", " ", "".join(standard_before_got)).strip().rstrip(":").strip()

    if headword_text == "--":
        # Prefix pattern: "-- [bold phrase] rest"
        phrase = bold_text
        if pre_got_text:
            phrase = f"{phrase} {pre_got_text}"
        return phrase.strip().rstrip(":")
    elif bold_text == "--":
        # Suffix pattern: "headword -- phrase"
        if pre_got_text:
            return pre_got_text.strip().rstrip(":")
        return headword_text
    else:
        # Direct entry
        return headword_text


def clean_citation_form(
    raw: str,
) -> tuple[str, int | None, bool, bool]:
    """Extract citation form, frequency, reconstructed flag, uncertain flag.

    Args:
        raw: Raw text from an <i> tag.

    Returns:
        Tuple of (cleaned_form, frequency, reconstructed, uncertain).
    """
    # Collapse internal whitespace (HTML line breaks within <i> tags)
    text = re.sub(r"\s+", " ", raw).strip()

    # Handle malformed italic: truncate at ": nhd." if present
    if ": nhd." in text:
        text = text[: text.index(": nhd.")].strip()

    # Handle malformed italic: truncate at ", got." or ", got.:" if present
    if ", got." in text:
        text = text[: text.index(", got.")].strip()

    reconstructed = text.startswith("*")
    if reconstructed:
        text = text[1:]

    uncertain = text.endswith("?")
    if uncertain:
        text = text[:-1].rstrip()

    # Extract frequency: trailing number, possibly with = (e.g., "134=133")
    freq_match = re.search(r"\s+(\d+(?:=\d+)?)\s*$", text)
    frequency = None
    if freq_match:
        freq_str = freq_match.group(1)
        frequency = int(freq_str.split("=")[0])
        text = text[: freq_match.start()].strip()

    # Remove parenthetical notes like (sik), (sik du), (1), (2), (?)
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()

    # Remove trailing * or ?
    text = text.rstrip("*?").strip()

    return text, frequency, reconstructed, uncertain


def parse_gothic_body(segments: list[Segment]) -> list[GothicSubEntry]:
    """Parse Gothic sub-entries from the body segments of a dictionary entry.

    Splits the body at <i> (italic) boundaries. Each italic segment starts a new
    Gothic sub-entry, and the following standard text contains morphology, POS,
    and glosses.
    """
    # Find the first "got." in standard text to locate the start of Gothic entries
    body_start = None
    for i, (text, role) in enumerate(segments):
        if role in ("standard",) and "got." in text:
            body_start = i
            break
    if body_start is None:
        return []

    # Collect segments from body_start onward, grouping by italic boundaries
    # Each group: one italic segment followed by standard segments until next italic
    groups: list[tuple[str, str]] = []
    current_italic = None
    current_standard_parts: list[str] = []
    started = False

    for i, (text, role) in enumerate(segments):
        if i < body_start:
            continue

        # Skip text before the first "got."
        if not started:
            if role == "standard" and "got." in text:
                after_got = text[text.index("got.") + 4:]
                started = True
                if after_got.strip():
                    current_standard_parts.append(after_got)
            continue

        if role == "italic":
            # Save previous group if any
            if current_italic is not None:
                standard_text = " ".join(current_standard_parts)
                groups.append((current_italic, standard_text))
            current_italic = text
            current_standard_parts = []
        else:
            # Check for additional "got." markers within standard text
            # (some entries have "; got." between sub-entries without italic)
            current_standard_parts.append(text)

    # Save the last group
    if current_italic is not None:
        standard_text = " ".join(current_standard_parts)
        groups.append((current_italic, standard_text))

    entries = []
    for italic_text, standard_text in groups:
        entry = parse_single_gothic_entry(italic_text, standard_text)
        if entry is not None:
            entries.append(entry)

    return entries


def parse_single_gothic_entry(
    italic_text: str,
    standard_text: str,
) -> GothicSubEntry | None:
    """Parse a single Gothic sub-entry from its italic and standard text parts.

    Args:
        italic_text: Text from the <i> tag (citation form + maybe frequency).
        standard_text: Text following the </i> tag until the next sub-entry.
    """
    citation_form, frequency, reconstructed, uncertain = clean_citation_form(
        italic_text
    )
    if not citation_form:
        return None

    # Normalize whitespace in standard text
    body = re.sub(r"\s+", " ", standard_text).strip()

    # Strip leading comma
    body = body.lstrip(",").strip()

    # Extract morphology: the first comma-separated token with hyphens
    morphology = None
    morph_match = re.match(r"([a-zA-ZþðƕōēāīūáéíóúïëäöüÞÐ*?.\-]+),?\s*", body)
    if morph_match:
        candidate = morph_match.group(1).strip().rstrip(",")
        # Must contain at least one hyphen to be morphological breakdown
        if "-" in candidate:
            morphology = candidate.rstrip("*?").strip()
            body = body[morph_match.end():]

    # Extract POS: text before ": nhd." or before ": ne."
    pos = None
    german = None
    english_glosses = []

    # Try to find nhd. and ne. markers
    nhd_match = re.search(r":\s*nhd\.\s*", body)
    ne_match = re.search(r";\s*ne\.\s*", body)

    if nhd_match:
        # POS is everything before ": nhd."
        pos_text = body[: nhd_match.start()].strip()
        if pos_text:
            # Clean up POS: remove Crimean Gothic forms (e.g., "ita, krim")
            # and other dialect markers that precede the actual POS
            pos = pos_text.strip().rstrip(",").strip()

        # German gloss is between "nhd." and "; ne." (or end)
        german_start = nhd_match.end()
        if ne_match and ne_match.start() > german_start:
            german = body[german_start: ne_match.start()].strip().rstrip(";").strip()
        else:
            german = body[german_start:].strip().rstrip(";").strip()

    if ne_match:
        # English glosses are after "ne." until end of this sub-entry
        # Sub-entries are delimited by "; <next italic>" but we've already split
        eng_text = body[ne_match.end():].strip()
        # Remove trailing content that belongs to next sub-entry
        # (shouldn't happen since we split at italic boundaries, but be safe)
        eng_text = eng_text.rstrip(";").strip()
        if eng_text:
            english_glosses = split_glosses(eng_text)

    return GothicSubEntry(
        citation_form=citation_form,
        frequency=frequency,
        reconstructed=reconstructed,
        uncertain=uncertain,
        morphology=morphology,
        pos=pos,
        german=german,
        english_glosses=english_glosses,
    )


def split_glosses(text: str) -> list[str]:
    """Split a comma-separated gloss string into individual glosses.

    Handles parenthetical qualifiers like "catch (V.)" as single glosses.
    """
    glosses = []
    current = []
    paren_depth = 0

    for char in text:
        if char == "(":
            paren_depth += 1
            current.append(char)
        elif char == ")":
            paren_depth -= 1
            current.append(char)
        elif char == "," and paren_depth == 0:
            gloss = "".join(current).strip()
            if gloss:
                glosses.append(gloss)
            current = []
        else:
            current.append(char)

    # Don't forget the last gloss
    gloss = "".join(current).strip()
    if gloss:
        glosses.append(gloss)

    return glosses


def parse_dictionary(html_path: str) -> list[DictionaryEntry]:
    """Parse the full Koebler dictionary HTML into structured entries."""
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    parser = KoeblerHTMLParser()
    parser.feed(content)

    entries = []
    for segments in parser.paragraphs:
        # Skip paragraphs that don't contain dictionary entries
        has_headword = any(role == "headword" for _, role in segments)
        has_got = any("got." in text for text, role in segments if role == "standard")
        if not has_headword or not has_got:
            continue

        english = parse_headword(segments)
        gothic_entries = parse_gothic_body(segments)

        if english and gothic_entries:
            entries.append(DictionaryEntry(
                english=english,
                gothic_entries=gothic_entries,
            ))

    return entries


def print_statistics(entries: list[DictionaryEntry]):
    """Print summary statistics to stderr."""
    total_entries = len(entries)
    total_gothic = sum(len(e.gothic_entries) for e in entries)
    multi_gothic = sum(1 for e in entries if len(e.gothic_entries) > 1)
    reconstructed = sum(
        1 for e in entries for g in e.gothic_entries if g.reconstructed
    )
    uncertain = sum(
        1 for e in entries for g in e.gothic_entries if g.uncertain
    )
    no_english_glosses = sum(
        1 for e in entries for g in e.gothic_entries if not g.english_glosses
    )
    unique_citation_forms = len(set(
        g.citation_form for e in entries for g in e.gothic_entries
    ))

    print(f"Dictionary entries:       {total_entries}", file=sys.stderr)
    print(f"Gothic sub-entries:       {total_gothic}", file=sys.stderr)
    print(f"Unique citation forms:    {unique_citation_forms}", file=sys.stderr)
    print(f"Entries with multiple:    {multi_gothic}", file=sys.stderr)
    print(f"Reconstructed (*) forms:  {reconstructed}", file=sys.stderr)
    print(f"Uncertain (?) forms:      {uncertain}", file=sys.stderr)
    print(f"Missing English glosses:  {no_english_glosses}", file=sys.stderr)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Parse the Koebler Gothic-English dictionary from HTML to JSON.",
    )
    arg_parser.add_argument(
        "--input",
        default="data/gothic_dictionaries/koeblergerhard_dictionary_utf8.html",
        help="Path to the UTF-8 HTML dictionary file.",
    )
    arg_parser.add_argument(
        "--output",
        default="data/gothic_dictionaries/koebler_gothic_english.json",
        help="Path to write the output JSON.",
    )
    args = arg_parser.parse_args()

    print(f"Parsing {args.input}...", file=sys.stderr)
    entries = parse_dictionary(args.input)

    print_statistics(entries)

    output_data = [asdict(entry) for entry in entries]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(entries)} entries to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
