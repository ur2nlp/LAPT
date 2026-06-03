#!/usr/bin/env python3
"""
Parse Joseph Wright's Gothic glossary (Germanic Lexicon Project HTML) to JSON.

Wright's *Grammar of the Gothic Language* (1910) is in the US public domain, and
the Germanic Lexicon Project's proofread HTML of its glossary is in a "mature
state of correction" -- a legally and structurally clean source for a
Gothic->English lexicon, unlike the copyright-retained Koebler dictionary or the
uncorrected-OCR Balg scan.

Each glossary entry has the shape:

    <strong>headword,</strong> <em>POS.</em> gloss, gloss, <em>page refs</em>
    Cognate. cognate, ... <br>

This parser keeps the headword, POS, and English glosses (split into a list) and
discards page references, cognate/etymology runs, and grammar-section markers.
Special Gothic characters are encoded as <IMG ... chars/NAME.png> tags rather
than Unicode and are mapped back to Unicode here.

Input:  data/gothic_dictionaries/wright_gothic_glossary.html
Output: data/gothic_dictionaries/wright_gothic_english.json
"""

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser

from gothic.orthography import normalize_orthography

# Map the GLP character-image basenames to Unicode. Only the Gothic-relevant
# glyphs are mapped; Greek glyphs (alpha, nu, *-oxia, ...) occur solely in the
# etymology runs that this parser discards and are dropped (mapped to "").
GOTHIC_CHAR_IMAGES = {
    "thorn": "þ",
    "hw": "ƕ",
    "a-long": "ā",
    "e-long": "ē",
    "i-long": "ī",
    "o-long": "ō",
    "u-long": "ū",
    "u-long-short": "ū",
    "aelig-long": "ǣ",
    "z-tail": "z",
}

# Tokens that introduce a cognate/etymology run. When one of these appears in
# the gloss text (the no-page-ref entries keep gloss and etymology in a single
# untagged run), the gloss is cut at the first occurrence.
COGNATE_INTROS = [
    "OE.",
    "OHG.",
    "OS.",
    "O.Sax.",
    "O.Icel.",
    "O.Bulg.",
    "O.Slav.",
    "O.H.G.",
    "Goth.",
    "Gr.",
    "Lat.",
    "Skr.",
    "Lith.",
    "Idg.",
    "cp.",
    "cf.",
]
COGNATE_RE = re.compile(
    r"(?:^|\s)(?:" + "|".join(re.escape(intro) for intro in COGNATE_INTROS) + r")"
)

# Inflectional paradigm notes Wright appends after the gloss (e.g. "pl. nom.
# -eis", "gen. -ē", "superl. reikista"). They share the gloss's untagged run and
# get split in as pseudo-glosses; drop any gloss that leads with a case / number
# / degree abbreviation. (A leading bare ending like "-fold" is a real gloss, so
# this keys on the grammatical abbreviation, not the hyphen.)
MORPH_NOTE_RE = re.compile(
    r"^(?:pl|sg|du|nom|gen|acc|dat|voc|loc|instr|abl|masc|fem|neut|comp|superl)\.",
    re.IGNORECASE,
)

# Token kinds emitted by the parser: "strong", "em", "std" (untagged text), "br".
Token = tuple[str, str]


@dataclass
class WrightEntry:
    headword: str
    headword_segmented: str
    headword_normalized: str
    variant_forms: list[str] = field(default_factory=list)
    pos: str | None = None
    reconstructed: bool = False
    glosses: list[str] = field(default_factory=list)
    cross_reference: list[str] = field(default_factory=list)


class WrightHTMLParser(HTMLParser):
    """Flatten the glossary HTML into a stream of (kind, text) tokens.

    Character images are folded into the surrounding text run as their Unicode
    equivalents; unmapped image basenames are collected for reporting.
    """

    def __init__(self):
        super().__init__()
        self.tokens: list[Token] = []
        self.mode = "std"
        self.buffer: list[str] = []
        self.unmapped_images: set[str] = set()

    def _flush(self):
        text = "".join(self.buffer)
        if text:
            self.tokens.append((self.mode, text))
        self.buffer = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag in ("strong", "em"):
            self._flush()
            self.mode = tag
        elif tag == "img":
            source = dict(attrs).get("src", "") or ""
            match = re.search(r"chars/([A-Za-z0-9_-]+)\.png", source)
            if match:
                name = match.group(1)
                char = GOTHIC_CHAR_IMAGES.get(name)
                if char is None:
                    self.unmapped_images.add(name)
                    char = ""
                self.buffer.append(char)
        elif tag == "br":
            self._flush()
            self.tokens.append(("br", ""))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]):
        # <IMG .../> and <br/> variants route through the start-tag handler.
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str):
        if tag in ("strong", "em"):
            self._flush()
            self.mode = "std"

    def handle_data(self, data: str):
        self.buffer.append(data)

    def handle_entityref(self, name: str):
        self.buffer.append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str):
        self.buffer.append(html.unescape(f"&#{name};"))


def segment_entries(tokens: list[Token]) -> list[list[Token]]:
    """Split the token stream into per-entry token lists.

    A <strong> token starts a new entry only when the previous significant
    token was a <br> (or it is the first headword). In-gloss <strong> tokens
    (cross-references and inline Gothic examples) are preceded by gloss text
    instead and stay within the current entry. Whitespace-only text tokens do
    not count as significant, so the blank lines between <br> and a headword are
    transparent.
    """
    entries: list[list[Token]] = []
    current: list[Token] | None = None
    previous_kind: str | None = None

    for kind, text in tokens:
        is_headword = kind == "strong" and previous_kind in (None, "br")
        if is_headword:
            if current is not None:
                entries.append(current)
            current = [(kind, text)]
        elif current is not None:
            current.append((kind, text))

        # Whitespace-only std tokens are transparent for boundary detection.
        if not (kind == "std" and not text.strip()):
            previous_kind = kind

    if current is not None:
        entries.append(current)
    return entries


def clean_headword(text: str) -> str:
    """Strip surrounding whitespace and the trailing comma from a headword."""
    return re.sub(r"\s+", " ", text).strip().rstrip(",").strip()


def clean_pos(text: str) -> str | None:
    """Normalize a part-of-speech abbreviation (e.g. ``sv. VII,`` -> ``sv. VII``)."""
    pos = re.sub(r"\s+", " ", text).strip().rstrip(",").strip()
    return pos or None


def split_glosses(text: str) -> list[str]:
    """Split a gloss string on commas and semicolons into individual glosses.

    Parenthetical qualifiers (e.g. ``catch (with the hand)``) are kept whole.
    Empty fragments and inflectional paradigm notes (``pl. nom. -eis``) are
    dropped; cross-reference ``see`` markers are handled in ``collect_gloss``.
    """
    glosses: list[str] = []
    current: list[str] = []
    paren_depth = 0

    for char in text:
        if char == "(":
            paren_depth += 1
            current.append(char)
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
            current.append(char)
        elif char in ",;" and paren_depth == 0:
            gloss = "".join(current).strip()
            if gloss:
                glosses.append(gloss)
            current = []
        else:
            current.append(char)

    gloss = "".join(current).strip()
    if gloss:
        glosses.append(gloss)

    cleaned = []
    for gloss in glosses:
        gloss = gloss.strip().rstrip(".").strip()
        # Strip a leaked grammatical-category abbreviation: present/past
        # participles are glossed "part. one sowing", "part. not bearing", etc.
        gloss = re.sub(r"^part\.\s*", "", gloss).strip()
        if not gloss:
            continue
        # Drop inflectional paradigm notes ("pl. nom. -eis", "gen. -ē"). A lone
        # "see" is handled structurally in collect_gloss, not dropped here (the
        # verb "see" is a real gloss).
        if MORPH_NOTE_RE.match(gloss):
            continue
        cleaned.append(gloss)
    return cleaned


def collect_gloss(tokens: list[Token], start: int) -> tuple[str, list[str]]:
    """Collect gloss text and cross-references from an entry's body.

    Walks tokens from ``start`` (just after the POS marker) and stops at the
    first page-reference ``<em>``, ``<br>``, or in-gloss ``<strong>``. Standard
    text is also cut at the first cognate/etymology intro token. An in-gloss
    ``<strong>`` preceded by ``see`` yields cross-reference forms.

    Returns:
        A tuple of (gloss_text, cross_reference_forms).
    """
    parts: list[str] = []
    cross_reference: list[str] = []
    saw_see = False

    for index in range(start, len(tokens)):
        kind, text = tokens[index]
        if kind in ("em", "br"):
            break
        if kind == "strong":
            if saw_see:
                for form in text.split(","):
                    form = form.strip().rstrip(".").strip()
                    if form:
                        cross_reference.append(
                            normalize_orthography(form).replace("-", "")
                        )
            break
        # kind == "std"
        match = COGNATE_RE.search(text)
        segment = text[: match.start()] if match else text
        parts.append(segment)
        # Case-insensitive: cross-references can be sentence-initial ("See X").
        if re.search(r"\bsee\b", segment, re.IGNORECASE):
            saw_see = True
        if match:
            break

    gloss_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    # Only strip a trailing "see" when it actually precedes a captured
    # cross-reference; otherwise "see" is a real gloss (the verb), e.g. saíƕan
    # "to see, behold". Case-insensitive to catch a sentence-initial "See".
    if cross_reference:
        gloss_text = re.sub(
            r"[;,]?\s*see\s*$", "", gloss_text, flags=re.IGNORECASE
        ).strip()
    return gloss_text, cross_reference


def parse_entry(tokens: list[Token]) -> WrightEntry | None:
    """Build a structured entry from one entry's token list, or None if empty."""
    raw_headword = clean_headword(tokens[0][1])
    if not raw_headword:
        return None

    # A comma inside the headword separates alternate spellings of one lemma
    # (e.g. "apaústaúlus, apaúlstulus"), not distinct entries. The first is the
    # primary form; the rest become normalized variant forms.
    raw_forms = [form.strip() for form in raw_headword.split(",") if form.strip()]
    headword = raw_forms[0]
    reconstructed = headword.startswith("*")
    if reconstructed:
        headword = headword[1:].strip()
    if not headword:
        return None

    pos = None
    pos_index = None
    for index in range(1, len(tokens)):
        if tokens[index][0] == "em":
            pos = clean_pos(tokens[index][1])
            pos_index = index
            break

    glosses: list[str] = []
    cross_reference: list[str] = []
    if pos_index is not None:
        gloss_text, cross_reference = collect_gloss(tokens, pos_index + 1)
        glosses = split_glosses(gloss_text)

    # Wright separates derivational morphemes with hyphens (e.g. "af-airzjan");
    # the gotica corpus writes such forms solid. Keep the hyphenated form as the
    # morphological segmentation and strip hyphens for the surface form.
    segmented = normalize_orthography(headword)
    variant_forms = []
    for raw_form in raw_forms[1:]:
        form = raw_form.lstrip("*").strip()
        if form:
            variant_forms.append(normalize_orthography(form).replace("-", ""))

    return WrightEntry(
        headword=headword,
        headword_segmented=segmented,
        headword_normalized=segmented.replace("-", ""),
        variant_forms=variant_forms,
        pos=pos,
        reconstructed=reconstructed,
        glosses=glosses,
        cross_reference=cross_reference,
    )


def parse_glossary(html_path: str) -> tuple[list[WrightEntry], set[str]]:
    """Parse the Wright glossary HTML into structured entries.

    Returns:
        A tuple of (entries, unmapped_image_basenames).
    """
    with open(html_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    parser = WrightHTMLParser()
    parser.feed(content)

    entries = []
    for entry_tokens in segment_entries(parser.tokens):
        entry = parse_entry(entry_tokens)
        if entry is not None and (entry.glosses or entry.cross_reference):
            entries.append(entry)

    return entries, parser.unmapped_images


def print_statistics(entries: list[WrightEntry], unmapped_images: set[str]):
    """Print summary statistics and any unmapped image names to stderr."""
    total = len(entries)
    with_glosses = sum(1 for entry in entries if entry.glosses)
    xref_only = sum(
        1 for entry in entries if entry.cross_reference and not entry.glosses
    )
    reconstructed = sum(1 for entry in entries if entry.reconstructed)
    total_glosses = sum(len(entry.glosses) for entry in entries)

    print(f"Entries:                  {total}", file=sys.stderr)
    print(f"Entries with glosses:     {with_glosses}", file=sys.stderr)
    print(f"Cross-reference only:     {xref_only}", file=sys.stderr)
    print(f"Reconstructed (*) forms:  {reconstructed}", file=sys.stderr)
    print(f"Total glosses:            {total_glosses}", file=sys.stderr)
    if unmapped_images:
        names = ", ".join(sorted(unmapped_images))
        print(f"Unmapped char images:     {names}", file=sys.stderr)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Parse Wright's Gothic glossary (GLP HTML) into JSON.",
    )
    arg_parser.add_argument(
        "--input",
        default="data/gothic_dictionaries/wright_gothic_glossary.html",
        help="Path to the GLP Wright glossary HTML file.",
    )
    arg_parser.add_argument(
        "--output",
        default="data/gothic_dictionaries/wright_gothic_english.json",
        help="Path to write the output JSON.",
    )
    args = arg_parser.parse_args()

    print(f"Parsing {args.input}...", file=sys.stderr)
    entries, unmapped_images = parse_glossary(args.input)
    print_statistics(entries, unmapped_images)

    output_data = [asdict(entry) for entry in entries]
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(entries)} entries to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
