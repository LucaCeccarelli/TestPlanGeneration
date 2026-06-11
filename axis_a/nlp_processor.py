"""NLP annotation for Axis A.

Runs a spaCy pipeline over page text to:
- detect sentence boundaries,
- flag normative sentences (modal verbs with POS ``AUX`` or ``VERB``),
- detect section headers via regex.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import spacy

logger = logging.getLogger(__name__)

# Load the spaCy model once at module level.
nlp = spacy.load("en_core_web_sm")

# Modal verbs that mark a normative clause (SPEC SS4).
MODAL_TERMS: set[str] = {
    "shall",
    "must",
    "should",
    "may",
    "shall not",
    "must not",
    "need not",
}

# Single-token modal heads used for the POS-aware scan.
_MODAL_HEADS: set[str] = {"shall", "must", "should", "may", "need"}

# Section header regex, compiled once at module level (SPEC SS4).
SECTION_HEADER_RE: re.Pattern[str] = re.compile(r"^(\d+(?:\.\d+)+)\s+\S")


@dataclass
class AnnotatedSentence:
    """One sentence with its normative annotation."""

    text: str
    is_normative: bool
    modals: list[str] = field(default_factory=list)
    section: str = ""


def detect_section_header(line: str) -> str | None:
    """Return the section number (e.g. ``"7.2.1"``) if *line* is a header."""
    match = SECTION_HEADER_RE.match(line.strip())
    if match:
        return match.group(1)
    return None


def detect_modals(doc_or_text: "spacy.tokens.Doc | str") -> list[str]:
    """Return modal terms found in the text whose POS is ``AUX`` or ``VERB``.

    Multi-word negated forms (``shall not``, ``must not``, ``need not``) are
    returned as a single entry when the modal head is immediately followed by
    a negation token.
    """
    doc = nlp(doc_or_text) if isinstance(doc_or_text, str) else doc_or_text
    found: list[str] = []
    for i, token in enumerate(doc):
        lower = token.text.lower()
        if lower not in _MODAL_HEADS:
            continue
        if token.pos_ not in {"AUX", "VERB"}:
            continue
        # Check for a following negation: "shall not", "must not", "need not".
        modal = lower
        if i + 1 < len(doc) and doc[i + 1].text.lower() in {"not", "n't"}:
            candidate = f"{lower} not"
            if candidate in MODAL_TERMS:
                modal = candidate
        if modal not in MODAL_TERMS:
            continue
        if modal not in found:
            found.append(modal)
    return found


def annotate_text(text: str) -> list[AnnotatedSentence]:
    """Annotate *text*: sentence boundaries, normative flags, section tracking.

    Section headers update the running ``section`` attached to each following
    sentence.
    """
    annotated: list[AnnotatedSentence] = []
    current_section = ""

    # Track section headers line by line first, mapping char offsets -> section.
    section_at_offset: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        header = detect_section_header(line)
        if header is not None:
            section_at_offset.append((offset, header))
        offset += len(line) + 1  # +1 for the newline

    def _section_for(char_offset: int) -> str:
        result = ""
        for start, sec in section_at_offset:
            if start <= char_offset:
                result = sec
            else:
                break
        return result

    doc = nlp(text)
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        modals = detect_modals(sent.as_doc())
        current_section = _section_for(sent.start_char) or current_section
        annotated.append(
            AnnotatedSentence(
                text=sent_text,
                is_normative=bool(modals),
                modals=modals,
                section=current_section,
            )
        )
    logger.debug("Annotated %d sentences", len(annotated))
    return annotated
