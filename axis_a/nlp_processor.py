"""NLP annotation for Axis A.

Runs a spaCy pipeline over page text to:
- detect sentence boundaries,
- flag normative sentences (modal verbs with POS ``AUX`` or ``VERB``),
- detect section headers via regex.

Subsection-scoped processing
-----------------------------
``annotate_text()`` splits the input into subsections before calling spaCy so
that each ``nlp()`` call operates on a single, self-contained subsection (the
text between two numbered section headers such as "7.2.1").

Benefits over running ``nlp()`` on the entire document:
* No sentence ever straddles two different sections -- every
  :class:`AnnotatedSentence` belongs to exactly one section.
* Each spaCy call is short (a few hundred to a few thousand characters),
  staying well within the default 1 M character limit and processing faster.
* Section labels are determined structurally (which subsection block the text
  belongs to) rather than heuristically (nearest prior header in char-offset
  space).

The external interface is unchanged: callers still call
``annotate_text(text)`` and receive a flat list of :class:`AnnotatedSentence`.
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
# Matches lines that start with a dotted number such as "7.2.1 Title text".
SECTION_HEADER_RE: re.Pattern[str] = re.compile(r"^(\d+(?:\.\d+)+)\s+\S")

# Same pattern but as a multiline search for splitting the full document.
_SECTION_SPLIT_RE: re.Pattern[str] = re.compile(
    r"(?m)^(\d+(?:\.\d+)+)\s+\S.*$"
)


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


def _split_into_subsections(text: str) -> list[tuple[str, str]]:
    """Split *text* into ``(section_number, body_text)`` pairs.

    The preamble (any text before the first numbered section header) is
    returned as ``("", preamble_text)``.  Each subsequent subsection starts at
    its header line and ends just before the next header.

    Returns
    -------
    list of (section_number, body_text) tuples
    """
    matches = list(_SECTION_SPLIT_RE.finditer(text))
    if not matches:
        # No section headers at all -- treat the whole text as one block.
        return [("", text)]

    subsections: list[tuple[str, str]] = []

    # Preamble: text before the first header.
    preamble = text[: matches[0].start()]
    if preamble.strip():
        subsections.append(("", preamble))

    for idx, match in enumerate(matches):
        section_number = match.group(1)
        body_start = match.start()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        subsections.append((section_number, body))

    return subsections


def annotate_text(text: str) -> list[AnnotatedSentence]:
    """Annotate *text*: sentence boundaries, normative flags, section tracking.

    The text is first split into subsections at numbered section headers (e.g.
    ``"7.2.1 Title"``).  spaCy is then run independently on each subsection so
    that sentences never straddle section boundaries and each call to ``nlp()``
    is short.  All resulting :class:`AnnotatedSentence` objects carry the
    section number of the subsection they came from.
    """
    annotated: list[AnnotatedSentence] = []

    subsections = _split_into_subsections(text)
    logger.debug("Annotating %d subsections", len(subsections))

    for section_number, body in subsections:
        if not body.strip():
            continue

        doc = nlp(body)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            modals = detect_modals(sent.as_doc())
            annotated.append(
                AnnotatedSentence(
                    text=sent_text,
                    is_normative=bool(modals),
                    modals=modals,
                    section=section_number,
                )
            )

    logger.debug("Annotated %d sentences across %d subsections", len(annotated), len(subsections))
    return annotated
