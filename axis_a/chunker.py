"""Semantic chunking for Axis A.

Splits the page-marked norm text (produced by ``axis_a.pdf_extractor``) into
:class:`~axis_b.schema.RequirementChunk` objects using LangChain's
``RecursiveCharacterTextSplitter`` with section-aware separators.

``page_start`` is recovered by parsing the ``<!-- PAGE {n} -->`` markers that
``pdf_extractor`` injects into the concatenated text. ``is_normative`` and
``modals`` are set from a regex scan of each chunk.
"""

from __future__ import annotations

import logging
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from axis_b.schema import RequirementChunk

logger = logging.getLogger(__name__)

# Splitter configuration -- exact values from SPEC SS4.3.
SEPARATORS: list[str] = [
    r"\n(?=\d+(?:\.\d+)+\s)",  # section heading boundary
    "\n\n",
    "\n",
    ". ",
    " ",
]
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Page markers injected by pdf_extractor.pages_to_marked_text().
PAGE_MARKER_RE: re.Pattern[str] = re.compile(r"<!-- PAGE (\d+) -->")

# Section headers at the start of a line, e.g. "7.2.1 Device engagement".
SECTION_RE: re.Pattern[str] = re.compile(r"^(\d+(?:\.\d+)+)\s+\S", re.MULTILINE)

# Modal verb scan -- longest alternatives first so "shall not" wins over "shall".
MODAL_RE: re.Pattern[str] = re.compile(
    r"\b(shall not|must not|need not|shall|must|should|may)\b",
    re.IGNORECASE,
)


def slugify_norm(norm_name: str) -> str:
    """Turn a norm name into a filename-safe slug: ``ISO/IEC 18013-5`` -> ``iso_iec_18013_5``."""
    slug = re.sub(r"[^a-z0-9]+", "_", norm_name.lower())
    return slug.strip("_")


def scan_modals(text: str) -> list[str]:
    """Return the distinct modal terms found in *text* (lowercased, in order)."""
    found: list[str] = []
    for match in MODAL_RE.finditer(text):
        modal = match.group(1).lower()
        if modal not in found:
            found.append(modal)
    return found


def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        is_separator_regex=True,
        keep_separator=True,
        add_start_index=True,
    )


def _page_offsets(text: str) -> list[tuple[int, int]]:
    """Return sorted ``(char_offset, page_number)`` pairs for all page markers."""
    return [(m.start(), int(m.group(1))) for m in PAGE_MARKER_RE.finditer(text)]


def _section_offsets(text: str) -> list[tuple[int, str]]:
    """Return sorted ``(char_offset, section_number)`` pairs for all headers."""
    return [(m.start(), m.group(1)) for m in SECTION_RE.finditer(text)]


def _value_at(offsets: list[tuple[int, object]], position: int, default: object) -> object:
    """Return the value of the last offset entry at or before *position*."""
    result = default
    for start, value in offsets:
        if start <= position:
            result = value
        else:
            break
    return result


def chunk_document(text: str, source_norm: str) -> list[RequirementChunk]:
    """Split page-marked *text* into :class:`RequirementChunk` objects.

    Args:
        text: Concatenated norm text containing ``<!-- PAGE {n} -->`` markers
            (see ``axis_a.pdf_extractor.pages_to_marked_text``).
        source_norm: Norm name, e.g. ``"ISO/IEC 18013-5"``.

    Returns:
        List of chunks with ``chunk_id``, ``section``, ``page_start``,
        ``is_normative`` and ``modals`` populated.
    """
    slug = slugify_norm(source_norm)
    page_offsets = _page_offsets(text)
    section_offsets = _section_offsets(text)

    splitter = _build_splitter()
    documents = splitter.create_documents([text])

    chunks: list[RequirementChunk] = []
    index = 0
    for doc in documents:
        start_index = int(doc.metadata.get("start_index", 0))
        raw_chunk_text = doc.page_content

        # Strip page markers from the chunk text itself.
        clean_text = PAGE_MARKER_RE.sub("", raw_chunk_text).strip()
        if not clean_text:
            continue

        page_start = int(_value_at(page_offsets, start_index, 1))  # type: ignore[arg-type]

        # Prefer a section header inside the chunk; fall back to the last
        # header seen before the chunk start.
        inner_match = SECTION_RE.search(raw_chunk_text)
        if inner_match is not None and inner_match.start() < len(raw_chunk_text) // 2:
            section = inner_match.group(1)
        else:
            section = str(_value_at(section_offsets, start_index, ""))

        modals = scan_modals(clean_text)
        chunk = RequirementChunk(
            chunk_id=f"{slug}_{index:04d}",
            text=clean_text,
            section=section,
            page_start=page_start,
            is_normative=bool(modals),
            modals=modals,
            source_norm=source_norm,
        )
        chunks.append(chunk)
        index += 1

    logger.info(
        "Chunked document into %d chunks (%d normative)",
        len(chunks),
        sum(1 for c in chunks if c.is_normative),
    )
    return chunks
