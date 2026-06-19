"""Semantic chunking for Axis A.

Splits the page-marked norm text (produced by ``axis_a.pdf_extractor``) into
:class:`~axis_b.schema.RequirementChunk` objects using a two-pass strategy:

Pass 1 (A1 — structure-aware):
    Split the concatenated text exclusively on section-header boundaries so
    that each numbered clause is kept intact as a primary unit.  Short clauses
    (≤ CHUNK_SIZE characters) are emitted directly with ``is_full_clause=True``.

Pass 2 (A1 — size-bounded):
    Clauses that exceed CHUNK_SIZE are passed through
    ``RecursiveCharacterTextSplitter``; the resulting sub-chunks carry
    ``is_full_clause=False``.

Cross-reference resolution (A3):
    After all chunks are produced, ``resolve_cross_references()`` scans each
    chunk for patterns like "§ 7.2.1" / "clause 5" / "Table 3" and appends
    the first 150 characters of the referenced chunk as ``context_refs``.

Defined-terms injection (A3):
    ``inject_defined_terms()`` matches the norm's Terms-and-Definitions
    vocabulary against each chunk's text and populates ``defined_terms``.
    ``extract_definitions_section()`` parses §2 / §3 of the norm to build
    the vocabulary dict.

``page_start`` is recovered by parsing the ``<!-- PAGE {n} -->`` markers that
``pdf_extractor`` injects into the concatenated text.  ``is_normative`` and
``modals`` are set from a regex scan of each chunk.
"""

from __future__ import annotations

import logging
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from axis_b.schema import RequirementChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Splitter configuration — exact values from SPEC §4.3
# ---------------------------------------------------------------------------

SEPARATORS: list[str] = [
    r"\n(?=\d+(?:\.\d+)+\s)",  # section heading boundary
    "\n\n",
    "\n",
    ". ",
    " ",
]
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ---------------------------------------------------------------------------
# Compiled regexes — all at module level so they are compiled once
# ---------------------------------------------------------------------------

# Page markers injected by pdf_extractor.pages_to_marked_text().
PAGE_MARKER_RE: re.Pattern[str] = re.compile(r"<!-- PAGE (\d+) -->")

# Section headers at the start of a line, e.g. "7.2.1 Device engagement".
SECTION_RE: re.Pattern[str] = re.compile(r"^(\d+(?:\.\d+)+)\s+\S", re.MULTILINE)

# Same pattern used to split the full text into clause units (multiline).
_CLAUSE_SPLIT_RE: re.Pattern[str] = re.compile(
    r"(?m)^(\d+(?:\.\d+)+)\s+\S.*$"
)

# Modal verb scan — longest alternatives first so "shall not" beats "shall".
MODAL_RE: re.Pattern[str] = re.compile(
    r"\b(shall not|must not|need not|shall|must|should|may)\b",
    re.IGNORECASE,
)

# A3: Cross-reference pattern — §7.2.1 / clause 5 / section 4.3 / table 2 / annex B
CROSS_REF_RE: re.Pattern[str] = re.compile(
    r"(?:§\s*|(?:clause|section|table|annex)\s+)(\d[\d.]*)",
    re.IGNORECASE,
)

# A3: Terms-and-definitions section detector — matches "2 Terms…" or "3 Terms…"
_DEFINITIONS_SECTION_RE: re.Pattern[str] = re.compile(
    r"(?m)^\d+(?:\.\d+)?\s+Terms?\s+and\s+[Dd]efinitions?.*$"
)

# A3: Definition entry patterns:
#   "mobile driving licence (mDL): the digital representation …"
#   "mDL — the digital representation …"
_DEFINITION_ENTRY_RE: re.Pattern[str] = re.compile(
    r"^([A-Za-z][A-Za-z0-9 /()_-]{1,60}?)\s*(?::|—|–)\s*(.+)$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def slugify_norm(norm_name: str) -> str:
    """Turn a norm name into a filename-safe slug.

    Example: ``"ISO/IEC 18013-5"`` → ``"iso_iec_18013_5"``
    """
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


# ---------------------------------------------------------------------------
# A3: Definitions extraction and injection
# ---------------------------------------------------------------------------

def extract_definitions_section(text: str) -> dict[str, str]:
    """Parse the Terms-and-Definitions block (§2 or §3) from the norm text.

    Returns a ``{term: definition}`` dict.  Terms are lowercased for
    case-insensitive lookup.  Only the first 200 characters of each
    definition are kept to stay within metadata limits.
    """
    match = _DEFINITIONS_SECTION_RE.search(text)
    if not match:
        return {}

    # Find where the next top-level or second-level section begins after the
    # definitions heading so we don't parse beyond it.
    section_start = match.start()
    next_section = _CLAUSE_SPLIT_RE.search(text, match.end())
    section_end = next_section.start() if next_section else len(text)
    defs_block = text[section_start:section_end]

    definitions: dict[str, str] = {}
    for entry in _DEFINITION_ENTRY_RE.finditer(defs_block):
        term = entry.group(1).strip().lower()
        definition = entry.group(2).strip()[:200]
        if term and definition and len(term) > 1:
            definitions[term] = definition

    logger.debug("Extracted %d term definitions from norm text", len(definitions))
    return definitions


def inject_defined_terms(
    chunks: list[RequirementChunk],
    definitions: dict[str, str],
) -> list[RequirementChunk]:
    """Populate ``chunk.defined_terms`` with entries whose keys appear in the chunk.

    Only terms that literally appear (case-insensitive) in the chunk text are
    injected, keeping metadata lean.
    """
    if not definitions:
        return chunks

    updated: list[RequirementChunk] = []
    for chunk in chunks:
        text_lower = chunk.text.lower()
        matched = {
            term: defn
            for term, defn in definitions.items()
            if term in text_lower
        }
        if matched:
            updated.append(chunk.model_copy(update={"defined_terms": matched}))
        else:
            updated.append(chunk)
    return updated


# ---------------------------------------------------------------------------
# A3: Cross-reference resolution
# ---------------------------------------------------------------------------

def resolve_cross_references(
    chunks: list[RequirementChunk],
) -> list[RequirementChunk]:
    """Populate ``chunk.context_refs`` with excerpts of cross-referenced clauses.

    For each chunk, scans its text for patterns like ``§ 7.2.1``,
    ``clause 5``, ``table 3``, ``annex B``.  For each matched section number
    that corresponds to an existing chunk, the first 150 characters of that
    chunk's text are appended to ``context_refs``.
    """
    # Build a lookup from section prefix → first chunk text excerpt.
    section_map: dict[str, str] = {}
    for chunk in chunks:
        if chunk.section and chunk.section not in section_map:
            section_map[chunk.section] = chunk.text[:150]

    updated: list[RequirementChunk] = []
    for chunk in chunks:
        refs: list[str] = []
        seen: set[str] = set()
        for match in CROSS_REF_RE.finditer(chunk.text):
            ref_section = match.group(1).rstrip(".")
            if ref_section in seen or ref_section == chunk.section:
                continue
            seen.add(ref_section)
            # Try exact match first, then prefix match (e.g. "7.2" covers "7.2.1").
            excerpt = section_map.get(ref_section)
            if excerpt is None:
                for sec, exc in section_map.items():
                    if sec.startswith(ref_section + ".") or ref_section.startswith(sec + "."):
                        excerpt = exc
                        break
            if excerpt:
                refs.append(f"[§{ref_section}] {excerpt}")
        if refs:
            updated.append(chunk.model_copy(update={"context_refs": refs}))
        else:
            updated.append(chunk)
    return updated


# ---------------------------------------------------------------------------
# Internal helpers for chunk_document
# ---------------------------------------------------------------------------

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


def _value_at(
    offsets: list[tuple[int, object]],
    position: int,
    default: object,
) -> object:
    """Return the value of the last offset entry at or before *position*."""
    result = default
    for start, value in offsets:
        if start <= position:
            result = value
        else:
            break
    return result


def _split_into_clause_units(text: str) -> list[tuple[str, str, int]]:
    """Split *text* into ``(section_number, body_text, start_offset)`` triples.

    Each triple covers from a section header to just before the next one.
    A preamble block (text before the first header) is returned as
    ``("", preamble_text, 0)``.
    """
    matches = list(_CLAUSE_SPLIT_RE.finditer(text))
    if not matches:
        return [("", text, 0)]

    units: list[tuple[str, str, int]] = []

    preamble = text[: matches[0].start()]
    if preamble.strip():
        units.append(("", preamble, 0))

    for idx, match in enumerate(matches):
        section_number = match.group(1)
        body_start = match.start()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        units.append((section_number, body, body_start))

    return units


def _make_chunk(
    raw_text: str,
    slug: str,
    index: int,
    page_offsets: list[tuple[int, int]],
    section_offsets: list[tuple[int, str]],
    start_offset: int,
    is_full_clause: bool,
    source_norm: str,
    forced_section: str = "",
) -> RequirementChunk | None:
    """Construct one :class:`RequirementChunk` from a raw text segment.

    Returns ``None`` if the cleaned text is empty.
    """
    clean_text = PAGE_MARKER_RE.sub("", raw_text).strip()
    if not clean_text:
        return None

    page_start = int(_value_at(page_offsets, start_offset, 1))  # type: ignore[arg-type]

    if forced_section:
        section = forced_section
    else:
        inner_match = SECTION_RE.search(raw_text)
        if inner_match is not None and inner_match.start() < len(raw_text) // 2:
            section = inner_match.group(1)
        else:
            section = str(_value_at(section_offsets, start_offset, ""))

    modals = scan_modals(clean_text)
    return RequirementChunk(
        chunk_id=f"{slug}_{index:04d}",
        text=clean_text,
        section=section,
        page_start=page_start,
        is_normative=bool(modals),
        modals=modals,
        source_norm=source_norm,
        is_full_clause=is_full_clause,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(text: str, source_norm: str) -> list[RequirementChunk]:
    """Split page-marked *text* into :class:`RequirementChunk` objects.

    Uses a two-pass strategy (A1):
    - Pass 1: keeps each numbered clause intact when it fits within
      CHUNK_SIZE characters (``is_full_clause=True``).
    - Pass 2: applies ``RecursiveCharacterTextSplitter`` only to oversized
      clauses (``is_full_clause=False``).

    After splitting, applies cross-reference resolution (A3) but does NOT
    call ``inject_defined_terms`` here — that requires the caller to first
    run ``extract_definitions_section`` and pass the result in.  Use the
    pipeline-level helper ``chunk_document_full`` for the complete flow.

    Args:
        text: Concatenated norm text containing ``<!-- PAGE {n} -->`` markers.
        source_norm: Norm name, e.g. ``"ISO/IEC 18013-5"``.

    Returns:
        List of chunks with all fields populated (except ``defined_terms``).
    """
    slug = slugify_norm(source_norm)
    page_offsets_list = _page_offsets(text)
    section_offsets_list = _section_offsets(text)
    splitter = _build_splitter()

    clause_units = _split_into_clause_units(text)
    chunks: list[RequirementChunk] = []
    index = 0

    for section_number, body, body_start in clause_units:
        # Strip page markers from the body before measuring its length.
        body_clean = PAGE_MARKER_RE.sub("", body).strip()
        if not body_clean:
            continue

        if len(body_clean) <= CHUNK_SIZE:
            # --- Pass 1: keep the whole clause as a single chunk ---
            chunk = _make_chunk(
                raw_text=body,
                slug=slug,
                index=index,
                page_offsets=page_offsets_list,
                section_offsets=section_offsets_list,
                start_offset=body_start,
                is_full_clause=True,
                source_norm=source_norm,
                forced_section=section_number,
            )
            if chunk is not None:
                chunks.append(chunk)
                index += 1
        else:
            # --- Pass 2: split the oversized clause with the text splitter ---
            sub_docs = splitter.create_documents([body])
            for sub_doc in sub_docs:
                sub_start = body_start + int(sub_doc.metadata.get("start_index", 0))
                chunk = _make_chunk(
                    raw_text=sub_doc.page_content,
                    slug=slug,
                    index=index,
                    page_offsets=page_offsets_list,
                    section_offsets=section_offsets_list,
                    start_offset=sub_start,
                    is_full_clause=False,
                    source_norm=source_norm,
                    forced_section=section_number,
                )
                if chunk is not None:
                    chunks.append(chunk)
                    index += 1

    # A3: resolve cross-references now that all chunks are known
    chunks = resolve_cross_references(chunks)

    logger.info(
        "Chunked document into %d chunks (%d normative, %d full-clause)",
        len(chunks),
        sum(1 for c in chunks if c.is_normative),
        sum(1 for c in chunks if c.is_full_clause),
    )
    return chunks


def chunk_document_full(
    text: str,
    source_norm: str,
) -> list[RequirementChunk]:
    """Full pipeline wrapper: chunk + cross-refs + defined-terms injection.

    Extracts the Terms-and-Definitions section from *text*, then calls
    :func:`chunk_document` followed by :func:`inject_defined_terms`.

    Prefer this function in production scripts.  Use :func:`chunk_document`
    directly in tests that only need the chunking behaviour.
    """
    definitions = extract_definitions_section(text)
    chunks = chunk_document(text, source_norm)
    chunks = inject_defined_terms(chunks, definitions)
    return chunks
