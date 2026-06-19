"""Tests for axis_a.chunker.chunk_document on synthetic text."""

from __future__ import annotations

from axis_a.chunker import (
    chunk_document,
    extract_definitions_section,
    inject_defined_terms,
    resolve_cross_references,
    scan_modals,
    slugify_norm,
)

NORM = "ISO/IEC 18013-5"


def _padding(sentence: str, repeats: int) -> str:
    return " ".join([sentence] * repeats)


def test_single_chunk_section_and_modals() -> None:
    text = (
        "\n\n<!-- PAGE 1 -->\n\n"
        "7.1 General\n"
        "The mDL reader may request data elements from the mDL."
    )
    chunks = chunk_document(text, NORM)

    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk.section == "7.1"
    assert chunk.is_normative is True
    assert chunk.modals == ["may"]
    assert chunk.source_norm == NORM
    assert chunk.page_start == 1
    assert chunk.chunk_id == f"{slugify_norm(NORM)}_0000"
    assert "<!-- PAGE" not in chunk.text


def test_non_normative_chunk() -> None:
    text = (
        "\n\n<!-- PAGE 1 -->\n\n"
        "3.1 Definitions\n"
        "This clause provides definitions of terms used in this document."
    )
    chunks = chunk_document(text, NORM)

    assert len(chunks) >= 1
    assert chunks[0].is_normative is False
    assert chunks[0].modals == []


def test_section_boundary_split_and_page_tracking() -> None:
    filler_a = _padding("This sentence is informative filler text for clause one.", 8)
    filler_b = _padding("This sentence is informative filler text for clause two.", 8)
    text = (
        "\n\n<!-- PAGE 1 -->\n\n"
        f"7.1 General\n{filler_a}\n"
        "\n\n<!-- PAGE 2 -->\n\n"
        "7.2 Device engagement\n"
        "The mDL shall not transmit data before device engagement is complete. "
        f"The interface must support NFC. {filler_b}"
    )
    chunks = chunk_document(text, NORM)

    assert len(chunks) >= 2
    sections = {chunk.section for chunk in chunks}
    assert "7.1" in sections
    assert "7.2" in sections

    sec_71 = [c for c in chunks if c.section == "7.1"]
    sec_72 = [c for c in chunks if c.section == "7.2"]
    assert all(c.page_start == 1 for c in sec_71)
    assert all(c.page_start == 2 for c in sec_72)

    normative = [c for c in sec_72 if c.is_normative]
    assert normative, "section 7.2 must contain a normative chunk"
    modals = normative[0].modals
    assert "shall not" in modals
    assert "must" in modals
    assert "shall" not in modals  # "shall not" must not double-count as "shall"

    assert all(not c.is_normative for c in sec_71)


def test_chunk_ids_are_sequential_and_unique() -> None:
    filler = _padding("Informative sentence used purely as padding content here.", 30)
    text = f"\n\n<!-- PAGE 1 -->\n\n5.1 Scope\n{filler}"
    chunks = chunk_document(text, NORM)

    assert len(chunks) > 1
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    slug = slugify_norm(NORM)
    assert ids == [f"{slug}_{i:04d}" for i in range(len(ids))]


def test_scan_modals_orders_and_deduplicates() -> None:
    text = "It shall work. It shall work. It must not fail and need not log."
    assert scan_modals(text) == ["shall", "must not", "need not"]


# ---------------------------------------------------------------------------
# A1 — Two-pass chunking tests
# ---------------------------------------------------------------------------

def test_full_clause_flag_short_clause() -> None:
    """A short normative clause (< 512 chars) must emerge with is_full_clause=True."""
    text = (
        "\n\n<!-- PAGE 1 -->\n\n"
        "7.3 Short requirement\n"
        "The mDL SHALL present a valid credential to the reader upon request."
    )
    chunks = chunk_document(text, NORM)
    assert len(chunks) >= 1
    # The clause is well under 512 chars — must survive pass 1 intact
    normative = [c for c in chunks if c.is_normative]
    assert normative, "Expected at least one normative chunk"
    assert normative[0].is_full_clause is True


def test_full_clause_flag_long_clause() -> None:
    """A clause padded beyond 512 chars must be split (is_full_clause=False)."""
    long_filler = " ".join(["word"] * 200)  # well over 512 chars
    text = (
        "\n\n<!-- PAGE 1 -->\n\n"
        f"7.4 Long requirement\n"
        f"The mDL SHALL comply with the following rules. {long_filler}"
    )
    chunks = chunk_document(text, NORM)
    # At least one sub-chunk must have is_full_clause=False
    assert any(not c.is_full_clause for c in chunks), (
        "Expected at least one sub-chunk with is_full_clause=False for oversized clause"
    )


# ---------------------------------------------------------------------------
# A3 — Cross-reference resolution and defined-terms injection tests
# ---------------------------------------------------------------------------

def test_cross_reference_resolution() -> None:
    """A chunk referencing §7.1 must have its context_refs populated."""
    text = (
        "\n\n<!-- PAGE 1 -->\n\n"
        "7.1 Device engagement\n"
        "The mDL reader shall initiate the NFC connection.\n"
        "\n\n<!-- PAGE 2 -->\n\n"
        "7.2 Data retrieval\n"
        "The system shall comply with the requirements specified in section 7.1."
    )
    chunks = chunk_document(text, NORM)
    # The §7.2 chunk references §7.1 — its context_refs must be non-empty
    referencing = [c for c in chunks if c.section == "7.2"]
    assert referencing, "Expected a chunk for section 7.2"
    chunk_72 = referencing[0]
    assert chunk_72.context_refs, (
        "chunk_72.context_refs should contain an excerpt from §7.1"
    )
    assert any("7.1" in ref for ref in chunk_72.context_refs)


def test_defined_terms_injection() -> None:
    """Only terms that appear in the chunk text are injected into defined_terms."""
    from axis_b.schema import RequirementChunk

    chunk = RequirementChunk(
        chunk_id="test_0000",
        text="The mDL reader shall authenticate the holder via biometric verification.",
        section="7.1",
        page_start=1,
        is_normative=True,
        modals=["shall"],
        source_norm=NORM,
    )
    definitions = {
        "mdl reader": "a device that reads mobile driving licence data",
        "biometric verification": "identity confirmation using physiological traits",
        "issuer": "the organisation that issues the mDL",  # NOT in chunk text
    }
    updated = inject_defined_terms([chunk], definitions)
    assert len(updated) == 1
    dt = updated[0].defined_terms
    assert "mdl reader" in dt
    assert "biometric verification" in dt
    assert "issuer" not in dt, "'issuer' should not be injected — not in chunk text"
