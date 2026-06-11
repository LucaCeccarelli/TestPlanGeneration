"""Tests for axis_a.chunker.chunk_document on synthetic text."""

from __future__ import annotations

from axis_a.chunker import chunk_document, scan_modals, slugify_norm

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
