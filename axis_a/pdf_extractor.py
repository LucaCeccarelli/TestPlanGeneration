"""PDF extraction for Axis A.

Uses ``pdfplumber`` to extract text and tables from the source norm PDF.
Tables are extracted first; their bounding boxes are then filtered out of the
text flow so table content is not duplicated.

``extract_pages()`` is a generator -- the whole PDF is never held in memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pdfplumber

logger = logging.getLogger(__name__)

PAGE_MARKER_TEMPLATE = "\n\n<!-- PAGE {n} -->\n\n"


@dataclass
class RawPage:
    """One extracted PDF page: page number, text flow, and tables."""

    page_number: int
    text: str
    tables: list[list[list[str | None]]] = field(default_factory=list)


def _filter_out_table_bboxes(page: Any, table_bboxes: list[tuple]) -> Any:
    """Return a filtered page object with all words inside table bboxes removed."""

    def _outside_tables(obj: dict) -> bool:
        obj_mid_x = (obj["x0"] + obj["x1"]) / 2.0
        obj_mid_y = (obj["top"] + obj["bottom"]) / 2.0
        for x0, top, x1, bottom in table_bboxes:
            if x0 <= obj_mid_x <= x1 and top <= obj_mid_y <= bottom:
                return False
        return True

    return page.filter(_outside_tables)


def extract_pages(pdf_path: Path) -> Iterator[RawPage]:
    """Yield one :class:`RawPage` per page of *pdf_path*.

    Tables are extracted separately; their bounding boxes are removed from the
    text flow to avoid duplicate content.

    Raises:
        FileNotFoundError: if *pdf_path* does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info("Opening PDF: %s", pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_number = page.page_number
            try:
                # 1. Extract tables first.
                found_tables = page.find_tables()
                tables: list[list[list[str | None]]] = []
                table_bboxes: list[tuple] = []
                for table in found_tables:
                    extracted = table.extract()
                    if extracted:
                        tables.append(extracted)
                        table_bboxes.append(table.bbox)

                # 2. Extract text with table regions filtered out.
                if table_bboxes:
                    filtered_page = _filter_out_table_bboxes(page, table_bboxes)
                    text = filtered_page.extract_text() or ""
                else:
                    text = page.extract_text() or ""

                yield RawPage(page_number=page_number, text=text, tables=tables)
            except Exception:
                logger.exception("Failed to extract page %d; yielding empty page", page_number)
                yield RawPage(page_number=page_number, text="", tables=[])
            finally:
                # Free pdfplumber's per-page cache so memory stays flat.
                page.flush_cache()


def pages_to_marked_text(pages: Iterator[RawPage]) -> str:
    """Concatenate page texts, injecting page markers for downstream chunking.

    Each page is prefixed with ``\\n\\n<!-- PAGE {n} -->\\n\\n`` so that
    :func:`axis_a.chunker.chunk_document` can recover ``page_start`` for each
    chunk. Table rows are appended after the page text as tab-joined lines so
    their content remains searchable.
    """
    parts: list[str] = []
    for raw_page in pages:
        parts.append(PAGE_MARKER_TEMPLATE.format(n=raw_page.page_number))
        parts.append(raw_page.text)
        for table in raw_page.tables:
            rows = [
                "\t".join(cell if cell is not None else "" for cell in row)
                for row in table
            ]
            if rows:
                parts.append("\n" + "\n".join(rows))
    return "".join(parts)
