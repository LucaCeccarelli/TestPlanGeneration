"""PDF extraction for Axis A.

Uses ``pdfplumber`` to extract text and tables from the source norm PDF.
Tables are extracted first; their bounding boxes are then filtered out of the
text flow so table content is not duplicated.

``extract_pages()`` is a generator -- the whole PDF is never held in memory.

Noise reduction
---------------
* **Header/footer crop**: a fixed-height band is cropped from the top and bottom
  of every page before extraction, removing running headers and footers.
* **Spatial layout reconstruction**: ``extract_text(layout=True)`` is used so
  that pdfplumber reconstructs text from its 2-D position rather than following
  the raw character stream.  This prevents multi-column PDFs from interleaving
  two columns into a single garbled line (e.g. ``TDehvei dceevEnicgea...``).
  If ``layout=True`` returns an empty string the extractor falls back to the
  default stream order.
* **Page skipping**: callers can pass a set of 1-based page numbers to exclude
  (e.g. cover page, table of contents).  Skipped pages are emitted as empty
  ``RawPage`` objects so downstream page-marker offsets stay correct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pdfplumber

logger = logging.getLogger(__name__)

PAGE_MARKER_TEMPLATE = "\n\n<!-- PAGE {n} -->\n\n"

# Points to crop from the top (running header) and bottom (running footer) of
# every page.  Adjust these constants if the norm PDF uses wider bands.
HEADER_HEIGHT: float = 40.0
FOOTER_HEIGHT: float = 40.0


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


def _extract_text_layout(page: Any) -> str:
    """Extract text using spatial layout reconstruction.

    Tries ``extract_text(layout=True)`` first which reconstructs text from the
    2-D position of each character, preventing multi-column interleaving.  Falls
    back to the default stream order if the layout mode returns an empty string.
    """
    try:
        text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3) or ""
        if text.strip():
            return text
    except Exception:
        logger.debug("layout=True extraction failed on page, falling back to stream mode")
    return page.extract_text() or ""


def extract_pages(
    pdf_path: Path,
    skip_pages: set[int] | None = None,
) -> Iterator[RawPage]:
    """Yield one :class:`RawPage` per page of *pdf_path*.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file to extract.
    skip_pages:
        Optional set of 1-based page numbers to skip entirely.  Skipped pages
        are emitted as empty ``RawPage`` objects so that page-marker offsets
        computed by downstream code remain correct.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    skip: set[int] = skip_pages or set()

    logger.info("Opening PDF: %s (skipping pages: %s)", pdf_path, sorted(skip) or "none")
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_number = page.page_number
            try:
                # Honour explicit page-skip list (cover, TOC, etc.)
                if page_number in skip:
                    logger.debug("Skipping page %d (user-requested)", page_number)
                    yield RawPage(page_number=page_number, text="", tables=[])
                    continue

                # Crop running header and footer bands.
                crop_top = HEADER_HEIGHT
                crop_bottom = page.height - FOOTER_HEIGHT
                if crop_bottom > crop_top:
                    working_page = page.within_bbox(
                        (0, crop_top, page.width, crop_bottom)
                    )
                else:
                    # Page too short to crop — use as-is.
                    working_page = page

                # 1. Extract tables first.
                found_tables = working_page.find_tables()
                tables: list[list[list[str | None]]] = []
                table_bboxes: list[tuple] = []
                for table in found_tables:
                    extracted = table.extract()
                    if extracted:
                        tables.append(extracted)
                        table_bboxes.append(table.bbox)

                # 2. Extract text with table regions filtered out.
                if table_bboxes:
                    filtered_page = _filter_out_table_bboxes(working_page, table_bboxes)
                    text = _extract_text_layout(filtered_page)
                else:
                    text = _extract_text_layout(working_page)

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
