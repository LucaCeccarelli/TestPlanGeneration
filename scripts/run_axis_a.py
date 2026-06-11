#!/usr/bin/env python3
"""Axis A runner -- extract requirements from a norm PDF and build the index.

Usage:
    python scripts/run_axis_a.py \\
        --pdf data/input/iso_18013_5.pdf \\
        --norm "ISO/IEC 18013-5" \\
        --output-chunks data/output/chunks/iso_18013_5_chunks.jsonl \\
        --output-index data/output/index/faiss_index
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable when running as `python scripts/run_axis_a.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Axis A: extract requirement chunks from a norm PDF and "
        "build a FAISS knowledge base."
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the norm PDF")
    parser.add_argument(
        "--norm", type=str, required=True, help='Norm name, e.g. "ISO/IEC 18013-5"'
    )
    parser.add_argument(
        "--output-chunks", type=Path, required=True,
        help="Output JSONL path for the requirement chunks",
    )
    parser.add_argument(
        "--output-index", type=Path, required=True,
        help="Output directory for the FAISS index",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    # spaCy model must be available before anything else runs.
    import spacy.util

    if not spacy.util.is_package("en_core_web_sm"):
        logger.error(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        )
        return 1

    try:
        from axis_a.chunker import PAGE_MARKER_RE, chunk_document
        from axis_a.indexer import build_index, save_chunks_jsonl
        from axis_a.nlp_processor import annotate_text
        from axis_a.pdf_extractor import extract_pages, pages_to_marked_text

        logger.info("Extracting pages from %s", args.pdf)
        marked_text = pages_to_marked_text(extract_pages(args.pdf))
        if not marked_text.strip():
            logger.error("No text extracted from %s", args.pdf)
            return 1

        # NLP annotation pass (sentence boundaries + normative sentence stats).
        plain_text = PAGE_MARKER_RE.sub("", marked_text)
        sentences = annotate_text(plain_text)
        normative_sentences = sum(1 for s in sentences if s.is_normative)
        logger.info(
            "NLP annotation: %d sentences, %d normative",
            len(sentences), normative_sentences,
        )

        chunks = chunk_document(marked_text, args.norm)
        if not chunks:
            logger.error("Chunking produced no chunks.")
            return 1

        save_chunks_jsonl(chunks, args.output_chunks)
        build_index(chunks, args.output_index)
    except Exception:
        logger.exception("Axis A pipeline failed")
        return 1

    print(f"[Axis A] Indexed {len(chunks)} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
