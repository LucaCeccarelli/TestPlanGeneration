"""Embedding and FAISS indexing for Axis A.

Embeds :class:`~axis_b.schema.RequirementChunk` objects with
``OllamaEmbeddings(model="nomic-embed-text")``, builds a FAISS index, and
persists both the index and a JSONL serialization of all chunks so the
pipeline can reload everything without re-processing the PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from axis_b.schema import RequirementChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"
CHUNKS_JSONL_NAME = "chunks.jsonl"


def get_embeddings() -> OllamaEmbeddings:
    """Factory for the embedding model (mockable in tests)."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def chunks_to_documents(chunks: list[RequirementChunk]) -> list[Document]:
    """Convert chunks to LangChain documents with full metadata."""
    return [
        Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "page_start": chunk.page_start,
                "is_normative": chunk.is_normative,
                "modals": chunk.modals,
                "source_norm": chunk.source_norm,
            },
        )
        for chunk in chunks
    ]


def save_chunks_jsonl(chunks: list[RequirementChunk], jsonl_path: Path) -> None:
    """Serialize *chunks* to a JSONL file (one chunk per line)."""
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")
    logger.info("Wrote %d chunks to %s", len(chunks), jsonl_path)


def load_chunks_jsonl(jsonl_path: Path) -> list[RequirementChunk]:
    """Load chunks from a JSONL file produced by :func:`save_chunks_jsonl`."""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Chunks JSONL not found: {jsonl_path}")
    chunks: list[RequirementChunk] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(RequirementChunk.model_validate_json(line))
    logger.info("Loaded %d chunks from %s", len(chunks), jsonl_path)
    return chunks


def build_index(chunks: list[RequirementChunk], index_path: Path) -> FAISS:
    """Embed *chunks*, build a FAISS index, and persist it to *index_path*.

    A JSONL copy of all chunks is written alongside the index so it can be
    reloaded later without re-processing.
    """
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    documents = chunks_to_documents(chunks)
    embeddings = get_embeddings()
    logger.info("Embedding %d documents with %s ...", len(documents), EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(index_path))
    save_chunks_jsonl(chunks, index_path / CHUNKS_JSONL_NAME)
    logger.info("FAISS index saved to %s", index_path)
    return vectorstore


def load_index(index_path: Path) -> FAISS:
    """Load a previously persisted FAISS index from *index_path*.

    Raises:
        FileNotFoundError: if the index directory does not exist.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/run_axis_a.py first to build it."
        )
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded from %s", index_path)
    return vectorstore
