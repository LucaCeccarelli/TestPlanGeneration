"""Embedding, FAISS indexing, and BM25 indexing for Axis A.

Embeds :class:`~axis_b.schema.RequirementChunk` objects with
``OllamaEmbeddings(model="nomic-embed-text")``, builds a FAISS index, and
persists both the index and a JSONL serialization of all chunks so the
pipeline can reload everything without re-processing the PDF.

A2 — Hybrid retrieval:
    A BM25 index (``rank_bm25.BM25Okapi``) is built and persisted alongside
    the FAISS index so that ``axis_b.llm_setup`` can combine dense and sparse
    scores via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

from axis_b.schema import RequirementChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"
CHUNKS_JSONL_NAME = "chunks.jsonl"
BM25_INDEX_NAME = "bm25_index.pkl"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

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
                "is_full_clause": chunk.is_full_clause,
            },
        )
        for chunk in chunks
    ]


# ---------------------------------------------------------------------------
# JSONL serialisation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# A2 — BM25 index build / persist / load
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase whitespace tokenisation for BM25."""
    return text.lower().split()


def build_bm25_index(
    chunks: list[RequirementChunk],
) -> tuple[BM25Okapi, list[str]]:
    """Build a BM25Okapi index over *chunks*.

    Returns ``(bm25, chunk_id_list)`` where ``chunk_id_list[i]`` is the
    ``chunk_id`` for the i-th document in the corpus, enabling result mapping.
    """
    corpus = [_tokenise(chunk.text) for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    logger.info("Built BM25 index over %d documents", len(corpus))
    return bm25, chunk_ids


def save_bm25_index(
    bm25: BM25Okapi,
    chunk_ids: list[str],
    index_path: Path,
) -> None:
    """Persist the BM25 index and its chunk-id mapping to *index_path*."""
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    pkl_path = index_path / BM25_INDEX_NAME
    with pkl_path.open("wb") as fh:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, fh)
    logger.info("BM25 index saved to %s", pkl_path)


def load_bm25_index(
    index_path: Path,
) -> tuple[BM25Okapi, list[str]]:
    """Load a previously persisted BM25 index from *index_path*.

    Returns ``(bm25, chunk_id_list)``.

    Raises:
        FileNotFoundError: if the pickle file does not exist.
    """
    pkl_path = Path(index_path) / BM25_INDEX_NAME
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {pkl_path}. "
            "Run scripts/run_axis_a.py to rebuild the index."
        )
    with pkl_path.open("rb") as fh:
        data = pickle.load(fh)
    bm25: BM25Okapi = data["bm25"]
    chunk_ids: list[str] = data["chunk_ids"]
    logger.info("BM25 index loaded from %s (%d documents)", pkl_path, len(chunk_ids))
    return bm25, chunk_ids


# ---------------------------------------------------------------------------
# FAISS index build / load
# ---------------------------------------------------------------------------

def build_index(chunks: list[RequirementChunk], index_path: Path) -> FAISS:
    """Embed *chunks*, build a FAISS index, and persist everything to *index_path*.

    Persists:
    - FAISS binary index + docstore (``index.faiss``, ``index.pkl``)
    - JSONL chunk file (``chunks.jsonl``)
    - BM25 index (``bm25_index.pkl``)  ← A2
    """
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    documents = chunks_to_documents(chunks)
    embeddings = get_embeddings()
    logger.info("Embedding %d documents with %s ...", len(documents), EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(index_path))

    save_chunks_jsonl(chunks, index_path / CHUNKS_JSONL_NAME)

    # A2: build and persist the BM25 index alongside FAISS
    bm25, chunk_ids = build_bm25_index(chunks)
    save_bm25_index(bm25, chunk_ids, index_path)

    logger.info("FAISS + BM25 index saved to %s", index_path)
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
