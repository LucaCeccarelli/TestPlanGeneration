"""LLM factory and LangChain tools shared by all Axis B / Axis C agents.

The two ``@tool`` functions hold module-level state initialised once via
:func:`init_tools` before any agent runs.

A2 — Hybrid retrieval:
    ``search_norm_knowledge_base`` now combines FAISS (dense) and BM25
    (sparse) results via Reciprocal Rank Fusion (RRF, k=60).  If the BM25
    index is not found on disk, the tool falls back to pure dense retrieval
    with a logged warning.

B3 — BERTScore self-evaluation:
    :func:`compute_bertscore_f1` wraps ``bert_score.score`` as a single
    mockable helper used by ``axis_b.agents.designer``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from axis_a.indexer import load_bm25_index, load_index

logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "llama3.2"

# B3
BERTSCORE_THRESHOLD: float = 0.80

# ---------------------------------------------------------------------------
# Module-level state (initialised by init_tools)
# ---------------------------------------------------------------------------

_vectorstore: FAISS | None = None
_bm25 = None          # BM25Okapi | None
_bm25_chunk_ids: list[str] = []


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def get_llm(model: str = DEFAULT_CHAT_MODEL) -> ChatOllama:
    """Factory for the chat model (mockable in tests, swappable for sensitivity tests)."""
    return ChatOllama(model=model, temperature=0.2, num_ctx=4096)


def init_tools(index_path: Path) -> None:
    """Load the FAISS (and optionally BM25) index and wire them into the tools.

    Must be called once before any agent runs.

    Raises:
        FileNotFoundError: if the FAISS index does not exist on disk.
    """
    global _vectorstore, _bm25, _bm25_chunk_ids
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/run_axis_a.py first to build the knowledge base."
        )
    _vectorstore = load_index(index_path)

    # A2: load BM25 index (optional — fall back gracefully if missing)
    try:
        _bm25, _bm25_chunk_ids = load_bm25_index(index_path)
        logger.info(
            "Tools initialised with FAISS + BM25 index at %s (%d BM25 docs)",
            index_path, len(_bm25_chunk_ids),
        )
    except FileNotFoundError:
        _bm25 = None
        _bm25_chunk_ids = []
        logger.warning(
            "BM25 index not found at %s — falling back to dense-only retrieval. "
            "Re-run scripts/run_axis_a.py to build the hybrid index.",
            index_path,
        )


def _require_vectorstore() -> FAISS:
    if _vectorstore is None:
        raise RuntimeError(
            "Vectorstore not initialized. Call init_tools(index_path) before "
            "running any agent."
        )
    return _vectorstore


# ---------------------------------------------------------------------------
# A2 — Reciprocal Rank Fusion helper
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense_ids: list[str],
    sparse_ids: list[str],
    k: int = 60,
) -> list[str]:
    """Return chunk_ids sorted by descending RRF score.

    RRF score for document d = Σ 1 / (k + rank(d, list))
    where rank is 1-based and missing documents score 0 from that list.
    """
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def search_norm_knowledge_base(query: str) -> str:
    """Search the norm knowledge base for clauses semantically related to the query.

    A2: Combines FAISS dense retrieval with BM25 sparse retrieval via
    Reciprocal Rank Fusion (RRF, k=60).  Falls back to dense-only when the
    BM25 index is unavailable.

    Returns the top 3 matching chunks of the source norm, each prefixed with
    its section number and chunk id.
    """
    vectorstore = _require_vectorstore()

    # --- Dense retrieval (top-10 candidates for RRF) ---
    dense_results = vectorstore.similarity_search(query, k=10)
    dense_ids = [doc.metadata.get("chunk_id", "") for doc in dense_results]
    dense_doc_map = {doc.metadata.get("chunk_id", ""): doc for doc in dense_results}

    if _bm25 is not None and _bm25_chunk_ids:
        # --- Sparse BM25 retrieval (top-10 candidates) ---
        import numpy as np  # transitive via faiss-cpu; kept local to avoid import overhead
        scores = _bm25.get_scores(query.lower().split())
        top_sparse_indices = np.argsort(scores)[::-1][:10].tolist()
        sparse_ids = [_bm25_chunk_ids[i] for i in top_sparse_indices if scores[i] > 0]

        # --- RRF fusion ---
        fused_ids = _rrf_fuse(dense_ids, sparse_ids)[:3]

        # Materialise the top-3 documents: prefer the FAISS doc map (has full
        # text); fall back to a docstore lookup for BM25-only results.
        results = []
        for cid in fused_ids:
            if cid in dense_doc_map:
                results.append(dense_doc_map[cid])
            else:
                # Try the FAISS docstore for BM25-only hits
                for doc in vectorstore.docstore._dict.values():
                    if doc.metadata.get("chunk_id") == cid:
                        results.append(doc)
                        break
    else:
        # Dense-only fallback
        results = dense_results[:3]

    if not results:
        return "No matching clauses found in the knowledge base."

    parts: list[str] = []
    for doc in results:
        section = doc.metadata.get("section", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        parts.append(f"[section {section} | {chunk_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@tool
def get_chunks_for_section(section_number: str) -> str:
    """Return all norm chunks whose section number starts with the given prefix.

    Example: section_number="7.2" returns chunks from 7.2, 7.2.1, 7.2.2, ...
    """
    vectorstore = _require_vectorstore()
    section_number = section_number.strip()
    matches: list[str] = []
    for doc in vectorstore.docstore._dict.values():
        section = str(doc.metadata.get("section", ""))
        if section.startswith(section_number):
            chunk_id = doc.metadata.get("chunk_id", "?")
            matches.append(f"[section {section} | {chunk_id}]\n{doc.page_content}")
    if not matches:
        return f"No chunks found for section prefix '{section_number}'."
    return "\n\n---\n\n".join(matches)


# ---------------------------------------------------------------------------
# B3 — BERTScore self-evaluation helper
# ---------------------------------------------------------------------------

def compute_bertscore_f1(text_a: str, text_b: str) -> float:
    """Compute BERTScore F1 between *text_a* and *text_b*.

    Uses ``bert_score.score`` with ``lang="en"``.  Returns a scalar float
    in [0, 1].  This function is isolated so tests can mock it with a single
    patch target (``axis_b.llm_setup.compute_bertscore_f1``).
    """
    from bert_score import score as bs_score  # local import to keep startup fast

    _, _, f1 = bs_score([text_a], [text_b], lang="en", verbose=False)
    return float(f1[0].item())
