"""LLM factory and LangChain tools shared by all Axis B / Axis C agents.

The two ``@tool`` functions hold a module-level ``_vectorstore`` reference
which must be initialized exactly once via :func:`init_tools` before any
agent runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from axis_a.indexer import load_index

logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "llama3.2"

_vectorstore: FAISS | None = None


def get_llm(model: str = DEFAULT_CHAT_MODEL) -> ChatOllama:
    """Factory for the chat model (mockable in tests, swappable for sensitivity tests)."""
    return ChatOllama(model=model, temperature=0.2, num_ctx=4096)


def init_tools(index_path: Path) -> None:
    """Load the FAISS index built by Axis A and wire it into the tools.

    Must be called once (e.g. by ``axis_b.pipeline``) before any agent runs.

    Raises:
        FileNotFoundError: if the index does not exist on disk.
    """
    global _vectorstore
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/run_axis_a.py first to build the knowledge base."
        )
    _vectorstore = load_index(index_path)
    logger.info("Tools initialized with FAISS index at %s", index_path)


def _require_vectorstore() -> FAISS:
    if _vectorstore is None:
        raise RuntimeError(
            "Vectorstore not initialized. Call init_tools(index_path) before "
            "running any agent."
        )
    return _vectorstore


@tool
def search_norm_knowledge_base(query: str) -> str:
    """Search the norm knowledge base for clauses semantically related to the query.

    Returns the top 3 matching chunks of the source norm, each prefixed with
    its section number and chunk id.
    """
    vectorstore = _require_vectorstore()
    results = vectorstore.similarity_search(query, k=3)
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
