"""Retrieval evaluation for Axis A.

Compares the semantic chunking strategy against a ground-truth annotation set
(``data/ground_truth/annotated_requirements.jsonl``) and against a naive
``CharacterTextSplitter`` baseline.

Ground-truth JSONL format (one object per line)::

    {"key_phrase": "...", "section": "7.2.1", "query": "optional query text"}

If ``query`` is missing, ``key_phrase`` is used as the query. A retrieved
document is a *hit* if it contains the annotated key phrase
(case-insensitive).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

from axis_a.chunker import chunk_document, scan_modals, slugify_norm
from axis_a.indexer import chunks_to_documents, get_embeddings
from axis_b.schema import RequirementChunk

logger = logging.getLogger(__name__)

TOP_K = 5


@dataclass
class GroundTruthEntry:
    """One manually annotated requirement clause."""

    key_phrase: str
    section: str
    query: str


@dataclass
class EvaluationResult:
    """Precision / Recall / F1 for one retrieval configuration."""

    name: str
    precision: float
    recall: float
    f1: float
    n_queries: int


def load_ground_truth(jsonl_path: Path) -> list[GroundTruthEntry]:
    """Load ground-truth entries from a JSONL file."""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {jsonl_path}")
    entries: list[GroundTruthEntry] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append(
                GroundTruthEntry(
                    key_phrase=obj["key_phrase"],
                    section=obj.get("section", ""),
                    query=obj.get("query") or obj["key_phrase"],
                )
            )
    logger.info("Loaded %d ground-truth entries from %s", len(entries), jsonl_path)
    return entries


def evaluate_retrieval(
    vectorstore: FAISS,
    ground_truth: list[GroundTruthEntry],
    name: str = "semantic",
    top_k: int = TOP_K,
) -> EvaluationResult:
    """Query *vectorstore* for each ground-truth entry and compute metrics.

    - ``recall``: fraction of queries with at least one hit in the top-k.
    - ``precision``: fraction of all retrieved documents that are hits.
    - ``f1``: harmonic mean of the two.
    """
    if not ground_truth:
        raise ValueError("Ground truth is empty; cannot evaluate.")

    total_retrieved = 0
    total_relevant_retrieved = 0
    queries_with_hit = 0

    for entry in ground_truth:
        results = vectorstore.similarity_search(entry.query, k=top_k)
        total_retrieved += len(results)
        phrase = entry.key_phrase.lower()
        hits = sum(1 for doc in results if phrase in doc.page_content.lower())
        total_relevant_retrieved += hits
        if hits > 0:
            queries_with_hit += 1

    precision = total_relevant_retrieved / total_retrieved if total_retrieved else 0.0
    recall = queries_with_hit / len(ground_truth)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    result = EvaluationResult(
        name=name,
        precision=precision,
        recall=recall,
        f1=f1,
        n_queries=len(ground_truth),
    )
    logger.info(
        "[%s] P=%.3f R=%.3f F1=%.3f over %d queries",
        name, precision, recall, f1, len(ground_truth),
    )
    return result


def build_baseline_chunks(text: str, source_norm: str) -> list[RequirementChunk]:
    """Chunk *text* with a naive ``CharacterTextSplitter`` (baseline)."""
    splitter = CharacterTextSplitter(chunk_size=512, separator="\n", chunk_overlap=0)
    slug = slugify_norm(source_norm)
    chunks: list[RequirementChunk] = []
    for index, piece in enumerate(splitter.split_text(text)):
        clean = piece.strip()
        if not clean:
            continue
        modals = scan_modals(clean)
        chunks.append(
            RequirementChunk(
                chunk_id=f"{slug}_baseline_{index:04d}",
                text=clean,
                section="",
                page_start=1,
                is_normative=bool(modals),
                modals=modals,
                source_norm=source_norm,
            )
        )
    return chunks


def compare_with_baseline(
    text: str,
    source_norm: str,
    ground_truth: list[GroundTruthEntry],
    top_k: int = TOP_K,
) -> dict[str, EvaluationResult]:
    """Run the full experiment: semantic chunking vs naive baseline.

    Builds an in-memory FAISS index for each strategy from the same raw text
    and evaluates both against the same ground truth. Returns a dict with
    ``"semantic"``, ``"baseline"`` results and logs the F1 delta.
    """
    embeddings = get_embeddings()

    semantic_chunks = chunk_document(text, source_norm)
    semantic_store = FAISS.from_documents(chunks_to_documents(semantic_chunks), embeddings)
    semantic_result = evaluate_retrieval(semantic_store, ground_truth, "semantic", top_k)

    baseline_chunks = build_baseline_chunks(text, source_norm)
    baseline_store = FAISS.from_documents(chunks_to_documents(baseline_chunks), embeddings)
    baseline_result = evaluate_retrieval(baseline_store, ground_truth, "baseline", top_k)

    delta_f1 = semantic_result.f1 - baseline_result.f1
    logger.info(
        "F1 delta (semantic - baseline): %+.3f (semantic=%.3f, baseline=%.3f)",
        delta_f1, semantic_result.f1, baseline_result.f1,
    )
    return {"semantic": semantic_result, "baseline": baseline_result}
