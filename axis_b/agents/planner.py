"""Planner agent (Axis B — B1).

Runs ONCE before the per-chunk generation loop.  Reads all normative chunks,
clusters them into feature-set groups using k-means over their FAISS
embeddings, and produces a plan skeleton that the pipeline passes to each
Designer call.

Why k-means over greedy cosine threshold
-----------------------------------------
K-means over FAISS embeddings is superior to a greedy threshold approach
because:
- It guarantees a globally coherent partition (no order-dependent drift).
- The number of clusters k is derived automatically from the document structure
  (number of distinct top-level section prefixes), so no hyperparameter tuning
  is needed for a new norm.
- FAISS provides a highly optimised k-means implementation (``faiss.Kmeans``)
  that scales to thousands of chunks without additional dependencies.
- The result is reproducible: same seed → same clusters, enabling fair
  comparison across LLM sensitivity runs.

Shared preconditions
---------------------
Within each cluster, sentences that (a) contain a normative modal AND (b) use
a setup verb ("establish", "provision", "configure", "initialise", "initialize",
"authenticate", "connect") are collected.  A sentence is promoted to
``shared_preconditions`` if it appears (by substring match) in ≥ 50% of the
chunks in the cluster, OR if the cluster has only one chunk (in which case all
normative setup sentences become shared preconditions by definition).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from langchain_ollama import OllamaEmbeddings

from axis_a.indexer import EMBEDDING_MODEL
from axis_b.schema import RequirementChunk

logger = logging.getLogger(__name__)

# Normative modals used for shared-precondition detection
_MODAL_RE = re.compile(r"\b(shall|must|should|may)\b", re.IGNORECASE)
# Setup verbs that signal a precondition sentence
_SETUP_VERB_RE = re.compile(
    r"\b(establish|provision|configure|initialise|initialize|"
    r"authenticate|connect|enable|activate)\b",
    re.IGNORECASE,
)
# Minimum fraction of cluster members a sentence must appear in to be "shared"
_SHARED_THRESHOLD = 0.5


def _get_embedder() -> OllamaEmbeddings:
    """Factory — mockable in tests."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def _embed_chunks(chunks: list[RequirementChunk]) -> np.ndarray:
    """Embed all chunks and return a float32 matrix of shape (n, dim)."""
    embedder = _get_embedder()
    texts = [c.text for c in chunks]
    vectors = embedder.embed_documents(texts)
    return np.array(vectors, dtype="float32")


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows in place and return the matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _infer_k(chunks: list[RequirementChunk]) -> int:
    """Derive k from the number of distinct top-level section prefixes.

    A top-level prefix is the first dotted component of the section number,
    e.g. "7" for "7.2.1".  We add one cluster per prefix present so that
    each major section becomes its own feature set.
    """
    prefixes: set[str] = set()
    for c in chunks:
        sec = c.section
        if sec:
            prefixes.add(sec.split(".")[0])
    k = max(2, len(prefixes))
    # Cap at min(k, n) to avoid more clusters than documents
    return min(k, len(chunks))


def _kmeans_cluster(matrix: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Run FAISS k-means and return the label array (shape: n,)."""
    import faiss  # available via faiss-cpu

    d = matrix.shape[1]
    km = faiss.Kmeans(d, k, niter=20, seed=seed, verbose=False)
    km.train(matrix)
    _, labels = km.index.search(matrix, 1)
    return labels.flatten()


def _extract_setup_sentences(text: str) -> list[str]:
    """Extract sentences that contain a normative modal AND a setup verb."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if sent and _MODAL_RE.search(sent) and _SETUP_VERB_RE.search(sent):
            result.append(sent)
    return result


def _derive_shared_preconditions(
    cluster_chunks: list[RequirementChunk],
) -> list[str]:
    """Return sentences that appear as setup preconditions across the cluster.

    A sentence qualifies when:
    - It contains a normative modal + a setup verb, AND
    - It appears (substring, case-insensitive) in ≥ _SHARED_THRESHOLD of the
      cluster members OR the cluster has only one member.
    """
    if not cluster_chunks:
        return []

    # Collect all candidate sentences from all chunks in the cluster
    all_candidates: list[str] = []
    for chunk in cluster_chunks:
        all_candidates.extend(_extract_setup_sentences(chunk.text))

    if not all_candidates:
        return []

    # Deduplicate while preserving insertion order
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for s in all_candidates:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(s)

    n = len(cluster_chunks)
    threshold = 1 if n == 1 else max(1, int(n * _SHARED_THRESHOLD))

    shared: list[str] = []
    for candidate in unique_candidates:
        lower_candidate = candidate.lower()
        count = sum(
            1 for chunk in cluster_chunks if lower_candidate in chunk.text.lower()
        )
        if count >= threshold:
            shared.append(candidate)

    return shared


def _label_cluster(
    cluster_chunks: list[RequirementChunk],
    cluster_idx: int,
) -> str:
    """Derive a human-readable feature-set name for a cluster.

    Uses the most common top-level section prefix in the cluster, falling
    back to a generic name if sections are absent.
    """
    from collections import Counter

    prefixes = [c.section.split(".")[0] for c in cluster_chunks if c.section]
    if not prefixes:
        return f"Feature_{cluster_idx + 1:02d}"
    top_prefix = Counter(prefixes).most_common(1)[0][0]
    # Map common ISO 18013-5 section numbers to meaningful names
    _SECTION_NAMES: dict[str, str] = {
        "4": "General_Requirements",
        "5": "Credential_Format",
        "6": "Device_Retrieval",
        "7": "Data_Model",
        "8": "Security",
        "9": "Privacy",
        "10": "Issuer_Infrastructure",
        "11": "Reader_Authentication",
        "12": "Device_Authentication",
    }
    name = _SECTION_NAMES.get(top_prefix, f"Section_{top_prefix}")
    return name


def run_planner(
    chunks: list[RequirementChunk],
    seed: int = 42,
) -> dict[str, dict]:
    """Cluster normative chunks and produce a plan skeleton.

    Args:
        chunks: All chunks (normative and non-normative); the function filters
            to normative ones internally.
        seed: Random seed for k-means reproducibility.

    Returns:
        A dict ``{feature_set_name: {"chunk_ids": [...],
        "shared_preconditions": [...], "section_range": "X.Y–X.Z"}}``

        If embedding fails (e.g. Ollama not running), returns an empty dict
        and logs an error — the pipeline must handle this gracefully.
    """
    normative = [c for c in chunks if c.is_normative]
    if not normative:
        logger.warning("run_planner: no normative chunks — returning empty skeleton.")
        return {}

    logger.info("Planner: embedding %d normative chunks for clustering …", len(normative))
    try:
        matrix = _embed_chunks(normative)
    except Exception as exc:
        logger.error("Planner: embedding failed (%s) — skipping plan skeleton.", exc)
        return {}

    matrix = _normalise(matrix)
    k = _infer_k(normative)
    logger.info("Planner: k-means with k=%d (seed=%d)", k, seed)

    labels = _kmeans_cluster(matrix, k, seed=seed)

    # Group chunks by cluster label
    cluster_map: dict[int, list[RequirementChunk]] = {}
    for chunk, label in zip(normative, labels.tolist()):
        cluster_map.setdefault(int(label), []).append(chunk)

    skeleton: dict[str, dict] = {}
    for cluster_idx, cluster_chunks in sorted(cluster_map.items()):
        fs_name = _label_cluster(cluster_chunks, cluster_idx)
        # Resolve name collisions
        if fs_name in skeleton:
            fs_name = f"{fs_name}_{cluster_idx}"

        sections = sorted(
            {c.section for c in cluster_chunks if c.section},
            key=lambda s: [int(x) for x in s.split(".") if x.isdigit()],
        )
        section_range = (
            f"{sections[0]}–{sections[-1]}" if len(sections) > 1
            else (sections[0] if sections else "")
        )

        skeleton[fs_name] = {
            "chunk_ids": [c.chunk_id for c in cluster_chunks],
            "shared_preconditions": _derive_shared_preconditions(cluster_chunks),
            "section_range": section_range,
        }
        logger.debug(
            "Planner: cluster '%s' — %d chunks, %d shared preconditions",
            fs_name, len(cluster_chunks),
            len(skeleton[fs_name]["shared_preconditions"]),
        )

    logger.info(
        "Planner: produced %d feature-set clusters from %d normative chunks",
        len(skeleton), len(normative),
    )
    return skeleton
