"""Auditor agent (Axis C).

Verifies each generated :class:`~axis_b.schema.TestCase` against its source
:class:`~axis_b.schema.RequirementChunk` text and the norm knowledge base.
Detects hallucinations, contradictions and omissions; the final verdict is
computed deterministically in Python and overrides whatever the LLM said.

C1 — Embedding-distance pre-audit:
    Before the LLM agent runs, :func:`compute_embedding_distance` embeds the
    test case objective and the source chunk text and returns their cosine
    distance.  When the distance exceeds ``EMBEDDING_DISTANCE_THRESHOLD``, a
    warning is prepended to the agent's prompt and ``pre_audit_flag=True`` is
    set in the audit report.  The distance is also recorded as
    ``embedding_distance`` for quantitative analysis.

CC — Traceability confidence:
    ``traceability_confidence`` (cosine similarity = 1 − distance) is added
    to every audit report as a float in [0, 1], making traceability quality
    measurable across LLM sensitivity runs.
"""

from __future__ import annotations

import logging
import json
import re

import numpy as np
from langchain_ollama import OllamaEmbeddings

from axis_a.indexer import EMBEDDING_MODEL
from axis_b.deepagents_compat import Agent
from axis_b.llm_setup import get_llm, search_norm_knowledge_base
from axis_b.schema import TestCase

logger = logging.getLogger(__name__)

# C1: cosine-distance threshold above which a pre-audit flag is raised
EMBEDDING_DISTANCE_THRESHOLD: float = 0.55

SYSTEM_PROMPT = """You are a meticulous quality auditor for generated test
plans. You receive one ISO 29119-3 test case, the exact source clause of the
norm it was generated from, and norm evidence retrieved from the knowledge
base.

Check the test case against the norm:
- hallucinations: claims in the test case with NO basis in the source clause
  or retrieved norm evidence.
- contradictions: statements that CONFLICT with the source clause or norm
  evidence (e.g. an expected_result that contradicts the requirement).
- omissions: aspects of the source clause the test case FAILS to cover.

You may use the search_norm_knowledge_base tool to retrieve more evidence.

Respond ONLY with a raw JSON object (no markdown, no commentary) with exactly
these keys:
{
  "hallucinations": ["<finding>", "..."],
  "contradictions": ["<finding>", "..."],
  "omissions": ["<finding>", "..."],
  "verdict": "PASS" | "FAIL" | "WARNING",
  "confidence": <float between 0 and 1>,
  "corrected_objective": "<better objective>" or null
}"""

RETRY_INSTRUCTION = (
    "Your previous response was not valid JSON. Return only a raw JSON object."
)


# ---------------------------------------------------------------------------
# C1: Embedding helpers
# ---------------------------------------------------------------------------

_embedder: OllamaEmbeddings | None = None


def _get_embedder() -> OllamaEmbeddings:
    """Factory for the embedding model (mockable in tests)."""
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _embedder


def compute_embedding_distance(
    tc: TestCase,
    source_chunk_text: str,
) -> float:
    """Compute cosine distance between the test case objective and the source chunk.

    Returns a float in [0, 2] (0 = identical, 2 = perfectly opposite).
    Practical values for semantically unrelated text are typically 0.4–0.8.
    """
    embedder = _get_embedder()
    vec_obj = np.array(embedder.embed_query(tc.objective), dtype="float64")
    vec_src = np.array(embedder.embed_query(source_chunk_text), dtype="float64")

    norm_obj = np.linalg.norm(vec_obj)
    norm_src = np.linalg.norm(vec_src)
    if norm_obj == 0 or norm_src == 0:
        return 1.0  # undefined similarity treated as maximum distance

    cosine_sim = float(np.dot(vec_obj, vec_src) / (norm_obj * norm_src))
    # Clamp to [-1, 1] to guard against floating-point drift
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    return 1.0 - cosine_sim


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _strip_markdown_fences(raw: str) -> str:
    """Remove a surrounding ``` / ```json fence from the agent output."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())


def compute_verdict(audit: dict) -> str:
    """Deterministic verdict logic — overrides the LLM's own verdict."""
    if audit["hallucinations"]:
        verdict = "FAIL"
    elif audit["contradictions"]:
        verdict = "FAIL"
    elif audit["omissions"]:
        verdict = "WARNING"
    else:
        verdict = "PASS"
    return verdict


# ---------------------------------------------------------------------------
# Main auditor function
# ---------------------------------------------------------------------------

def run_auditor(test_case: TestCase, source_chunk_text: str) -> dict:
    """Audit one test case against its source requirement chunk.

    C1: Computes embedding distance before the LLM call and injects a warning
    into the prompt when distance > EMBEDDING_DISTANCE_THRESHOLD.  Records
    ``pre_audit_flag`` and ``embedding_distance`` in the returned dict.

    CC: Adds ``traceability_confidence`` (cosine similarity) to the dict.

    The objective and the first input's expected result are always verified
    against the norm knowledge base (pre-LLM retrieval).

    Returns the audit dict with the verdict recomputed in Python.
    """
    # --- C1: pre-audit embedding distance ---
    try:
        distance = compute_embedding_distance(test_case, source_chunk_text)
    except Exception as exc:
        logger.warning("Embedding distance computation failed: %s — skipping pre-audit.", exc)
        distance = 0.0

    pre_audit_flag = distance > EMBEDDING_DISTANCE_THRESHOLD
    traceability_confidence = round(1.0 - distance, 4)

    # --- Mandatory knowledge-base retrieval (pre-LLM) ---
    objective_evidence = search_norm_knowledge_base.invoke(test_case.objective)
    if test_case.inputs:
        result_evidence = search_norm_knowledge_base.invoke(
            test_case.inputs[0].expected_result
        )
    else:
        result_evidence = "No inputs present in the test case."

    agent = Agent(
        name="auditor",
        llm=get_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=[search_norm_knowledge_base],
    )

    # C1: prepend embedding distance warning when relevant
    distance_warning = ""
    if pre_audit_flag:
        distance_warning = (
            f"WARNING: embedding distance between objective and source chunk is "
            f"{distance:.3f}, indicating possible semantic drift. "
            f"Scrutinise hallucinations carefully.\n\n"
        )

    prompt = (
        distance_warning
        + "Audit this test case against its source clause.\n"
        f"--- TEST CASE ---\n{test_case.model_dump_json(indent=2)}\n"
        f"--- SOURCE CLAUSE ---\n{source_chunk_text}\n"
        f"--- NORM EVIDENCE FOR OBJECTIVE ---\n{objective_evidence}\n"
        f"--- NORM EVIDENCE FOR FIRST EXPECTED RESULT ---\n{result_evidence}\n"
        "--- END ---"
    )

    last_error: Exception | None = None
    for attempt in range(2):
        raw = agent.run(prompt if attempt == 0 else f"{prompt}\n\n{RETRY_INSTRUCTION}")
        cleaned = _strip_markdown_fences(raw)
        try:
            audit = json.loads(cleaned)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "Auditor returned invalid JSON for %s (attempt %d/2)",
                test_case.tc_id, attempt + 1,
            )
    else:
        raise ValueError(
            f"Auditor agent failed to produce valid JSON for test case "
            f"{test_case.tc_id}: {last_error}"
        )

    if not isinstance(audit, dict):
        raise ValueError(
            f"Auditor agent returned non-object JSON for test case {test_case.tc_id}"
        )

    audit.setdefault("hallucinations", [])
    audit.setdefault("contradictions", [])
    audit.setdefault("omissions", [])
    audit.setdefault("corrected_objective", None)
    try:
        audit["confidence"] = float(audit.get("confidence", 0.0))
    except (TypeError, ValueError):
        audit["confidence"] = 0.0

    # Override the LLM's verdict with the deterministic Python logic
    audit["verdict"] = compute_verdict(audit)
    audit["tc_id"] = test_case.tc_id

    # C1: attach pre-audit signals
    audit["pre_audit_flag"] = pre_audit_flag
    audit["embedding_distance"] = round(distance, 4)

    # CC: traceability confidence
    audit["traceability_confidence"] = traceability_confidence

    logger.debug(
        "Audit of %s: verdict=%s, distance=%.3f, pre_audit_flag=%s",
        test_case.tc_id, audit["verdict"], distance, pre_audit_flag,
    )
    return audit
