"""Auditor agent (Axis C).

Verifies each generated :class:`~axis_b.schema.TestCase` against its source
:class:`~axis_b.schema.RequirementChunk` text and the norm knowledge base.
Detects hallucinations, contradictions and omissions; the final verdict is
computed deterministically in Python and overrides whatever the LLM said.
"""

from __future__ import annotations

import json
import logging
import re

from axis_b.deepagents_compat import Agent

from axis_b.llm_setup import get_llm, search_norm_knowledge_base
from axis_b.schema import TestCase

logger = logging.getLogger(__name__)

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


def _strip_markdown_fences(raw: str) -> str:
    """Remove a surrounding ``` / ```json fence from the agent output."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())


def compute_verdict(audit: dict) -> str:
    """Deterministic verdict logic -- overrides the LLM's own verdict."""
    if audit["hallucinations"]:
        verdict = "FAIL"
    elif audit["contradictions"]:
        verdict = "FAIL"
    elif audit["omissions"]:
        verdict = "WARNING"
    else:
        verdict = "PASS"
    return verdict


def run_auditor(test_case: TestCase, source_chunk_text: str) -> dict:
    """Audit one test case against its source requirement chunk.

    The objective and the first input's expected result are always verified
    against the norm knowledge base via ``search_norm_knowledge_base``; the
    retrieved evidence is injected into the agent prompt.

    Returns the audit dict with the verdict recomputed in Python.
    """
    # Mandatory knowledge-base verification of the objective and the first
    # expected result (AGENTS.md requirement) -- performed programmatically so
    # it cannot be skipped by the LLM.
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

    prompt = (
        "Audit this test case against its source clause.\n"
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

    # Override the LLM's verdict with the deterministic Python logic.
    audit["verdict"] = compute_verdict(audit)
    audit["tc_id"] = test_case.tc_id

    logger.debug("Audit of %s: verdict=%s", test_case.tc_id, audit["verdict"])
    return audit
