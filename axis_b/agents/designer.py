"""Designer agent (Axis B, agent 3 of 3).

Writes a complete ISO 29119-3 test case from the RAG Router's context
package. The agent has no tools -- it works from context only. The final
output is validated against the :class:`~axis_b.schema.TestCase` model.
"""

from __future__ import annotations

import json
import logging
import re

from axis_b.deepagents_compat import Agent
from pydantic import ValidationError

from axis_b.llm_setup import get_llm
from axis_b.schema import TestCase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior test designer producing ISO/IEC/IEEE
29119-3 conformant test cases for technical standards.

You receive a context package: an analyst's interpretation of one normative
clause plus supporting clauses, definitions and test method hints retrieved
from the norm. Write ONE complete test case for the requirement.

Rules:
- "inputs" must contain AT LEAST 2 entries, numbered sequentially from 1.
- Each input is an object: {"input_number": <int>, "action": "<step>",
  "expected_result": "<observable outcome>"}.
- "actual_results" must be the empty string "" (filled during execution).
- "priority" is one of "High", "Medium", "Low" (SHALL requirements are
  usually "High", SHOULD "Medium", MAY "Low").
- "traceability" must reference the source chunk_id and section.
- "feature_set" is the functional area of the norm the clause belongs to
  (e.g. "Authentication", "Data Retrieval", "Security Mechanisms").

Respond ONLY with a raw JSON object (no markdown, no commentary) with exactly
these keys:
{
  "tc_id": "TC-<norm>-<section>-<seq>",
  "objective": "...",
  "priority": "High" | "Medium" | "Low",
  "traceability": "<chunk_id> (section <section>)",
  "preconditions": ["..."],
  "inputs": [{"input_number": 1, "action": "...", "expected_result": "..."}],
  "actual_results": "",
  "requirement_type": "SHALL" | "SHOULD" | "MAY",
  "coverage_item_id": "TCI-<norm>-<section>-<seq>",
  "feature_set": "...",
  "notes": ""
}"""

RETRY_INSTRUCTION = (
    "Your previous response was not valid JSON. Return only a raw JSON object."
)


def _strip_markdown_fences(raw: str) -> str:
    """Remove a surrounding ``` / ```json fence from the agent output."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())


def run_designer(router_output: dict) -> TestCase:
    """Run the Designer agent on the RAG Router's context package.

    Returns a validated :class:`TestCase`. Retries once on invalid JSON;
    raises :class:`ValueError` (carrying the chunk_id) on JSON failure after
    two attempts or on schema validation failure.
    """
    analyst_output = router_output.get("analyst_output", {})
    chunk_id = analyst_output.get("chunk_id", "<unknown>")
    section = analyst_output.get("section", "")

    agent = Agent(
        name="designer",
        llm=get_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=[],
    )

    prompt = (
        "Write one ISO 29119-3 test case from this context package.\n"
        f"--- CONTEXT PACKAGE ---\n{json.dumps(router_output, indent=2)}\n"
        "--- END CONTEXT PACKAGE ---"
    )

    last_error: Exception | None = None
    for attempt in range(2):
        raw = agent.run(prompt if attempt == 0 else f"{prompt}\n\n{RETRY_INSTRUCTION}")
        cleaned = _strip_markdown_fences(raw)
        try:
            data = json.loads(cleaned)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "Designer returned invalid JSON for %s (attempt %d/2)",
                chunk_id, attempt + 1,
            )
    else:
        raise ValueError(
            f"Designer agent failed to produce valid JSON for chunk "
            f"{chunk_id}: {last_error}"
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"Designer agent returned non-object JSON for chunk {chunk_id}"
        )

    # Enforce the execution-phase placeholder and the traceability contract
    # regardless of what the LLM produced.
    data["actual_results"] = ""
    traceability = str(data.get("traceability", ""))
    if chunk_id != "<unknown>" and chunk_id not in traceability:
        data["traceability"] = (
            f"{chunk_id} (section {section})" if section else chunk_id
        )
    data.setdefault("notes", "")

    try:
        test_case = TestCase.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Designer output for chunk {chunk_id} failed TestCase schema "
            f"validation: {exc}"
        ) from exc

    logger.debug("Designer completed test case %s for chunk %s", test_case.tc_id, chunk_id)
    return test_case
