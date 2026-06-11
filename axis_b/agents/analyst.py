"""Analyst agent (Axis B, agent 1 of 3).

Interprets a raw :class:`~axis_b.schema.RequirementChunk`: identifies what the
system must do, under what conditions, and what the testable assertion is.
"""

from __future__ import annotations

import json
import logging
import re

from axis_b.deepagents_compat import Agent

from axis_b.llm_setup import get_llm, search_norm_knowledge_base
from axis_b.schema import RequirementChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior requirements analyst for technical standards.
You receive one clause (chunk) of a normative document and must interpret it.

Use the search_norm_knowledge_base tool to look up related clauses when the
chunk references other sections or uses terms defined elsewhere.

Respond ONLY with a raw JSON object (no markdown, no commentary) with exactly
these keys:
{
  "requirement_summary": "<one-sentence summary of what the system must do>",
  "testable_assertion": "<a single concrete, verifiable assertion>",
  "preconditions": ["<condition 1>", "..."],
  "related_sections": ["<section number>", "..."],
  "requirement_type": "SHALL" | "SHOULD" | "MAY",
  "chunk_id": "<the chunk id you were given>",
  "section": "<the section number you were given>"
}"""

RETRY_INSTRUCTION = (
    "Your previous response was not valid JSON. Return only a raw JSON object."
)

EXPECTED_KEYS = {
    "requirement_summary",
    "testable_assertion",
    "preconditions",
    "related_sections",
    "requirement_type",
    "chunk_id",
    "section",
}


def _strip_markdown_fences(raw: str) -> str:
    """Remove a surrounding ``` / ```json fence from the agent output."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())


def _derive_requirement_type(chunk: RequirementChunk) -> str:
    """Derive SHALL/SHOULD/MAY from the chunk's modal verbs."""
    modals = {m.split()[0] for m in chunk.modals}
    if modals & {"shall", "must"}:
        return "SHALL"
    if "should" in modals:
        return "SHOULD"
    if "may" in modals or "need" in modals:
        return "MAY"
    return "SHALL"


def run_analyst(chunk: RequirementChunk) -> dict:
    """Run the Analyst agent on one requirement chunk.

    Returns the parsed analysis dict. Retries once on invalid JSON; raises
    :class:`ValueError` (carrying the chunk_id) after two failures.
    """
    agent = Agent(
        name="analyst",
        llm=get_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=[search_norm_knowledge_base],
    )

    prompt = (
        f"Analyse this clause from {chunk.source_norm}.\n"
        f"chunk_id: {chunk.chunk_id}\n"
        f"section: {chunk.section}\n"
        f"modal verbs found: {', '.join(chunk.modals) or 'none'}\n"
        f"--- CLAUSE TEXT ---\n{chunk.text}\n--- END CLAUSE TEXT ---"
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
                "Analyst returned invalid JSON for %s (attempt %d/2)",
                chunk.chunk_id, attempt + 1,
            )
    else:
        raise ValueError(
            f"Analyst agent failed to produce valid JSON for chunk "
            f"{chunk.chunk_id}: {last_error}"
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"Analyst agent returned non-object JSON for chunk {chunk.chunk_id}"
        )

    # Enforce traceability fields and a valid requirement_type regardless of
    # what the LLM produced.
    data["chunk_id"] = chunk.chunk_id
    data["section"] = chunk.section
    if data.get("requirement_type") not in {"SHALL", "SHOULD", "MAY"}:
        data["requirement_type"] = _derive_requirement_type(chunk)
    data.setdefault("requirement_summary", chunk.text[:200])
    data.setdefault("testable_assertion", chunk.text[:200])
    data.setdefault("preconditions", [])
    data.setdefault("related_sections", [])

    missing = EXPECTED_KEYS - set(data)
    if missing:
        raise ValueError(
            f"Analyst output for chunk {chunk.chunk_id} missing keys: {sorted(missing)}"
        )
    logger.debug("Analyst completed for chunk %s", chunk.chunk_id)
    return data
