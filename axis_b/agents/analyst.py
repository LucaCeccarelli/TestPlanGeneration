"""Analyst agent (Axis B, agent 1 of 3).

Interprets a raw :class:`~axis_b.schema.RequirementChunk`: identifies what the
system must do, under what conditions, and what the testable assertion is.

CC — TraceLLM traceability enrichment:
    The system prompt is built dynamically per chunk so that it opens with an
    explicit traceability role header naming the chunk_id, section, and norm.
    This anchors every claim the Analyst makes to the specific clause it is
    processing, in line with the TraceLLM prompt-engineering approach.
"""

from __future__ import annotations

import json
import logging
import re

from axis_b.deepagents_compat import Agent
from axis_b.llm_setup import get_llm, search_norm_knowledge_base
from axis_b.schema import RequirementChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CC: Dynamic system prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BODY = """Use the search_norm_knowledge_base tool to look up related clauses when the
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


def _build_system_prompt(chunk: RequirementChunk) -> str:
    """Build a traceability-enriched system prompt for this specific chunk."""
    header = (
        f"You are the first agent in a three-stage traceability chain. "
        f"Your output will be cited in a formal test plan under ISO 29119-3. "
        f"Every claim you make must be directly derivable from the provided chunk "
        f"(chunk_id: {chunk.chunk_id}, section: {chunk.section}, "
        f"norm: {chunk.source_norm}). "
        f"Do not introduce information from general knowledge.\n\n"
        f"You are a senior requirements analyst for technical standards.\n"
        f"You receive one clause (chunk) of a normative document and must interpret it.\n"
    )
    return header + _SYSTEM_PROMPT_BODY


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
        system_prompt=_build_system_prompt(chunk),  # CC: dynamic per chunk
        tools=[search_norm_knowledge_base],
    )

    # Build the prompt, including context_refs and defined_terms if available (A3)
    context_block = ""
    if chunk.context_refs:
        refs_text = "\n".join(chunk.context_refs[:5])  # cap to avoid prompt overflow
        context_block += f"\n--- CROSS-REFERENCED CLAUSES ---\n{refs_text}\n"
    if chunk.defined_terms:
        terms_text = "\n".join(
            f"  {term}: {defn}" for term, defn in list(chunk.defined_terms.items())[:10]
        )
        context_block += f"\n--- RELEVANT DEFINED TERMS ---\n{terms_text}\n"

    prompt = (
        f"Analyse this clause from {chunk.source_norm}.\n"
        f"chunk_id: {chunk.chunk_id}\n"
        f"section: {chunk.section}\n"
        f"modal verbs found: {', '.join(chunk.modals) or 'none'}\n"
        f"is_full_clause: {chunk.is_full_clause}\n"
        f"--- CLAUSE TEXT ---\n{chunk.text}\n--- END CLAUSE TEXT ---"
        + context_block
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
