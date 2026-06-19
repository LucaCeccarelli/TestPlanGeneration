"""RAG Router agent (Axis B, agent 2 of 3).

Enriches the Analyst's output with supporting context retrieved from the
norm knowledge base.

A2 — Minimum-evidence guard:
    Before constructing the LLM prompt, checks whether the hybrid retrieval
    returned at least one normative chunk for the analyst's testable
    assertion.  If not, falls back to ``get_chunks_for_section`` to guarantee
    the Designer always receives normative grounding.
"""

from __future__ import annotations

import json
import logging
import re

from axis_b.deepagents_compat import Agent
from axis_b.llm_setup import (
    get_chunks_for_section,
    get_llm,
    search_norm_knowledge_base,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the second agent in a three-stage traceability chain.
You are a retrieval specialist for technical standards. Your role is to enrich
the analyst's interpretation with supporting context drawn ONLY from the norm
knowledge base. Cite each retrieved chunk by its chunk_id.

Use your tools:
- search_norm_knowledge_base(query): semantic + keyword hybrid search over the norm.
- get_chunks_for_section(section_number): fetch all clauses of a section.

Look up the related sections the analyst listed, find definitions of technical
terms, and find clauses that hint at how the requirement can be verified.

Respond ONLY with a raw JSON object (no markdown, no commentary) with exactly
these keys:
{
  "supporting_clauses": ["<verbatim clause or summary>", "..."],
  "cross_norm_refs": ["<referenced external norm or section>", "..."],
  "definitions": {"<term>": "<definition>"},
  "test_method_hints": "<how this requirement could be verified>",
  "full_context_summary": "<concise summary of all gathered context>"
}"""

RETRY_INSTRUCTION = (
    "Your previous response was not valid JSON. Return only a raw JSON object."
)

_NORMATIVE_MARKER = re.compile(r"\b(SHALL|MUST|SHOULD|MAY)\b")


def _has_normative_content(retrieved_text: str) -> bool:
    """Return True if the retrieved text contains at least one normative modal."""
    return bool(_NORMATIVE_MARKER.search(retrieved_text))


def _strip_markdown_fences(raw: str) -> str:
    """Remove a surrounding ``` / ```json fence from the agent output."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())


def run_rag_router(analyst_output: dict) -> dict:
    """Run the RAG Router agent on the Analyst's output.

    A2: Before building the agent prompt, checks that the hybrid search
    returns normative content.  If not, invokes ``get_chunks_for_section``
    automatically and prepends the results as ``SECTION FALLBACK CONTEXT``.

    Returns the enriched context package (including ``analyst_output``).
    Retries once on invalid JSON; raises :class:`ValueError` (carrying the
    chunk_id) after two failures.
    """
    chunk_id = analyst_output.get("chunk_id", "<unknown>")
    section = analyst_output.get("section", "")
    testable_assertion = analyst_output.get("testable_assertion", "")

    # --- A2: minimum-evidence guard ---
    fallback_context = ""
    if testable_assertion:
        probe = search_norm_knowledge_base.invoke(testable_assertion)
        if not _has_normative_content(probe):
            logger.info(
                "No normative evidence for chunk %s assertion; "
                "falling back to section %s retrieval.",
                chunk_id, section,
            )
            if section:
                fallback_text = get_chunks_for_section.invoke(section)
                if fallback_text and "No chunks found" not in fallback_text:
                    fallback_context = (
                        f"\n\n--- SECTION FALLBACK CONTEXT (§{section}) ---\n"
                        f"{fallback_text}\n"
                        "--- END SECTION FALLBACK CONTEXT ---"
                    )

    agent = Agent(
        name="rag_router",
        llm=get_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=[search_norm_knowledge_base, get_chunks_for_section],
    )

    base_prompt = (
        "Gather supporting context for this analysed requirement.\n"
        f"--- ANALYST OUTPUT ---\n{json.dumps(analyst_output, indent=2)}\n"
        "--- END ANALYST OUTPUT ---"
        + fallback_context
    )

    last_error: Exception | None = None
    for attempt in range(2):
        raw = agent.run(
            base_prompt if attempt == 0 else f"{base_prompt}\n\n{RETRY_INSTRUCTION}"
        )
        cleaned = _strip_markdown_fences(raw)
        try:
            data = json.loads(cleaned)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "RAG Router returned invalid JSON for %s (attempt %d/2)",
                chunk_id, attempt + 1,
            )
    else:
        raise ValueError(
            f"RAG Router agent failed to produce valid JSON for chunk "
            f"{chunk_id}: {last_error}"
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"RAG Router agent returned non-object JSON for chunk {chunk_id}"
        )

    data.setdefault("supporting_clauses", [])
    data.setdefault("cross_norm_refs", [])
    data.setdefault("definitions", {})
    data.setdefault("test_method_hints", "")
    data.setdefault("full_context_summary", "")
    # The analyst output is attached programmatically so it is never lost or
    # mutated by the LLM.
    data["analyst_output"] = analyst_output

    logger.debug("RAG Router completed for chunk %s", chunk_id)
    return data
