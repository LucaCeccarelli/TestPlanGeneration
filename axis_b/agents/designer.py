"""Designer agent (Axis B, agent 3 of 3).

Writes a complete ISO 29119-3 test case from the RAG Router's context
package.  The agent has no tools — it works from context only.

B2 — Schema-constrained generation with error-reflective retry:
    The full JSON Schema of ``TestCase`` is embedded in the system prompt so
    the LLM sees the exact structure it must produce.  On ``ValidationError``
    (valid JSON but wrong types / missing fields), the retry prompt names the
    specific failing field and its constraint rather than sending a generic
    "return raw JSON" message.  Up to 3 total attempts are made.

B3 — BERTScore self-evaluation gate:
    Two candidates are generated per chunk (temperature 0.2 and 0.5).
    BERTScore F1 between the candidates is computed; the better one (higher
    individual score against the source text) is selected.  Low-confidence
    outputs (F1 < BERTSCORE_THRESHOLD) are flagged in ``notes``.

CC — TraceLLM traceability footer:
    The system prompt closes with an explicit traceability anchor injected at
    call time (chunk_id and section), reinforcing the designer's role in the
    traceability chain.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from axis_b.deepagents_compat import Agent
from axis_b.llm_setup import BERTSCORE_THRESHOLD, compute_bertscore_f1, get_llm
from axis_b.schema import TestCase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# B2: embed the TestCase JSON Schema into the system prompt once at import time
# ---------------------------------------------------------------------------

_TC_SCHEMA_JSON = json.dumps(TestCase.model_json_schema(), indent=2)

_SYSTEM_PROMPT_TEMPLATE = (
    """You are the third and final agent in a three-stage traceability chain.
You are a senior test designer producing ISO/IEC/IEEE 29119-3 conformant test
cases for technical standards.

You receive a context package: an analyst's interpretation of one normative
clause plus supporting clauses, definitions and test method hints retrieved
from the norm. Write ONE complete test case for the requirement.

Rules:
- "inputs" must contain AT LEAST 2 entries, numbered sequentially from 1.
- Each input is an object: {{"input_number": <int>, "action": "<step>",
  "expected_result": "<observable outcome>"}}.
- "actual_results" must be the empty string "" (filled during execution).
- "priority" is one of "High", "Medium", "Low" (SHALL requirements are
  usually "High", SHOULD "Medium", MAY "Low").
- "requirement_type" must be one of "SHALL", "SHOULD", "MAY".
- "feature_set" is the functional area of the norm the clause belongs to
  (e.g. "Authentication", "Data Retrieval", "Security Mechanisms").

The JSON schema your output must exactly satisfy is:
{tc_schema}

Respond ONLY with a raw JSON object (no markdown, no commentary).

Traceability anchor — set the 'traceability' field to exactly '{chunk_id} """
    """(section {section})'. Do not generate a tc_id that does not correspond
to this chunk_id."""
)

RETRY_JSON_INSTRUCTION = (
    "Your previous response was not valid JSON. Return only a raw JSON object."
)


def _build_system_prompt(chunk_id: str, section: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        tc_schema=_TC_SCHEMA_JSON,
        chunk_id=chunk_id,
        section=section,
    )


def _strip_markdown_fences(raw: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())


def _build_validation_retry_instruction(exc: ValidationError) -> str:
    """Construct a targeted correction prompt from a Pydantic ValidationError.

    Names each failing field and its constraint so the LLM has a concrete
    signal rather than a generic "invalid JSON" message.
    """
    lines = ["Your previous response did not satisfy the schema. Fix only the following:"]
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"]) if error["loc"] else "?"
        msg = error["msg"]
        lines.append(f'  - Field "{field}": {msg}')
    lines.append("Return the corrected full JSON object.")
    return "\n".join(lines)


def _run_single_attempt(
    agent: "Agent",
    prompt: str,
    chunk_id: str,
    section: str,
) -> TestCase:
    """Run the agent for up to 3 attempts with error-reflective retry (B2).

    Attempt 1: standard call.
    Attempt 2 (on JSONDecodeError): generic "return raw JSON" retry.
    Attempt 3 (on ValidationError after attempt 1 or 2): field-targeted retry.
    """
    data: dict | None = None
    last_json_error: Exception | None = None
    last_val_error: ValidationError | None = None

    for attempt in range(3):
        if attempt == 0:
            current_prompt = prompt
        elif last_json_error is not None:
            current_prompt = f"{prompt}\n\n{RETRY_JSON_INSTRUCTION}"
            last_json_error = None
        elif last_val_error is not None:
            correction = _build_validation_retry_instruction(last_val_error)
            current_prompt = f"{prompt}\n\n{correction}"
            last_val_error = None
        else:
            break

        raw = agent.run(current_prompt)
        cleaned = _strip_markdown_fences(raw)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_json_error = exc
            logger.warning(
                "Designer returned invalid JSON for %s (attempt %d/3)",
                chunk_id, attempt + 1,
            )
            continue

        if not isinstance(data, dict):
            last_json_error = ValueError("non-object JSON")
            continue

        # Enforce execution placeholder and traceability contract
        data["actual_results"] = ""
        traceability = str(data.get("traceability", ""))
        if chunk_id != "<unknown>" and chunk_id not in traceability:
            data["traceability"] = (
                f"{chunk_id} (section {section})" if section else chunk_id
            )
        data.setdefault("notes", "")

        try:
            return TestCase.model_validate(data)
        except ValidationError as exc:
            last_val_error = exc
            logger.warning(
                "Designer output for %s failed schema validation (attempt %d/3): %s",
                chunk_id, attempt + 1, exc,
            )

    # Exhausted all attempts
    if last_val_error is not None:
        raise ValueError(
            f"Designer output for chunk {chunk_id} failed TestCase schema "
            f"validation after 3 attempts: {last_val_error}"
        ) from last_val_error
    raise ValueError(
        f"Designer agent failed to produce valid JSON for chunk "
        f"{chunk_id} after 3 attempts: {last_json_error}"
    )


def _make_agent(chunk_id: str, section: str) -> "Agent":
    return Agent(
        name="designer",
        llm=get_llm(),
        system_prompt=_build_system_prompt(chunk_id, section),
        tools=[],
    )


def _make_agent_with_temperature(chunk_id: str, section: str, temperature: float) -> "Agent":
    """Create a Designer agent with a custom temperature (for B3 dual-candidate)."""
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=get_llm().model, temperature=temperature, num_ctx=4096)
    return Agent(
        name="designer",
        llm=llm,
        system_prompt=_build_system_prompt(chunk_id, section),
        tools=[],
    )


def run_designer(
    router_output: dict,
    source_text: str = "",
) -> TestCase:
    """Run the Designer agent on the RAG Router's context package.

    B3: Generates two candidates (temperature 0.2 and 0.5), selects the one
    with the higher BERTScore F1 against *source_text*, and flags outputs
    whose inter-candidate F1 falls below BERTSCORE_THRESHOLD.

    Returns a validated :class:`TestCase`.
    """
    analyst_output = router_output.get("analyst_output", {})
    chunk_id = analyst_output.get("chunk_id", "<unknown>")
    section = analyst_output.get("section", "")

    prompt = (
        "Write one ISO 29119-3 test case from this context package.\n"
        f"--- CONTEXT PACKAGE ---\n{json.dumps(router_output, indent=2)}\n"
        "--- END CONTEXT PACKAGE ---"
    )

    # B3: generate two candidates with different temperatures
    agent_low = _make_agent_with_temperature(chunk_id, section, temperature=0.2)
    agent_high = _make_agent_with_temperature(chunk_id, section, temperature=0.5)

    candidate_a = _run_single_attempt(agent_low, prompt, chunk_id, section)
    candidate_b = _run_single_attempt(agent_high, prompt, chunk_id, section)

    # Compute inter-candidate BERTScore F1
    text_a = candidate_a.objective + " " + (
        candidate_a.inputs[0].expected_result if candidate_a.inputs else ""
    )
    text_b = candidate_b.objective + " " + (
        candidate_b.inputs[0].expected_result if candidate_b.inputs else ""
    )
    inter_f1 = compute_bertscore_f1(text_a, text_b)

    # Select the candidate with the higher individual BERTScore vs source text
    if source_text:
        score_a = compute_bertscore_f1(text_a, source_text)
        score_b = compute_bertscore_f1(text_b, source_text)
        chosen = candidate_a if score_a >= score_b else candidate_b
    else:
        chosen = candidate_a  # default when no source text provided

    # Flag low-confidence outputs
    if inter_f1 < BERTSCORE_THRESHOLD:
        note = f"low-confidence — BERTScore F1={inter_f1:.2f} — manual review recommended"
        existing_notes = chosen.notes.strip()
        updated_notes = f"{existing_notes}; {note}" if existing_notes else note
        chosen = chosen.model_copy(update={"notes": updated_notes})
        logger.info(
            "Designer: low BERTScore F1=%.3f for chunk %s — flagged in notes",
            inter_f1, chunk_id,
        )

    logger.debug(
        "Designer: selected test case %s for chunk %s (inter-F1=%.3f)",
        chosen.tc_id, chunk_id, inter_f1,
    )
    return chosen
