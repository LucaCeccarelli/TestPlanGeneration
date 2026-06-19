"""Axis B orchestrator.

Runs the Analyst -> RAG Router -> Designer chain over every normative
requirement chunk, collects the resulting test cases into the full ISO
29119-3 hierarchy, and writes the final :class:`~axis_b.schema.TestPlan`
to disk.

B1 — Planner agent:
    :func:`run_planner` is called once before the per-chunk loop.  It
    clusters all normative chunks into feature-set groups and identifies
    shared preconditions.  Each Designer call receives the cluster's
    ``feature_set_name`` and ``shared_preconditions`` as hints injected into
    the prompt via ``router_output``.

B3 — BERTScore gate:
    ``run_designer`` now accepts ``source_text`` (the raw chunk text) so it
    can score both candidates against the source requirement.
"""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from axis_a.chunker import slugify_norm
from axis_b.agents.analyst import run_analyst
from axis_b.agents.designer import run_designer
from axis_b.agents.planner import run_planner
from axis_b.agents.rag_router import run_rag_router
from axis_b.llm_setup import init_tools
from axis_b.schema import (
    FeatureSet,
    RequirementChunk,
    TestCase,
    TestCondition,
    TestCoverageItem,
    TestPlan,
)

logger = logging.getLogger(__name__)

_PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}


def _highest_priority(priorities: list[str]) -> str:
    """Return the highest priority in the list (High > Medium > Low)."""
    if not priorities:
        return "Medium"
    return min(priorities, key=lambda p: _PRIORITY_RANK.get(p, 1))


def _section_from_traceability(traceability: str) -> str:
    """Extract the section number from a 'chunk_id (section X.Y)' string."""
    import re

    match = re.search(r"section\s+([\d.]+)", traceability)
    return match.group(1).rstrip(".") if match else ""


def _find_cluster_for_chunk(
    chunk_id: str,
    plan_skeleton: dict[str, dict],
) -> tuple[str, list[str]]:
    """Return ``(feature_set_name, shared_preconditions)`` for this chunk_id.

    Falls back to ``("General", [])`` if no skeleton entry covers the chunk.
    """
    for fs_name, entry in plan_skeleton.items():
        if chunk_id in entry.get("chunk_ids", []):
            return fs_name, entry.get("shared_preconditions", [])
    return "General", []


def generate_test_cases(
    chunks: list[RequirementChunk],
    plan_skeleton: dict[str, dict] | None = None,
) -> tuple[list[TestCase], dict[str, str]]:
    """Run the three-agent chain over all normative chunks.

    B1: If *plan_skeleton* is provided, each chunk is looked up in the
    skeleton to obtain its ``feature_set_name`` and ``shared_preconditions``,
    which are injected into the ``router_output`` dict before the Designer
    call.

    Returns the generated test cases and a mapping ``tc_id -> source section``
    used for coverage reporting. A failure on one chunk is logged and skipped.
    """
    if plan_skeleton is None:
        plan_skeleton = {}

    normative_chunks = [c for c in chunks if c.is_normative]
    logger.info(
        "Processing %d normative chunks (of %d total)",
        len(normative_chunks), len(chunks),
    )

    test_cases: list[TestCase] = []
    tc_sections: dict[str, str] = {}
    for i, chunk in enumerate(normative_chunks, start=1):
        try:
            analyst_output = run_analyst(chunk)
            router_output = run_rag_router(analyst_output)

            # B1: inject planner hints into the router package
            if plan_skeleton:
                fs_name, shared_pcs = _find_cluster_for_chunk(
                    chunk.chunk_id, plan_skeleton
                )
                router_output["_planner_feature_set"] = fs_name
                router_output["_planner_shared_preconditions"] = shared_pcs

            # B3: pass source_text for BERTScore candidate selection
            test_case = run_designer(router_output, source_text=chunk.text)

            # B1: override feature_set with planner assignment when available
            if plan_skeleton:
                fs_name = router_output.get("_planner_feature_set", test_case.feature_set)
                if fs_name and fs_name != "General":
                    test_case = test_case.model_copy(update={"feature_set": fs_name})

            test_cases.append(test_case)
            tc_sections[test_case.tc_id] = chunk.section
            logger.info(
                "[%d/%d] chunk %s -> test case %s",
                i, len(normative_chunks), chunk.chunk_id, test_case.tc_id,
            )
        except Exception as e:
            logger.error(
                "[%d/%d] chunk %s failed, skipping: %s",
                i, len(normative_chunks), chunk.chunk_id, e,
            )
    return test_cases, tc_sections


def build_test_plan(
    test_cases: list[TestCase],
    tc_sections: dict[str, str],
    norm: str,
    plan_id: str | None = None,
) -> TestPlan:
    """Assemble test cases into the full ISO 29119-3 hierarchy.

    Test cases are grouped by their ``feature_set`` field into
    ``FeatureSet -> TestCondition -> TestCoverageItem`` structures.
    """
    slug = slugify_norm(norm).upper().replace("_", "-")
    if plan_id is None:
        plan_id = f"TP-{slug}-001"

    groups: dict[str, list[TestCase]] = defaultdict(list)
    for tc in test_cases:
        groups[tc.feature_set or "General"].append(tc)

    feature_sets: list[FeatureSet] = []
    for fs_index, (fs_name, fs_cases) in enumerate(sorted(groups.items()), start=1):
        fs_priority = _highest_priority([tc.priority for tc in fs_cases])
        fs_sections = sorted(
            {
                tc_sections.get(tc.tc_id) or _section_from_traceability(tc.traceability)
                for tc in fs_cases
            }
            - {""}
        )
        coverage_items = [
            TestCoverageItem(
                tci_id=tc.coverage_item_id,
                description=tc.objective,
                priority=tc.priority,
                traceability=tc.traceability,
            )
            for tc in fs_cases
        ]
        condition = TestCondition(
            tc_condition_id=f"TCOND-{slug}-{fs_index:03d}",
            description=f"Verifiable requirements of feature set '{fs_name}'",
            priority=fs_priority,
            traceability=(
                f"FS-{slug}-{fs_index:03d}; norm sections: "
                f"{', '.join(fs_sections) or 'n/a'}"
            ),
            coverage_items=coverage_items,
            test_cases=fs_cases,
        )
        feature_sets.append(
            FeatureSet(
                fs_id=f"FS-{slug}-{fs_index:03d}",
                objective=f"Verify conformance of '{fs_name}' requirements of {norm}",
                priority=fs_priority,
                traceability=f"{norm} sections: {', '.join(fs_sections) or 'n/a'}",
                test_conditions=[condition],
            )
        )

    coverage_sections = sorted(
        {section for section in tc_sections.values() if section}
    )
    return TestPlan(
        plan_id=plan_id,
        title=f"Test Plan for {norm}",
        norm_reference=norm,
        generation_date=datetime.date.today().isoformat(),
        test_scope=(
            f"All normative (SHALL/SHOULD/MAY) requirements of {norm} extracted "
            "by Axis A. Informative content is out of scope."
        ),
        assumptions=[
            "The system under test implements the norm as published.",
            "Test environment provides access to all interfaces named in the norm.",
            "Test cases are high-level plans; no executable code is included.",
        ],
        feature_sets=feature_sets,
        coverage_sections=coverage_sections,
    )


def _salvage_partial_plan(plan: TestPlan) -> TestPlan:
    """Rebuild the plan keeping only test cases that validate individually."""
    valid_cases: list[TestCase] = []
    sections: dict[str, str] = {}
    for fs in plan.feature_sets:
        for cond in fs.test_conditions:
            for tc in cond.test_cases:
                try:
                    valid = TestCase.model_validate(tc.model_dump())
                    valid_cases.append(valid)
                    sections[valid.tc_id] = _section_from_traceability(valid.traceability)
                except ValidationError as exc:
                    logger.error("Dropping invalid test case %s: %s", tc.tc_id, exc)
    return build_test_plan(valid_cases, sections, plan.norm_reference, plan.plan_id)


def run_pipeline(
    chunks: list[RequirementChunk],
    index_path: Path,
    norm: str,
    output_path: Path,
    plan_id: str | None = None,
) -> TestPlan:
    """Run the full Axis B pipeline and write the resulting plan to disk."""
    init_tools(Path(index_path))

    # B1: run the Planner once to produce the feature-set skeleton
    logger.info("Running Planner agent to build feature-set skeleton …")
    plan_skeleton = run_planner(chunks)
    if not plan_skeleton:
        logger.warning("Planner returned empty skeleton — proceeding without clustering.")

    test_cases, tc_sections = generate_test_cases(chunks, plan_skeleton=plan_skeleton)
    plan = build_test_plan(test_cases, tc_sections, norm, plan_id)

    # Validate the full plan before writing the research artifact.
    try:
        TestPlan.model_validate(plan.model_dump())
    except ValidationError as exc:
        logger.error("Full plan validation failed, writing partial plan: %s", exc)
        plan = _salvage_partial_plan(plan)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Wrote test plan %s with %d test cases to %s",
        plan.plan_id, len(test_cases), output_path,
    )
    return plan
