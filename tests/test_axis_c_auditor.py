"""Tests for the Axis C Auditor verdict override logic.

The LLM and knowledge-base tool are fully mocked; no external services run.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from axis_b.schema import TestCase, TestInput
from axis_c.agents.auditor import compute_verdict, run_auditor


def _make_test_case() -> TestCase:
    return TestCase(
        tc_id="TC-18013-5-7.2.1-001",
        objective="Verify that the mDL does not transmit data before engagement.",
        priority="High",
        traceability="iso_iec_18013_5_0042 (section 7.2.1)",
        preconditions=["mDL is provisioned"],
        inputs=[
            TestInput(
                input_number=1,
                action="Request data before engagement.",
                expected_result="No data is transmitted.",
            ),
            TestInput(
                input_number=2,
                action="Engage, then request data.",
                expected_result="Data is transmitted.",
            ),
        ],
        actual_results="",
        requirement_type="SHALL",
        coverage_item_id="TCI-18013-5-7.2.1-001",
        feature_set="Device Engagement",
    )


def _run_auditor_with_mock_llm_output(audit_payload: dict) -> dict:
    """Run run_auditor with Agent, LLM and KB tool fully mocked."""
    mock_agent_instance = MagicMock()
    mock_agent_instance.run.return_value = json.dumps(audit_payload)

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "[section 7.2.1 | iso_iec_18013_5_0042]\nEvidence."

    with (
        patch("axis_c.agents.auditor.Agent", return_value=mock_agent_instance),
        patch("axis_c.agents.auditor.get_llm", return_value=MagicMock()),
        patch("axis_c.agents.auditor.search_norm_knowledge_base", mock_tool),
    ):
        return run_auditor(_make_test_case(), "The mDL shall not transmit data.")


def test_compute_verdict_hallucination_is_fail() -> None:
    audit = {"hallucinations": ["fake claim"], "contradictions": [], "omissions": []}
    assert compute_verdict(audit) == "FAIL"


def test_compute_verdict_contradiction_is_fail() -> None:
    audit = {"hallucinations": [], "contradictions": ["conflict"], "omissions": []}
    assert compute_verdict(audit) == "FAIL"


def test_compute_verdict_omission_is_warning() -> None:
    audit = {"hallucinations": [], "contradictions": [], "omissions": ["missing"]}
    assert compute_verdict(audit) == "WARNING"


def test_compute_verdict_clean_is_pass() -> None:
    audit = {"hallucinations": [], "contradictions": [], "omissions": []}
    assert compute_verdict(audit) == "PASS"


def test_verdict_overridden_to_fail_despite_llm_pass() -> None:
    """A hallucination must force FAIL even if the LLM claims PASS."""
    audit = _run_auditor_with_mock_llm_output(
        {
            "hallucinations": ["invented a biometric requirement"],
            "contradictions": [],
            "omissions": [],
            "verdict": "PASS",  # LLM verdict must be ignored
            "confidence": 0.99,
            "corrected_objective": None,
        }
    )
    assert audit["verdict"] == "FAIL"


def test_verdict_overridden_to_warning_for_omissions() -> None:
    audit = _run_auditor_with_mock_llm_output(
        {
            "hallucinations": [],
            "contradictions": [],
            "omissions": ["does not cover the retry behaviour"],
            "verdict": "PASS",
            "confidence": 0.8,
            "corrected_objective": None,
        }
    )
    assert audit["verdict"] == "WARNING"


def test_verdict_pass_when_audit_is_clean() -> None:
    audit = _run_auditor_with_mock_llm_output(
        {
            "hallucinations": [],
            "contradictions": [],
            "omissions": [],
            "verdict": "FAIL",  # even a spurious LLM FAIL is overridden
            "confidence": 0.9,
            "corrected_objective": None,
        }
    )
    assert audit["verdict"] == "PASS"
    assert audit["confidence"] == pytest.approx(0.9)
    assert audit["tc_id"] == "TC-18013-5-7.2.1-001"


def test_kb_verification_is_always_performed() -> None:
    """The objective and first expected result must be checked against the KB."""
    mock_agent_instance = MagicMock()
    mock_agent_instance.run.return_value = json.dumps(
        {
            "hallucinations": [],
            "contradictions": [],
            "omissions": [],
            "verdict": "PASS",
            "confidence": 1.0,
            "corrected_objective": None,
        }
    )
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "evidence"

    tc = _make_test_case()
    with (
        patch("axis_c.agents.auditor.Agent", return_value=mock_agent_instance),
        patch("axis_c.agents.auditor.get_llm", return_value=MagicMock()),
        patch("axis_c.agents.auditor.search_norm_knowledge_base", mock_tool),
    ):
        run_auditor(tc, "source clause")

    queries = [call.args[0] for call in mock_tool.invoke.call_args_list]
    assert tc.objective in queries
    assert tc.inputs[0].expected_result in queries
