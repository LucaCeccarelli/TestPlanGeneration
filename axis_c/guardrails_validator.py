"""Structural validation for Axis C (cheap checks before the Auditor agent).

The mandatory ISO 29119-3 conformance checks are implemented as explicit
Python (per AGENTS.md). Guardrails AI is used only for additional string
length / format checks beyond these basics; if the Guardrails validators are
unavailable, an equivalent plain-Python fallback is applied.

Returns ``(True, [])`` on pass, ``(False, [violation, ...])`` on failure.
"""

from __future__ import annotations

import logging

from axis_b.schema import TestCase

logger = logging.getLogger(__name__)

VALID_PRIORITIES = {"High", "Medium", "Low"}
VALID_REQUIREMENT_TYPES = {"SHALL", "SHOULD", "MAY"}

# String-format bounds enforced via Guardrails (or the fallback).
OBJECTIVE_MIN_LEN = 10
OBJECTIVE_MAX_LEN = 500
TC_ID_MAX_LEN = 100


def _guardrails_string_checks(tc: TestCase) -> list[str]:
    """Length / format checks via Guardrails AI, with plain-Python fallback."""
    violations: list[str] = []
    try:
        from guardrails import Guard
        from guardrails.validators import ValidLength

        objective_guard = Guard.from_string(
            validators=[
                ValidLength(
                    min=OBJECTIVE_MIN_LEN, max=OBJECTIVE_MAX_LEN, on_fail="exception"
                )
            ],
            description="Test case objective length check",
        )
        try:
            objective_guard.parse(tc.objective)
        except Exception as exc:
            violations.append(
                f"objective fails length check "
                f"({OBJECTIVE_MIN_LEN}-{OBJECTIVE_MAX_LEN} chars): {exc}"
            )

        tc_id_guard = Guard.from_string(
            validators=[ValidLength(min=1, max=TC_ID_MAX_LEN, on_fail="exception")],
            description="Test case id length check",
        )
        try:
            tc_id_guard.parse(tc.tc_id)
        except Exception as exc:
            violations.append(f"tc_id fails length check (1-{TC_ID_MAX_LEN} chars): {exc}")
    except ImportError:
        logger.warning(
            "Guardrails validators unavailable; using plain-Python fallback "
            "for string length checks."
        )
        if not (OBJECTIVE_MIN_LEN <= len(tc.objective) <= OBJECTIVE_MAX_LEN):
            violations.append(
                f"objective length {len(tc.objective)} outside "
                f"[{OBJECTIVE_MIN_LEN}, {OBJECTIVE_MAX_LEN}]"
            )
        if not (1 <= len(tc.tc_id) <= TC_ID_MAX_LEN):
            violations.append(
                f"tc_id length {len(tc.tc_id)} outside [1, {TC_ID_MAX_LEN}]"
            )
    return violations


def validate_test_case(tc: TestCase) -> tuple[bool, list[str]]:
    """Run all structural checks on one test case.

    Mandatory checks (explicit Python, per AGENTS.md):
    - ``len(tc.inputs) >= 2``
    - ``tc.priority in {"High", "Medium", "Low"}``
    - ``tc.requirement_type in {"SHALL", "SHOULD", "MAY"}``
    - ``tc.tc_id``, ``tc.traceability``, ``tc.coverage_item_id`` non-empty

    Returns:
        ``(True, [])`` if all checks pass, otherwise
        ``(False, list_of_violation_strings)``.
    """
    violations: list[str] = []

    if len(tc.inputs) < 2:
        violations.append(
            f"inputs must contain at least 2 entries (got {len(tc.inputs)})"
        )

    if tc.priority not in VALID_PRIORITIES:
        violations.append(
            f"priority '{tc.priority}' not in {sorted(VALID_PRIORITIES)}"
        )

    if tc.requirement_type not in VALID_REQUIREMENT_TYPES:
        violations.append(
            f"requirement_type '{tc.requirement_type}' not in "
            f"{sorted(VALID_REQUIREMENT_TYPES)}"
        )

    for field_name in ("tc_id", "traceability", "coverage_item_id"):
        value = getattr(tc, field_name)
        if not isinstance(value, str) or not value.strip():
            violations.append(f"{field_name} must be a non-empty string")

    # SHALL fields from SPEC SS6: preconditions must also be present.
    if not tc.preconditions:
        violations.append("preconditions must contain at least one entry")

    violations.extend(_guardrails_string_checks(tc))

    is_valid = not violations
    if not is_valid:
        logger.info(
            "Test case %s failed structural validation: %s", tc.tc_id, violations
        )
    return is_valid, violations
