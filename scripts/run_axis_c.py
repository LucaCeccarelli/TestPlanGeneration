#!/usr/bin/env python3
"""Axis C runner -- validate a generated test plan against the source norm.

Usage:
    python scripts/run_axis_c.py \\
        --plan data/output/test_plans/TP-ISO-IEC-18013-5-001.json \\
        --chunks data/output/chunks/iso_18013_5_chunks.jsonl \\
        --index data/output/index/faiss_index
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the project root importable when running as `python scripts/run_axis_c.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Axis C: structural + agentic validation of a generated "
        "test plan (hallucination / contradiction / omission detection)."
    )
    parser.add_argument(
        "--plan", type=Path, required=True, help="TestPlan JSON (Axis B output)"
    )
    parser.add_argument(
        "--chunks", type=Path, required=True,
        help="JSONL file of RequirementChunk objects (Axis A output)",
    )
    parser.add_argument(
        "--index", type=Path, required=True,
        help="Directory of the FAISS index (Axis A output)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for the audit report JSON "
        "(default: <plan-dir>/<plan-stem>_audit_report.json)",
    )
    return parser.parse_args(argv)


def _find_source_chunk_text(traceability: str, chunks_by_id: dict[str, str]) -> str:
    """Resolve the source chunk text from a test case's traceability string."""
    for chunk_id, text in chunks_by_id.items():
        if chunk_id in traceability:
            return text
    return ""


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    try:
        from axis_a.indexer import load_chunks_jsonl
        from axis_b.llm_setup import init_tools
        from axis_b.schema import TestPlan
        from axis_c.agents.auditor import run_auditor
        from axis_c.guardrails_validator import validate_test_case

        plan = TestPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
        chunks = load_chunks_jsonl(args.chunks)
        chunks_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}
        init_tools(args.index)

        test_cases = [
            tc
            for fs in plan.feature_sets
            for cond in fs.test_conditions
            for tc in cond.test_cases
        ]
        if not test_cases:
            logger.error("Plan %s contains no test cases.", plan.plan_id)
            return 1

        reports: list[dict] = []
        for tc in test_cases:
            # 1. Cheap structural validation first.
            is_valid, violations = validate_test_case(tc)
            report: dict = {
                "tc_id": tc.tc_id,
                "structural_valid": is_valid,
                "structural_violations": violations,
            }
            if not is_valid:
                report["audit"] = None
                report["verdict"] = "FAIL"
                reports.append(report)
                continue

            # 2. Agentic audit against the source chunk.
            source_text = _find_source_chunk_text(tc.traceability, chunks_by_id)
            if not source_text:
                logger.warning(
                    "No source chunk found for %s (traceability: %s)",
                    tc.tc_id, tc.traceability,
                )
            try:
                audit = run_auditor(tc, source_text)
                report["audit"] = audit
                report["verdict"] = audit["verdict"]
            except Exception as e:
                logger.error("Audit failed for %s: %s", tc.tc_id, e)
                report["audit"] = None
                report["verdict"] = "WARNING"
            reports.append(report)

        # Global metrics (SPEC SS6).
        total = len(reports)
        audited = [r for r in reports if r.get("audit")]
        n_halluc = sum(1 for r in audited if r["audit"]["hallucinations"])
        n_contra = sum(1 for r in audited if r["audit"]["contradictions"])
        n_omiss = sum(1 for r in audited if r["audit"]["omissions"])
        metrics = {
            "total_test_cases": total,
            "audited_test_cases": len(audited),
            "hallucination_rate": n_halluc / total if total else 0.0,
            "contradiction_rate": n_contra / total if total else 0.0,
            "omission_rate": n_omiss / total if total else 0.0,
            "pass": sum(1 for r in reports if r["verdict"] == "PASS"),
            "fail": sum(1 for r in reports if r["verdict"] == "FAIL"),
            "warning": sum(1 for r in reports if r["verdict"] == "WARNING"),
        }

        output_path = args.output or args.plan.parent / f"{args.plan.stem}_audit_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {"plan_id": plan.plan_id, "metrics": metrics, "reports": reports},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Audit report written to %s", output_path)
    except Exception:
        logger.exception("Axis C pipeline failed")
        return 1

    print(
        f"[Axis C] Audited {metrics['total_test_cases']} test cases: "
        f"{metrics['pass']} PASS, {metrics['fail']} FAIL, "
        f"{metrics['warning']} WARNING -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
