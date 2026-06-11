#!/usr/bin/env python3
"""Axis B runner -- generate an ISO 29119-3 test plan from requirement chunks.

Usage:
    python scripts/run_axis_b.py \\
        --chunks data/output/chunks/iso_18013_5_chunks.jsonl \\
        --index data/output/index/faiss_index \\
        --norm "ISO/IEC 18013-5" \\
        --output data/output/test_plans/TP-ISO-IEC-18013-5-001.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable when running as `python scripts/run_axis_b.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Axis B: generate an ISO 29119-3 test plan from the "
        "requirement chunks and FAISS index produced by Axis A."
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
        "--norm", type=str, required=True, help='Norm name, e.g. "ISO/IEC 18013-5"'
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for the TestPlan JSON",
    )
    parser.add_argument(
        "--plan-id", type=str, default=None,
        help="Optional explicit plan id (default: derived from the norm name)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    try:
        from axis_a.indexer import load_chunks_jsonl
        from axis_b.pipeline import run_pipeline

        chunks = load_chunks_jsonl(args.chunks)
        if not chunks:
            logger.error("No chunks loaded from %s", args.chunks)
            return 1

        plan = run_pipeline(
            chunks=chunks,
            index_path=args.index,
            norm=args.norm,
            output_path=args.output,
            plan_id=args.plan_id,
        )
    except Exception:
        logger.exception("Axis B pipeline failed")
        return 1

    total_cases = sum(
        len(cond.test_cases)
        for fs in plan.feature_sets
        for cond in fs.test_conditions
    )
    print(
        f"[Axis B] Generated test plan {plan.plan_id} with {total_cases} "
        f"test cases -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
