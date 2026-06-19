"""Compute Cohen's kappa between the Auditor's verdicts and human annotations.

Usage:
    python scripts/compute_kappa.py \
        --audit-json data/output/axis_c/audit_report.json \
        --human-csv  data/ground_truth/human_verdicts.csv \
        [--output kappa_result.json]

The human CSV must have columns: ``tc_id,human_verdict``
The audit JSON must be a list of objects each containing ``tc_id`` and
``verdict`` (as produced by ``scripts/run_axis_c.py``).

Exits with code 0 on success, 1 on error.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VALID_VERDICTS = {"PASS", "FAIL", "WARNING"}


def load_audit_json(path: Path) -> dict[str, str]:
    """Load ``{tc_id: verdict}`` from an Axis C audit JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Audit JSON must be a list of audit dicts, got {type(data)}")
    result: dict[str, str] = {}
    for entry in data:
        tc_id = entry.get("tc_id", "")
        verdict = str(entry.get("verdict", "")).upper()
        if not tc_id:
            logger.warning("Skipping audit entry with missing tc_id")
            continue
        if verdict not in VALID_VERDICTS:
            logger.warning("Audit entry %s has unrecognised verdict %r — skipping", tc_id, verdict)
            continue
        result[tc_id] = verdict
    return result


def load_human_csv(path: Path) -> dict[str, str]:
    """Load ``{tc_id: human_verdict}`` from the human-annotation CSV."""
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"tc_id", "human_verdict"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Human CSV must have columns {required}, "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            tc_id = row["tc_id"].strip()
            verdict = row["human_verdict"].strip().upper()
            if not tc_id:
                continue
            if verdict not in VALID_VERDICTS:
                logger.warning("Human CSV: tc_id=%s has unrecognised verdict %r — skipping", tc_id, verdict)
                continue
            result[tc_id] = verdict
    return result


def compute_kappa(
    audit_verdicts: dict[str, str],
    human_verdicts: dict[str, str],
) -> tuple[float, int]:
    """Align both dicts on tc_id and compute Cohen's kappa.

    Returns ``(kappa, n_aligned)`` where *n_aligned* is the number of
    tc_ids present in both dicts.
    """
    common_ids = sorted(set(audit_verdicts) & set(human_verdicts))
    if len(common_ids) < 2:
        raise ValueError(
            f"Only {len(common_ids)} tc_id(s) are common to both inputs; "
            "need at least 2 to compute kappa."
        )
    y_audit = [audit_verdicts[tc_id] for tc_id in common_ids]
    y_human = [human_verdicts[tc_id] for tc_id in common_ids]

    kappa = float(cohen_kappa_score(y_human, y_audit))
    return kappa, len(common_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa between Auditor verdicts and human annotations."
    )
    parser.add_argument(
        "--audit-json",
        required=True,
        type=Path,
        help="Path to the Axis C audit report JSON (list of audit dicts).",
    )
    parser.add_argument(
        "--human-csv",
        required=True,
        type=Path,
        help="Path to the human-annotation CSV (columns: tc_id, human_verdict).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the kappa result as JSON.",
    )
    args = parser.parse_args(argv)

    try:
        audit_verdicts = load_audit_json(args.audit_json)
        logger.info("Loaded %d audit verdicts from %s", len(audit_verdicts), args.audit_json)

        human_verdicts = load_human_csv(args.human_csv)
        logger.info("Loaded %d human verdicts from %s", len(human_verdicts), args.human_csv)

        kappa, n = compute_kappa(audit_verdicts, human_verdicts)
    except Exception as exc:
        logger.error("Error: %s", exc)
        return 1

    result_line = f"Cohen's kappa: {kappa:.4f} (N={n} test cases)"
    print(result_line)

    if args.output:
        result = {
            "kappa": kappa,
            "n_aligned": n,
            "audit_total": len(audit_verdicts),
            "human_total": len(human_verdicts),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Kappa result written to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
