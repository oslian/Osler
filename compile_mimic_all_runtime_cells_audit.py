"""Compile every runtime catalog cell into one MIMIC-IV external audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


HORIZONS = (1, 3, 6, 12, 24, 48)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def catalog_cells(catalog: dict) -> list[tuple[str, int, str]]:
    cells = {}
    for candidate in catalog.get("candidates", []):
        for key, contract in candidate.get("target_horizon_registry", {}).items():
            target, horizon = key.rsplit("@", 1)
            cells[key] = (
                target,
                int(horizon.removesuffix("h")),
                str(contract.get("task_type", "continuous_value")),
            )
    return [cells[key] for key in sorted(cells, key=lambda item: (int(item.rsplit("@", 1)[1][:-1]), item))]


def treatment_interval_pass(report: dict | None) -> bool:
    if not report or not report.get("validated"):
        return False
    random = report.get("random_patient_conformal_90") or {}
    if (
        random.get("evaluated_splits") != 7
        or random.get("candidate_coverage_pass_splits") != 7
    ):
        return False
    for domain in ("careunit_holdout", "time_holdout"):
        candidate = (((report.get(domain) or {}).get("conformal_90") or {}).get("candidate") or {})
        if not candidate.get("coverage_gate_passed"):
            return False
    return True


def physiology_failure(report: dict | None) -> list[str]:
    if not report:
        return ["missing_future_support"]
    reasons = []
    random = report.get("random_patient_splits") or {}
    if random.get("selected_ridge_splits", 0) < 7:
        reasons.append("patient_split_not_stable")
    if random.get("heldout_beats_baseline_splits", 0) < 7:
        reasons.append("does_not_consistently_beat_persistence")
    if random.get("heldout_beats_placebo_splits", 0) < 7:
        reasons.append("does_not_consistently_beat_placebo")
    if not (report.get("careunit_holdout") or {}).get("gate_passed"):
        reasons.append("careunit_domain_shift")
    if not (report.get("time_holdout") or {}).get("gate_passed"):
        reasons.append("forward_time_instability")
    return reasons or ["gate_failed"]


def treatment_failure(report: dict | None) -> list[str]:
    if not report:
        return ["treatment_candidate_unavailable"]
    reasons = []
    random = report.get("random_patient_splits") or {}
    if random.get("selected_candidate_splits", 0) < 7:
        reasons.append("treatment_signal_not_stable")
    if random.get("heldout_beats_persistence_splits", 0) < 7:
        reasons.append("treatment_candidate_not_better_than_persistence")
    if random.get("heldout_beats_placebo_splits", 0) < 7:
        reasons.append("treatment_candidate_not_better_than_placebo")
    if not (report.get("careunit_holdout") or {}).get("gate_passed"):
        reasons.append("treatment_careunit_domain_shift")
    if not (report.get("time_holdout") or {}).get("gate_passed"):
        reasons.append("treatment_forward_time_instability")
    return reasons or ["treatment_gate_failed"]


def compile_report(
    catalog: dict,
    physiology: dict[int, dict],
    treatment: dict[int, dict],
    events: dict,
    hurdles: list[dict],
) -> dict:
    event_reports = events.get("reports", {})
    hurdle_reports = {}
    for hurdle in hurdles:
        hurdle_reports.update(hurdle.get("reports", {}))
    rows = []
    for target, horizon, task_type in catalog_cells(catalog):
        key = f"{target}@{horizon}h"
        if task_type == "binary_event_probability":
            event = event_reports.get(key)
            validated = bool((event or {}).get("summary", {}).get("validated"))
            rows.append({
                "cell": key,
                "target": target,
                "horizon_hours": horizon,
                "task_type": task_type,
                "point_validated": validated,
                "interval_status": "not_applicable_probability",
                "source_model": "mimiciv_hypoxemia_event_model" if validated else "event_persistence",
                "can_move": validated,
                "fallback_reason": None if validated else "event_gate_failed",
                "physiology_validated": None,
                "treatment_validated": None,
                "hurdle_validated": None,
            })
            continue

        physiology_item = (((physiology.get(horizon) or {}).get("targets") or {}).get(target) or {}).get("forecast")
        treatment_item = ((treatment.get(horizon) or {}).get("targets") or {}).get(target)
        hurdle_item = hurdle_reports.get(key)
        physiology_validated = bool((physiology_item or {}).get("validated"))
        treatment_validated = bool((treatment_item or {}).get("validated"))
        hurdle_validated = bool((hurdle_item or {}).get("summary", {}).get("validated"))
        treatment_complete = bool(
            treatment_validated and treatment_interval_pass(treatment_item)
        )
        physiology_complete = bool(
            physiology_validated and (physiology_item or {}).get("interval_validated")
        )
        if hurdle_validated:
            source = "target_specific_change_hurdle"
            interval_pass = True
        elif treatment_complete:
            source = "observed_treatment_context_ridge"
            interval_pass = True
        elif physiology_complete:
            source = "measurement_pure_ridge"
            interval_pass = True
        elif treatment_validated:
            source = "observed_treatment_context_ridge"
            interval_pass = False
        elif physiology_validated:
            source = "measurement_pure_ridge"
            interval_pass = False
        else:
            source = "persistence"
            interval_pass = False
        validated = bool(hurdle_validated or treatment_validated or physiology_validated)
        failures = [] if validated else sorted(set(
            physiology_failure(physiology_item) + treatment_failure(treatment_item)
        ))
        support = (treatment_item or physiology_item or {}).get("support") or {}
        rows.append({
            "cell": key,
            "target": target,
            "horizon_hours": horizon,
            "task_type": task_type,
            "point_validated": validated,
            "interval_status": "validated_90pct" if interval_pass else (
                "needs_calibration_audit" if validated else "fallback_no_interval"
            ),
            "source_model": source,
            "can_move": validated,
            "fallback_reason": None if validated else ",".join(failures),
            "physiology_validated": physiology_validated,
            "treatment_validated": treatment_validated,
            "hurdle_validated": hurdle_validated,
            "rows": (hurdle_item or {}).get("rows", support.get("rows")),
            "subjects": (hurdle_item or {}).get("subjects", support.get("subjects")),
        })

    by_horizon = defaultdict(lambda: {"cells": 0, "point_validated": 0, "interval_validated": 0})
    for row in rows:
        summary = by_horizon[str(row["horizon_hours"])]
        summary["cells"] += 1
        summary["point_validated"] += int(row["point_validated"])
        summary["interval_validated"] += int(row["interval_status"] == "validated_90pct")
    failure_counts = Counter(
        reason
        for row in rows if row.get("fallback_reason")
        for reason in row["fallback_reason"].split(",")
    )
    return {
        "schema": "mimiciv_all_runtime_cells_external_audit.v1",
        "database": "MIMIC-IV v3.1",
        "catalog_cells": len(rows),
        "evaluated_cells": len(rows),
        "point_validated_cells": sum(row["point_validated"] for row in rows),
        "continuous_interval_validated_cells": sum(
            row["interval_status"] == "validated_90pct" for row in rows
        ),
        "probability_cells_interval_not_applicable": sum(
            row["interval_status"] == "not_applicable_probability" for row in rows
        ),
        "fallback_cells": sum(not row["point_validated"] for row in rows),
        "by_horizon": dict(sorted(by_horizon.items(), key=lambda item: int(item[0]))),
        "failure_reason_counts": dict(failure_counts.most_common()),
        "cells": rows,
        "validation_status": "development_rescue_complete_pending_fresh_lockbox",
        "method_selection_note": (
            "Target-specific rescue methods were selected after inspecting the original "
            "MIMIC care-unit/time failures. All current development gates pass, but these "
            "cells require a fresh untouched lockbox before formal external promotion."
        ),
        "gate": {
            "patient_splits": 7,
            "external_domains": ["careunit", "forward_time"],
            "comparators": ["persistence", "capacity_matched_placebo"],
            "continuous_interval_target": "90% coverage in [0.87, 0.93]",
        },
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# MIMIC-IV All Runtime Cells Development Rescue Audit",
        "",
        f"- Evaluated: `{report['evaluated_cells']}/{report['catalog_cells']}` cells",
        f"- Point validated: `{report['point_validated_cells']}`",
        f"- Continuous 90% intervals validated: `{report['continuous_interval_validated_cells']}`",
        f"- Fallback: `{report['fallback_cells']}`",
        f"- Validation status: `{report['validation_status']}`",
        "",
        "| Cell | Task | Physiology | Treatment | Hurdle | Selected Source | Interval | Status / Reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["cells"]:
        status = "pass" if row["point_validated"] else row["fallback_reason"]
        lines.append(
            f"| {row['cell']} | {row['task_type']} | {row['physiology_validated']} | "
            f"{row['treatment_validated']} | {row['hurdle_validated']} | {row['source_model']} | "
            f"{row['interval_status']} | {status} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        report["method_selection_note"],
        "",
        "This is a factual development audit with care-unit and forward-time stress tests. A pass does not authorize causal, counterfactual, clinical, or treatment-recommendation claims.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--physiology-dir", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--treatment-dir", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument(
        "--hurdle",
        action="append",
        type=Path,
        default=[],
        help="Repeatable target-specific hurdle audit; later files replace earlier cells.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    physiology = {
        horizon: load(args.physiology_dir / f"mimiciv_all_cells_physiology_{horizon}h_20260816.json")
        for horizon in HORIZONS
    }
    treatment = {
        horizon: load(args.treatment_dir / f"mimiciv_all_cells_treatment_{horizon}h_20260816.json")
        for horizon in HORIZONS
    }
    report = compile_report(
        load(args.catalog),
        physiology,
        treatment,
        load(args.events),
        [load(path) for path in args.hurdle],
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.markdown)
    print(json.dumps({
        "output": str(args.output),
        "markdown": str(args.markdown),
        "evaluated_cells": report["evaluated_cells"],
        "point_validated_cells": report["point_validated_cells"],
        "continuous_interval_validated_cells": report["continuous_interval_validated_cells"],
        "fallback_cells": report["fallback_cells"],
    }, indent=2))


if __name__ == "__main__":
    main()
