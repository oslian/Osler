#!/usr/bin/env python3
"""Audit the deployed 90% tolerance intervals and their release evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACCEPTED_COVERAGE = (0.87, 0.93)
EVENT_TARGET = "hypoxemia_event"


def _inside(value: float | None) -> bool:
    return value is not None and ACCEPTED_COVERAGE[0] <= float(value) <= ACCEPTED_COVERAGE[1]


def _original_evidence(cell: str, item: dict[str, Any]) -> dict[str, Any]:
    patient_splits = item.get("patient_splits", [])
    patient_pass = len(patient_splits) == 7 and all(
        split.get("pass")
        and _inside(split.get("row_coverage"))
        and _inside(split.get("patient_equal_coverage"))
        for split in patient_splits
    )
    careunit = item.get("careunit_holdout", {})
    time = item.get("forward_time", {})
    empirical_pass = bool(
        item.get("runtime_eligible")
        and patient_pass
        and careunit.get("pass")
        and _inside(careunit.get("careunit_equal_coverage"))
        and time.get("pass")
        and _inside(time.get("coverage"))
    )
    return {
        "cell": cell,
        "interval_source": "release_interval_calibrator",
        "method": item.get("runtime_interval_method", "constant"),
        "nominal_coverage": 0.90,
        "patient_split_coverage_pass": patient_pass,
        "careunit_equal_coverage": careunit.get("careunit_equal_coverage"),
        "forward_time_coverage": time.get("coverage"),
        "empirical_coverage_gate_pass": empirical_pass,
        "mean_width": item.get("mean_width"),
        "mean_width_available": item.get("mean_width") is not None,
        "crps_vs_persistence_available": "crps_vs_persistence" in item,
        "applicability_range_available": "applicability_range" in item,
    }


def _selector_scope_pass(scope: dict[str, Any]) -> bool:
    coverage = scope.get("coverage", {})
    careunit = scope.get("careunit_coverage", {})
    late = scope.get("late_coverage", {})
    return bool(
        coverage.get("pass")
        and _inside(coverage.get("patient_equal_coverage"))
        and careunit.get("pass")
        and _inside(careunit.get("careunit_equal_coverage"))
        and late.get("pass")
        and _inside(late.get("patient_equal_coverage"))
    )


def _selector_evidence(cell: str, item: dict[str, Any]) -> dict[str, Any]:
    validation = item.get("validation", {})
    method = item.get("interval_method")
    radius = item.get("radius")
    mean_width = 2.0 * float(radius) if radius is not None else None
    empirical_pass = bool(
        validation.get("promotion_allowed")
        and _selector_scope_pass(validation.get("mimic_test", {}))
        and _selector_scope_pass(validation.get("nwicu", {}))
    )
    return {
        "cell": cell,
        "interval_source": "unified_source_selector",
        "method": method,
        "nominal_coverage": 0.90,
        "patient_split_coverage_pass": None,
        "mimic_patient_equal_coverage": validation.get("mimic_test", {}).get("coverage", {}).get(
            "patient_equal_coverage"
        ),
        "nwicu_patient_equal_coverage": validation.get("nwicu", {}).get("coverage", {}).get(
            "patient_equal_coverage"
        ),
        "empirical_coverage_gate_pass": empirical_pass,
        "mean_width": mean_width,
        "mean_width_available": mean_width is not None,
        "crps_vs_persistence_available": "crps_vs_persistence" in validation,
        "applicability_range_available": "applicability_range" in item,
    }


def audit_release(model_dir: Path) -> dict[str, Any]:
    manifest = json.loads((model_dir / "MODEL_MANIFEST.json").read_text())
    registry = json.loads((model_dir / "runtime_registry.json").read_text())
    calibrator = json.loads((model_dir / "interval_calibrator.json").read_text())
    selector = json.loads((model_dir / "unified_source_selector.json").read_text())

    cells: list[dict[str, Any]] = []
    for contract in registry["cells"]:
        cell = str(contract["cell"])
        decision = selector["cells"][cell]
        # The selector is the final runtime authority: it may promote a source
        # after the frozen registry was written or retain a registry fallback.
        if contract["target"] == EVENT_TARGET or not decision["can_move"]:
            continue
        if decision.get("use_selector_interval"):
            cells.append(_selector_evidence(cell, decision))
            continue
        original = calibrator["cells"].get(cell)
        if not decision.get("source_changed") and original and original.get("runtime_eligible"):
            cells.append(_original_evidence(cell, original))

    expected = int(manifest["runtime_interval_cells"])
    calibrated = sum(item["empirical_coverage_gate_pass"] for item in cells)
    width_available = sum(item["mean_width_available"] for item in cells)
    crps_available = sum(item["crps_vs_persistence_available"] for item in cells)
    applicability_available = sum(item["applicability_range_available"] for item in cells)
    return {
        "schema": "whole_body_jepa.release_tolerance_interval_audit.v1",
        "model_release_date": manifest["release_date"],
        "nominal_coverage": 0.90,
        "accepted_coverage_range": list(ACCEPTED_COVERAGE),
        "summary": {
            "manifest_runtime_interval_cells": expected,
            "resolved_runtime_interval_cells": len(cells),
            "count_matches_manifest": len(cells) == expected,
            "empirical_coverage_gate_pass_cells": calibrated,
            "mean_width_available_cells": width_available,
            "crps_vs_persistence_available_cells": crps_available,
            "applicability_range_available_cells": applicability_available,
            "runtime_api_exposes_empirical_coverage": False,
            "coverage_calibration_release_ready": len(cells) == expected == calibrated,
            "distributional_release_complete": bool(
                len(cells) == expected == calibrated == width_available == crps_available == applicability_available
            ),
        },
        "cells": cells,
        "interpretation": {
            "coverage": "Runtime intervals are eligible only when patient, domain, and time coverage gates pass.",
            "distribution_score": "CRPS versus an empirical persistence distribution is not stored for the current release.",
            "applicability": "Calibration database is recorded, but target-value applicability ranges are not stored per cell.",
            "api": "Runtime emits lower/upper and interval_status, but not nominal and empirical coverage as separate fields.",
        },
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/release_tolerance_interval_audit_20260904.json"),
    )
    args = parser.parse_args()
    result = audit_release(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
