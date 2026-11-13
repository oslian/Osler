"""Aggregate fail-closed renal coupling audits from isolated processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


REQUIRED_EXTERNAL_SCOPES = {"hospital", "care_unit", "time"}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_counts(runs: list[dict[str, object]], key: str) -> dict[str, int]:
    tasks = [run["tasks"][key] for run in runs]
    return {
        "runs": len(tasks),
        "point": sum(task["passes_point_gate"] for task in tasks),
        "coupling": sum(task["passes_coupling_gate"] for task in tasks),
        "coverage": sum(
            task["adaptive_conformal"]["passes_0_87_to_0_93"] for task in tasks
        ),
        "narrower_than_independent": sum(
            task["passes_coupled_interval_gate"] for task in tasks
        ),
    }


def _interval_summary(runs: list[dict[str, object]], key: str) -> dict[str, float]:
    tasks = [run["tasks"][key] for run in runs]
    comparisons = [task["matched_independent_interval_width"] for task in tasks]
    candidate_half = mean(item["candidate_mean_half_width"] for item in comparisons)
    independent_half = mean(item["population_mean_half_width"] for item in comparisons)
    reduction = 2.0 * (independent_half - candidate_half)
    independent_full = 2.0 * independent_half
    return {
        "coupled_mean_half_width": round(candidate_half, 6),
        "independent_mean_half_width": round(independent_half, 6),
        "coupled_mean_full_width": round(2.0 * candidate_half, 6),
        "independent_mean_full_width": round(independent_full, 6),
        "mean_full_width_reduction": round(reduction, 6),
        "mean_full_width_reduction_percent": round(
            100.0 * reduction / independent_full if independent_full > 0.0 else 0.0,
            3,
        ),
        "median_coupling_gain": round(median(task["coupling_gain"] for task in tasks), 6),
        "mean_candidate_coverage": round(
            mean(task["adaptive_conformal"]["coverage"] for task in tasks), 6
        ),
        "mean_independent_coverage": round(
            mean(
                item["independent_coverage"]
                for item in comparisons
            ),
            6,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--external-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    patient_reports = [_load(path) for path in args.patient_reports]
    external_reports = [_load(path) for path in args.external_reports]
    if len(patient_reports) != 7:
        raise ValueError("exactly seven isolated patient reports are required")
    patient_runs = [report["patient_runs"][0] for report in patient_reports]
    patient_seeds = {run["seed"] for run in patient_runs}
    if len(patient_seeds) != 7:
        raise ValueError("patient reports must contain seven distinct seeds")
    external_scopes = {report["external_gate_scope"] for report in external_reports}
    if external_scopes != REQUIRED_EXTERNAL_SCOPES:
        raise ValueError(
            f"external reports must cover {sorted(REQUIRED_EXTERNAL_SCOPES)} exactly"
        )
    external_runs = [report["external_runs"][0] for report in external_reports]
    task_keys = tuple(patient_runs[0]["tasks"])
    if any(tuple(run["tasks"]) != task_keys for run in patient_runs + external_runs):
        raise ValueError("audit reports have inconsistent renal task registries")

    cells = {}
    for key in task_keys:
        patient = _gate_counts(patient_runs, key)
        external = _gate_counts(external_runs, key)
        promoted = all(
            patient[name] == 7 for name in ("point", "coupling", "coverage", "narrower_than_independent")
        ) and all(
            external[name] == 3 for name in ("point", "coupling", "coverage", "narrower_than_independent")
        )
        cells[key] = {
            "patient_heldout": patient,
            "external_heldout": external,
            "patient_interval": _interval_summary(patient_runs, key),
            "external_interval": _interval_summary(external_runs, key),
            "coupled_precision_promoted": promoted,
        }

    output = {
        "schema": "renal_shared_patient_state_combined.v1",
        "architecture": {
            "anchor": "independently validated patient-specific renal expert",
            "coupling": "calibration-selected shared renal-state residual",
            "fail_closed": True,
        },
        "patient_seeds": sorted(patient_seeds),
        "external_scopes": sorted(external_scopes),
        "cells": cells,
        "coupled_precision_promoted_cells": [
            key for key, cell in cells.items() if cell["coupled_precision_promoted"]
        ],
        "boundary": {
            "factual_only": True,
            "causal_claim_allowed": False,
            "independent_artifacts_remain_active_for_unpromoted_cells": True,
        },
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "coupled_precision_promoted_cells": output["coupled_precision_promoted_cells"],
    }, indent=2))


if __name__ == "__main__":
    main()
