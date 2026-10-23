"""Materialize the target-horizon validation registry from a gate report.

This is useful when the model gate was already run before the registry fields
were added to the gate schema.  It only derives metadata from held-out results;
it never changes a metric or promotes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from whole_body_shared_gate import SEEDS


def add_registry(report: dict) -> dict:
    patient = report["patient_heldout"]["aggregate"]
    conformal = report["patient_heldout"]["conformal"]
    hospital_values = report["hospital_heldout"]["values"]
    hospital_conformal = report["hospital_heldout"]["conformal"]
    grouped = {}
    for cell in report["module_balanced"]["cells"]:
        key = f"{cell['target']}@{int(cell['horizon_hours']) if float(cell['horizon_hours']).is_integer() else cell['horizon_hours']}h"
        grouped.setdefault(key, []).append(cell)
    module_summary = {
        key: {
            "n_modules": len(cells),
            "pass_rate": float(np.mean([cell["pass"] for cell in cells])),
            "pass_module_threshold": bool(
                len(cells) >= 3 and np.mean([cell["pass"] for cell in cells]) >= 0.80
            ),
        }
        for key, cells in sorted(grouped.items())
    }
    registry = {}
    validated = []
    for key, value in patient.items():
        module_evidence = module_summary.get(key, {})
        status = bool(
            value.get("n_splits") == len(SEEDS)
            and value.get("pass_all_splits", False)
            and conformal.get(key, {}).get("n_splits") == len(SEEDS)
            and conformal.get(key, {}).get("pass_all_splits", False)
            and hospital_values.get(key, {}).get("pass_value", False)
            and hospital_conformal.get(key, {}).get("pass", False)
            and module_evidence.get("pass_module_threshold", False)
        )
        registry[key] = {
            "validated": status,
            "patient_value": bool(value.get("pass_all_splits", False)),
            "patient_conformal": bool(conformal.get(key, {}).get("pass_all_splits", False)),
            "hospital_value": bool(hospital_values.get(key, {}).get("pass_value", False)),
            "hospital_conformal": bool(hospital_conformal.get(key, {}).get("pass", False)),
            "module_evidence": module_evidence,
        }
        if status:
            validated.append(key)
    report = dict(report)
    report["module_balanced_by_target_horizon"] = module_summary
    report["validated_target_horizon_cells"] = sorted(validated)
    report["validated_target_horizon_count"] = len(validated)
    report["target_horizon_registry"] = registry
    report["shared_latent_partially_validated"] = bool(validated)
    if not report.get("shared_latent_validated", False):
        report["fail_closed_reason"] = (
            "global shared latent remains candidate; only validated target-horizon cells may move"
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = add_registry(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "validated_target_horizon_count": report["validated_target_horizon_count"],
                "shared_latent_validated": report.get("shared_latent_validated", False),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
