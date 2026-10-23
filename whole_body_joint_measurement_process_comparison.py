"""Compare observed versus measurement-process-neutralized joint gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "rows": report.get("rows"),
        "subjects": report.get("subjects"),
        "validated_target_horizon_cells": report.get("validated_target_horizon_cells", []),
        "validated_target_horizon_count": report.get("validated_target_horizon_count", 0),
        "promotion_status": report.get("promotion_status"),
        "measurement_process_mode": report.get("measurement_process_mode"),
        "asof_age_policy": report.get("asof_age_policy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--neutralized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "whole_body_joint_measurement_process_comparison.v1",
        "observed": load(args.observed),
        "neutralized": load(args.neutralized),
        "interpretation": {
            "measurement_process_is_not_promoted_as_physiology": True,
            "neutralized_mode_removes_masks_ages_and_module_presence": True,
            "factual_only": True,
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "observed_validated": report["observed"]["validated_target_horizon_count"],
        "neutralized_validated": report["neutralized"]["validated_target_horizon_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
