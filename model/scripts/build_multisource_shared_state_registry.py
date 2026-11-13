"""Compile the care-unit-corrected audit; never hide unmet coverage goals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from whole_body_jepa.multisource_intervals import load_calibrator, sha256


def build_registry(audit: dict, interval_audit: dict) -> dict:
    cells = []
    for row in audit["cells"]:
        evaluated = row.get("status") == "evaluated"
        corrected = interval_audit["cells"].get(row["cell"], {})
        point_pass = bool(evaluated and corrected.get("corrected_nwicu_point_gate", {}).get("pass"))
        interval_pass = bool(point_pass and corrected.get("runtime_interval_eligible"))
        cells.append(
            {
                "cell": row["cell"],
                "status": row["status"],
                "shared_state_consumed": evaluated,
                "shared_state_point_transfer_pass": point_pass,
                "calibrated_interval_pass": interval_pass,
                "latent_incremental_pass": False,
                "latent_incremental_status": "needs_careunit_corrected_ablation" if evaluated else "not_evaluable",
                "selected_feature_view": (
                    row.get("selection", {}).get("feature_view") if evaluated else None
                ),
                "fallback_reason": (
                    None
                    if point_pass
                    else row.get("reason", "corrected_external_point_transfer_not_passed")
                ),
                "interval_failure_reasons": corrected.get("failure_reasons", ["not_evaluable"]),
            }
        )

    point_cells = [
        row["cell"] for row in cells if row["shared_state_point_transfer_pass"]
    ]
    interval_cells = [row["cell"] for row in cells if row["calibrated_interval_pass"]]
    return {
        "schema": "whole_body_jepa.multisource_shared_state_registry.v1",
        "catalog_cells": len(cells),
        "shared_state_point_cells": len(point_cells),
        "shared_state_point_cell_ids": point_cells,
        "shared_state_consuming_cells": sum(row["shared_state_consumed"] for row in cells),
        "calibrated_interval_cells": len(interval_cells),
        "calibrated_interval_cell_ids": interval_cells,
        "latent_incremental_cells": 0,
        "latent_incremental_cell_ids": [],
        "latent_incremental_revalidation_required": True,
        "requested_point_coverage_goal": 35,
        "requested_point_coverage_goal_met": len(point_cells) >= 35,
        "not_externally_evaluable_cells": sum(
            row["status"] != "evaluated" for row in cells
        ),
        "development_databases": audit["development_databases"],
        "external_lockbox": audit["external_lockbox"],
        "external_point_results_previously_inspected": True,
        "new_untouched_lockbox_claim": False,
        "external_labels_used_for_selection": False,
        "point_forecast_candidate_allowed": True,
        "runtime_replacement_allowed": False,
        "runtime_blocker": "candidate_only_current_release_not_replaced",
        "supersedes_prior_count": "37_point_pass_and_5_latent_pass_used_lost_careunit_provenance",
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--interval-audit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    intervals = load_calibrator(args.interval_audit, args.model_dir)
    registry = build_registry(audit, intervals)
    registry["interval_calibrator_sha256"] = sha256(args.interval_audit)
    if registry["catalog_cells"] != 41:
        raise ValueError("expected the frozen 41-cell catalog")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "catalog_cells": registry["catalog_cells"],
                "shared_state_point_cells": registry["shared_state_point_cells"],
                "latent_incremental_cells": registry["latent_incremental_cells"],
                "calibrated_interval_cells": registry["calibrated_interval_cells"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
