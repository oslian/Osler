#!/usr/bin/env python3
"""Freeze the eICU-training to MIMIC-IV replication inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def build_registry(model_dir: Path) -> dict:
    runtime = json.loads((model_dir / "runtime_registry.json").read_text())
    external = json.loads((model_dir / "external_validation_report.json").read_text())
    heads = joblib.load(model_dir / "unified_shared_jepa_head_bundle.joblib")
    head_cells = set(heads["heads"])
    external_cells = {row["cell"]: row for row in external["cells"]}

    cells = []
    for row in runtime["cells"]:
        cell = str(row["cell"])
        target = str(row["target"])
        is_event = target.endswith("_event")
        if is_event:
            artifact = "event_risk_heads.joblib"
            candidate_kind = "calibrated_event_risk_head"
        elif cell in head_cells:
            artifact = "unified_shared_jepa_head_bundle.joblib"
            candidate_kind = "target_specific_head"
        else:
            artifact = "whole_body_joint_jepa.pt"
            candidate_kind = "shared_jepa_base"
        strict = bool(row.get("can_move"))
        external_detail = external_cells.get(cell, {}).get("external")
        cells.append(
            {
                "cell": cell,
                "target": target,
                "horizon_hours": int(row["horizon_hours"]),
                "task_type": "binary_event_probability" if is_event else "continuous_value",
                "candidate_kind": candidate_kind,
                "artifact": artifact,
                "eicu_trained_or_selected": True,
                "mimic_labels_used_for_selection": False,
                "mimic_frozen_weight_replication_pass": strict,
                "runtime_eligible": strict,
                "status": (
                    "strict_eicu_to_mimic_replication"
                    if strict
                    else "challenge_candidate_not_replicated"
                ),
                "mimic_external": external_detail
                or {
                    "patient_gate": row.get("patient_gate"),
                    "careunit_gate": row.get("careunit_gate"),
                    "forward_time_gate": row.get("forward_time_gate"),
                    "direction_accuracy": row.get("direction_accuracy"),
                    "delta_correlation": row.get("delta_correlation"),
                },
            }
        )

    strict = [row["cell"] for row in cells if row["runtime_eligible"]]
    challenge = [row["cell"] for row in cells if not row["runtime_eligible"]]
    if len(strict) != 27 or len(challenge) != 14:
        raise ValueError("release replication inventory must remain 27 strict / 14 challenge")
    return {
        "schema": "whole_body_jepa.cross_database_replication_registry.v1",
        "training_and_selection_database": "eICU",
        "frozen_external_validation_database": "MIMIC-IV",
        "transfer_type": "frozen_weight_direct_transfer",
        "mimic_retraining_allowed": False,
        "mimic_labels_used_for_selection": False,
        "catalog_cells": len(cells),
        "strict_replicated_cells": len(strict),
        "challenge_cells": len(challenge),
        "strict_cell_names": strict,
        "challenge_cell_names": challenge,
        "new_database_policy": {
            "freeze_all_weights": True,
            "required_gates": ["patient", "domain", "forward_time"],
            "selection_on_new_lockbox_allowed": False,
            "adaptation_requires_a_new_separate_lockbox": True,
        },
        "required_scored_lockbox_columns": [
            "patient_id",
            "cell",
            "truth",
            "candidate",
            "persistence",
            "domain_id",
            "time_block",
        ],
        "cells": cells,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/cross_database_replication_registry.json"),
    )
    args = parser.parse_args()
    registry = build_registry(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "strict_replicated_cells": registry["strict_replicated_cells"],
                "challenge_cells": registry["challenge_cells"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
