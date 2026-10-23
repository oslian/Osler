"""Test whether joint-JEPA signal depends on how much of the body is observed.

The whole-body model is not evaluated on a pooled mixture of sparse and dense
anchors.  Each threshold keeps only anchors with at least N simultaneously
observed body-system modules, rebuilds the patient history within that cohort,
and then runs the same patient, hospital, conformal, cross-system, and
module-balance gates independently.

This is factual research only.  It distinguishes a diluted joint signal from
an unsupported whole-body latent; it does not estimate intervention effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from whole_body_joint_gate import run_gate
from whole_body_joint_jepa import load_joint_cohort


DEFAULT_MODULES = (
    "sepsis,aki,respiratory,integumentary_skin_wound,toxic_metabolic,"
    "electrolyte_acid_base,endocrine_stress,gi_pancreatic_nutrition,"
    "cardiac_injury,musculoskeletal_rhabdo,immune_inflammatory,"
    "cardiovascular_instability,acute_neuro,hepatic_failure,coagulopathy_heme"
)


def _support(frame) -> dict:
    count = frame["_module_count"].to_numpy(dtype=np.int64)
    return {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].astype(str).nunique()),
        "hospitals": int(frame["hospitalid"].nunique()),
        "mean_modules_per_row": float(np.mean(count)) if len(count) else None,
        "module_count_distribution": {
            str(value): int((count == value).sum()) for value in sorted(set(count.tolist()))
        },
    }


def _summary(gate: dict) -> dict:
    registry = gate.get("joint_runtime_registry", gate.get("target_horizon_registry", {}))
    return {
        "promotion_status": gate.get("promotion_status"),
        "validated_runtime_cells": gate.get(
            "validated_joint_runtime_cells", gate.get("validated_target_horizon_cells", [])
        ),
        "validated_runtime_cell_count": int(
            gate.get(
                "validated_joint_runtime_cell_count",
                gate.get("validated_target_horizon_count", 0),
            )
        ),
        "cells_with_patient_value": int(
            sum(bool(value.get("patient_value", False)) for value in registry.values())
        ),
        "cells_with_patient_cross_system": int(
            sum(bool(value.get("patient_cross_system", False)) for value in registry.values())
        ),
        "cells_with_patient_module_balance": int(
            sum(bool(value.get("patient_module_balanced", False)) for value in registry.values())
        ),
        "cells_with_hospital_value": int(
            sum(bool(value.get("hospital_value", False)) for value in registry.values())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--thresholds", default="2,3,4,5")
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-subjects", type=int, default=100)
    parser.add_argument("--validated-edge-adapters", action="store_true")
    parser.add_argument("--validated-edge-stage-epochs", type=int, default=0)
    parser.add_argument("--validated-edge-targets", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    thresholds = tuple(sorted({int(value.strip()) for value in args.thresholds.split(",") if value.strip()}))
    if not thresholds or thresholds[0] < 2:
        raise ValueError("thresholds must contain module counts of at least 2")

    frame, variables, module_names = load_joint_cohort(
        args.data_root,
        modules,
        horizons,
        max_stays_per_horizon=args.max_stays,
        min_modules=min(thresholds),
        alignment="asof",
        asof_max_age_hours=6.0,
        future_label_tolerance_hours=0.0,
        asof_age_policy="horizon_scaled",
    )
    reports = {}
    for threshold in thresholds:
        subset = frame.loc[frame["_module_count"] >= threshold].copy().reset_index(drop=True)
        subset.attrs = frame.attrs.copy()
        support = _support(subset)
        key = f"at_least_{threshold}_modules"
        if support["subjects"] < int(args.min_subjects):
            reports[key] = {
                "threshold": threshold,
                "support": support,
                "status": "insufficient_subject_support",
                "minimum_subjects": int(args.min_subjects),
            }
            print(json.dumps({"threshold": threshold, **support, "status": "insufficient_subject_support"}), flush=True)
            continue
        gate = run_gate(
            subset,
            variables,
            module_names,
            epochs=args.epochs,
            batch_size=args.batch_size,
            measurement_process_mode="observed",
            cross_forecast_weight=1.0,
            masked_forecast_probability=0.50,
            module_mask_probability=0.75,
            world_model=True,
            world_model_weight=1.0,
            future_target_mode="window",
            target_min_observations=2,
            regime_balanced_loss=True,
            target_encoder_mode="fast",
            target_horizon_regime_adapter=True,
            patient_residual_adapter=True,
            future_task_balanced_sampling=True,
            validated_edge_adapters=args.validated_edge_adapters,
            validated_edge_stage_epochs=args.validated_edge_stage_epochs,
            validated_edge_targets=(
                tuple(value.strip() for value in args.validated_edge_targets.split(",") if value.strip())
                if args.validated_edge_targets
                else None
            ),
        )
        reports[key] = {
            "threshold": threshold,
            "support": support,
            "status": "evaluated",
            "gate_summary": _summary(gate),
            "gate": gate,
        }
        args.output.write_text(
            json.dumps(
                {
                    "schema": "whole_body_joint_observability_threshold_audit.v1",
                    "modules": list(module_names),
                    "horizons": list(horizons),
                    "thresholds": list(thresholds),
                    "reports": reports,
                    "interpretation": "A threshold can only support joint runtime when its independent joint gate validates target cells. Fewer rows at high thresholds are reported as support limits, not evidence of no physiology.",
                    "causal_claim_allowed": False,
                    "clinical_promotion_allowed": False,
                    "validated_edge_targets": (
                        [value.strip() for value in args.validated_edge_targets.split(",") if value.strip()]
                        if args.validated_edge_targets
                        else None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"threshold": threshold, **support, **_summary(gate)}), flush=True)

    report = {
        "schema": "whole_body_joint_observability_threshold_audit.v1",
        "modules": list(module_names),
        "horizons": list(horizons),
        "thresholds": list(thresholds),
        "reports": reports,
        "interpretation": "A threshold can only support joint runtime when its independent joint gate validates target cells. Fewer rows at high thresholds are reported as support limits, not evidence of no physiology.",
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "validated_edge_targets": (
            [value.strip() for value in args.validated_edge_targets.split(",") if value.strip()]
            if args.validated_edge_targets
            else None
        ),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "thresholds": {key: value["status"] for key, value in reports.items()}}, indent=2))


if __name__ == "__main__":
    main()
