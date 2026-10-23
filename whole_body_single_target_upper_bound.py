"""Single-target upper-bound diagnostic for the joint whole-body JEPA.

This is not a promotion path.  It keeps the complete as-of whole-body input,
but trains the future objective for one target at a time.  If this improves a
target that the joint model misses, the evidence points to multi-task negative
transfer.  If it does not, the limiting factor is the target signal or its
observability rather than shared-task interference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nonlinear_latent_rollout import set_seed
from whole_body_joint_gate import (
    SEEDS,
    _forward,
    _metrics,
    _split_by_column,
    _split_train_calibration,
)
from whole_body_joint_jepa import fit_joint, load_joint_cohort


DEFAULT_TARGETS = (
    "glucose",
    "map",
    "bicarbonate",
    "creatinine",
    "potassium",
    "heart_rate",
)


def _target_results(frame, variables, module_names, target, args):
    target_index = variables.index(target)
    patient = []
    patient_conformal = []
    for seed in SEEDS:
        train_rows, test_rows = _split_by_column(frame, "subject_id", seed)
        fit_rows, calibration_rows = _split_train_calibration(frame, train_rows, seed + 1001)
        model, arrays, losses = fit_joint(
            frame,
            variables,
            module_names,
            fit_rows,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            measurement_process_mode=args.measurement_process_mode,
            cross_forecast_weight=0.0,
            masked_forecast_probability=args.masked_forecast_probability,
            module_mask_probability=args.module_mask_probability,
            world_model=False,
            state_normalization=args.state_normalization,
            hospital_invariance_weight=args.hospital_invariance_weight,
            target_horizon_regime_adapter=args.target_horizon_regime_adapter,
            patient_residual_adapter=args.patient_residual_adapter,
            future_task_balanced_sampling=args.future_task_balanced_sampling,
            uncertainty_gate=args.uncertainty_gate,
            supervised_targets=(target,),
        )
        prediction = _forward(model, arrays, test_rows)
        metrics = _metrics(frame, variables, arrays, prediction, test_rows, {})
        patient.append({key: value for key, value in metrics.items() if key.startswith(f"{target}@")})
        # Reuse the gate's conformal implementation without importing a
        # second evaluator.  The upper-bound report still treats calibration
        # as evidence only and never promotes this model.
        _, _, current, _, _, future, future_mask, _, horizon, _ = arrays[:10]
        scaler = arrays[0]
        cal_prediction = _forward(model, arrays, calibration_rows)
        cal_raw = cal_prediction * scaler.scales + scaler.medians
        test_raw = prediction * scaler.scales + scaler.medians
        future_raw = future * scaler.scales + scaler.medians
        cells = {}
        for horizon_value in sorted(set(horizon[calibration_rows]).intersection(horizon[test_rows])):
            cal_mask = (
                (horizon[calibration_rows] == float(horizon_value))
                & np.asarray(future_mask[calibration_rows, target_index], dtype=bool)
            )
            test_mask = (
                (horizon[test_rows] == float(horizon_value))
                & np.asarray(future_mask[test_rows, target_index], dtype=bool)
            )
            if cal_mask.sum() < 30 or test_mask.sum() < 30:
                continue
            residual = np.abs(
                cal_raw[cal_mask, target_index]
                - future_raw[calibration_rows[cal_mask], target_index]
            )
            q90 = float(np.quantile(residual, 0.90, method="higher"))
            observed = future_raw[test_rows[test_mask], target_index]
            predicted = test_raw[test_mask, target_index]
            coverage = float(
                ((observed >= predicted - q90) & (observed <= predicted + q90)).mean()
            )
            key = f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            cells[key] = {
                "coverage": coverage,
                "calibration_n": int(cal_mask.sum()),
                "test_n": int(test_mask.sum()),
                "q90": q90,
                "pass": bool(0.87 <= coverage <= 0.93),
            }
        patient_conformal.append(cells)
        print(json.dumps({"target": target, "scope": "patient", "seed": seed, "cells": len(patient[-1]), "loss": losses[-1]}), flush=True)

    hospital_train, hospital_test = _split_by_column(frame, "hospitalid", 2026)
    fit_rows, calibration_rows = _split_train_calibration(frame, hospital_train, 3027)
    model, arrays, losses = fit_joint(
        frame,
        variables,
        module_names,
        fit_rows,
        seed=2026,
        epochs=args.epochs,
        batch_size=args.batch_size,
        measurement_process_mode=args.measurement_process_mode,
        cross_forecast_weight=0.0,
        masked_forecast_probability=args.masked_forecast_probability,
        module_mask_probability=args.module_mask_probability,
        world_model=False,
        state_normalization=args.state_normalization,
        hospital_invariance_weight=args.hospital_invariance_weight,
        target_horizon_regime_adapter=args.target_horizon_regime_adapter,
        patient_residual_adapter=args.patient_residual_adapter,
        future_task_balanced_sampling=args.future_task_balanced_sampling,
        uncertainty_gate=args.uncertainty_gate,
        supervised_targets=(target,),
    )
    prediction = _forward(model, arrays, hospital_test)
    metrics = _metrics(frame, variables, arrays, prediction, hospital_test, {})
    hospital = {key: value for key, value in metrics.items() if key.startswith(f"{target}@")}
    return {
        "target": target,
        "patient": patient,
        "patient_conformal": patient_conformal,
        "hospital": hospital,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "supervised_targets": [target],
            "world_model": False,
            "full_body_input": True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--modules", required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--max-stays", type=int, default=200)
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument("--alignment", choices=("exact", "asof"), default="asof")
    parser.add_argument("--asof-max-age-hours", type=float, default=6.0)
    parser.add_argument("--asof-age-policy", choices=("fixed", "horizon_scaled"), default="horizon_scaled")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--measurement-process-mode", choices=("observed", "neutralized"), default="neutralized")
    parser.add_argument("--state-normalization", choices=("standard", "robust"), default="robust")
    parser.add_argument("--hospital-invariance-weight", type=float, default=0.05)
    parser.add_argument("--masked-forecast-probability", type=float, default=0.50)
    parser.add_argument("--module-mask-probability", type=float, default=0.75)
    parser.add_argument("--target-horizon-regime-adapter", action="store_true")
    parser.add_argument("--patient-residual-adapter", action="store_true")
    parser.add_argument("--future-task-balanced-sampling", action="store_true")
    parser.add_argument("--uncertainty-gate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    frame, variables, module_names = load_joint_cohort(
        args.data_root,
        modules,
        tuple(int(value) for value in args.horizons.split(",") if value.strip()),
        args.max_stays,
        min_modules=args.min_modules,
        alignment=args.alignment,
        asof_max_age_hours=args.asof_max_age_hours,
        asof_age_policy=args.asof_age_policy,
    )
    unknown = sorted(set(targets) - set(variables))
    if unknown:
        raise ValueError(f"Targets absent from cohort: {unknown}")
    set_seed(20260717)
    reports = [_target_results(frame, variables, module_names, target, args) for target in targets]
    output = {
        "schema": "whole_body_single_target_upper_bound.v1",
        "data_root": str(args.data_root.resolve()),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "hospitals": int(frame["hospitalid"].nunique()),
        "modules": list(module_names),
        "targets": list(targets),
        "reports": reports,
        "interpretation": {
            "purpose": "diagnose multi-task negative transfer, not promote a checkpoint",
            "full_body_input": True,
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "targets": list(targets), "rows": len(frame)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
