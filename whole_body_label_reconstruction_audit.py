"""Evaluate a future-label reconstruction candidate on common raw labels.

An interpolated future-label candidate cannot be compared to a raw-label model
by looking at each run's own MAE: the labels differ. This audit trains on one
label source but always scores held-out predictions against the same strict
raw-event labels. It is diagnostic only and never promotes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_joint_gate import SEEDS, _forward, _split_by_column
from whole_body_joint_jepa import fit_joint, load_joint_cohort
from whole_body_scoring_audit import _bootstrap_ci, _subject_mae


def _row_key(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[["stay_id", "_source_horizon", "_time_hours"]]
    )


def _align_raw_evaluation_frame(raw: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Order raw labels to the candidate rows without using future metadata."""

    raw_index = _row_key(raw)
    train_index = _row_key(train)
    if not raw_index.is_unique or not train_index.is_unique:
        raise ValueError("common-label audit requires unique stay/horizon/anchor rows")
    positions = raw_index.get_indexer(train_index)
    if (positions < 0).any():
        raise ValueError("candidate rows are not a subset of the raw-label cohort")
    return raw.iloc[positions].reset_index(drop=True)


def _raw_target_mask(raw: pd.DataFrame, target: str) -> np.ndarray:
    current = pd.to_numeric(raw[f"{target}_t"], errors="coerce").to_numpy(dtype=np.float64)
    future = pd.to_numeric(raw[f"future_{target}"], errors="coerce").to_numpy(dtype=np.float64)
    weight = pd.to_numeric(
        raw.get(f"future_label_weight_{target}", 1.0), errors="coerce"
    )
    if not isinstance(weight, pd.Series):
        weight = pd.Series(weight, index=raw.index)
    return (
        np.isfinite(current)
        & np.isfinite(future)
        & (weight.to_numpy(dtype=np.float64) >= 1.0 - 1e-6)
    )


def _cell_metric(
    train_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    variables: tuple[str, ...],
    arrays,
    prediction: np.ndarray,
    rows: np.ndarray,
    train_rows: np.ndarray,
    target: str,
    horizon: float,
    seed: int,
) -> dict | None:
    if target not in variables:
        return None
    index = variables.index(target)
    rows = np.asarray(rows, dtype=np.int64)
    keep = (
        (pd.to_numeric(raw_frame["_source_horizon"], errors="coerce").to_numpy(dtype=np.float64)[rows] == float(horizon))
        & _raw_target_mask(raw_frame, target)[rows]
    )
    selected = rows[keep]
    if len(selected) < 30:
        return None
    local = np.flatnonzero(keep)
    scaler = arrays[0]
    truth = pd.to_numeric(raw_frame.iloc[selected][f"future_{target}"], errors="coerce").to_numpy(dtype=np.float64)
    current = pd.to_numeric(raw_frame.iloc[selected][f"{target}_t"], errors="coerce").to_numpy(dtype=np.float64)
    candidate = prediction[local, index] * scaler.scales[index] + scaler.medians[index]
    candidate_error = np.abs(candidate - truth)
    persistence_error = np.abs(current - truth)
    candidate_by_subject = _subject_mae(train_frame, selected, candidate_error)
    persistence_by_subject = _subject_mae(train_frame, selected, persistence_error)
    paired = {
        subject: candidate_by_subject[subject] - persistence_by_subject[subject]
        for subject in candidate_by_subject
        if subject in persistence_by_subject
    }
    bootstrap = _bootstrap_ci(paired.values(), seed=20260722 + int(seed))
    return {
        "target": target,
        "horizon_hours": float(horizon),
        "n_rows": int(len(selected)),
        "n_subjects": int(len(paired)),
        "raw_event_mae": float(candidate_error.mean()),
        "persistence_mae": float(persistence_error.mean()),
        "patient_equal_delta_vs_persistence": float(np.mean(list(paired.values()))) if paired else None,
        "patient_bootstrap": bootstrap,
        "pass_raw_event_bootstrap": bool(
            bootstrap["high"] is not None and bootstrap["high"] < 0.0
        ),
    }


def run(args: argparse.Namespace) -> dict:
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    raw_frame, raw_variables, raw_modules = load_joint_cohort(
        args.raw_data_root,
        modules,
        horizons,
        args.max_stays,
        min_modules=args.min_modules,
        alignment=args.alignment,
        asof_max_age_hours=args.asof_max_age_hours,
        asof_age_policy=args.asof_age_policy,
    )
    train_frame, variables, module_names = load_joint_cohort(
        args.train_data_root,
        modules,
        horizons,
        args.max_stays,
        min_modules=args.min_modules,
        alignment=args.alignment,
        asof_max_age_hours=args.asof_max_age_hours,
        asof_age_policy=args.asof_age_policy,
    )
    if tuple(raw_variables) != tuple(variables) or tuple(raw_modules) != tuple(module_names):
        raise ValueError("raw and candidate roots expose different module/variable contracts")
    raw_frame = _align_raw_evaluation_frame(raw_frame, train_frame)
    reports = []
    for seed in SEEDS:
        train_rows, test_rows = _split_by_column(train_frame, "subject_id", seed)
        model, arrays, losses = fit_joint(
            train_frame,
            variables,
            module_names,
            train_rows,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cross_forecast_weight=args.cross_forecast_weight,
            measurement_process_mode=args.measurement_process_mode,
            world_model=args.world_model,
            state_normalization=args.state_normalization,
            hospital_invariance_weight=args.hospital_invariance_weight,
        )
        prediction = _forward(model, arrays, test_rows)
        cells = {}
        for target in targets:
            for horizon in horizons:
                cell = _cell_metric(
                    train_frame,
                    raw_frame,
                    variables,
                    arrays,
                    prediction,
                    test_rows,
                    train_rows,
                    target,
                    horizon,
                    seed,
                )
                if cell is not None:
                    cells[f"{target}@{horizon}h"] = cell
        reports.append({"seed": int(seed), "loss": float(losses[-1]), "cells": cells})
        print(json.dumps({"seed": int(seed), "cells": len(cells), "loss": float(losses[-1])}), flush=True)
    return {
        "schema": "whole_body_label_reconstruction_audit.v1",
        "train_data_root": str(args.train_data_root.resolve()),
        "raw_evaluation_root": str(args.raw_data_root.resolve()),
        "modules": list(module_names),
        "variables": list(variables),
        "targets": list(targets),
        "horizons": list(horizons),
        "rows": int(len(train_frame)),
        "subjects": int(train_frame["subject_id"].nunique()),
        "reports": reports,
        "evaluation_contract": {
            "common_heldout_raw_event_labels": True,
            "interpolated_training_labels_never_used_for_evaluation": True,
            "patient_heldout_seeds": list(SEEDS),
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "promotion_status": "candidate_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--train-data-root", type=Path, required=True)
    parser.add_argument("--modules", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--max-stays", type=int, default=200)
    parser.add_argument("--min-modules", type=int, default=1)
    parser.add_argument("--alignment", choices=("exact", "asof"), default="asof")
    parser.add_argument("--asof-max-age-hours", type=float, default=6.0)
    parser.add_argument("--asof-age-policy", choices=("fixed", "horizon_scaled"), default="horizon_scaled")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cross-forecast-weight", type=float, default=1.0)
    parser.add_argument("--measurement-process-mode", choices=("observed", "neutralized"), default="neutralized")
    parser.add_argument("--world-model", action="store_true")
    parser.add_argument("--state-normalization", choices=("standard", "robust"), default="robust")
    parser.add_argument("--hospital-invariance-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rows": report["rows"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
