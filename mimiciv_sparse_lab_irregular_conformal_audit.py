"""Split-conformal interval audit for validated irregular-time sparse labs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_target_router import _feature_columns, _fit_ridge
from mimiciv_sparse_lab_enhanced_audit import (
    _add_log_features,
    _fit_log_delta,
    add_event_history_features,
)
from mimiciv_sparse_lab_irregular_time_audit import (
    TIME_BINS,
    build_irregular_pairs,
)
from mimiciv_sparse_lab_long_horizon_audit import (
    MIMIC_DIR,
    _event_arrays,
    load_events,
)
from teacher_anchored_joint_jepa_audit import (
    _patient_cluster_conformal_quantile,
    _patient_coverage_diagnostics,
)


SEEDS = (7, 11, 19, 23, 37, 53, 71)
LEVEL = 0.90
VALIDATED_CELLS = (
    ("inr", "6_24h", "log_event"),
    ("inr", "24_72h", "log_event"),
    ("ptt", "0_6h", "log_event"),
    ("fibrinogen", "6_24h", "raw_event"),
    ("cortisol", "0_6h", "raw_event"),
)


def _split_groups(frame: pd.DataFrame, group_column: str, seed: int) -> tuple[set[object], set[object], set[object]]:
    groups = np.asarray(sorted(frame[group_column].dropna().unique().tolist()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    if len(groups) < 6:
        return set(), set(), set()
    fit_end = max(1, int(round(len(groups) * 0.50)))
    cal_end = min(len(groups) - 1, fit_end + max(1, int(round(len(groups) * 0.25))))
    return set(groups[:fit_end]), set(groups[fit_end:cal_end]), set(groups[cal_end:])


def _features(train: pd.DataFrame, predict: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base = _feature_columns(train, target)
    extras = [
        "target_previous_value",
        "target_previous_age_hr",
        "target_previous_delta",
        "target_previous_slope_per_hr",
        "target_observation_count",
        "forecast_horizon_hr",
    ]
    columns = sorted(set(base + extras))
    train_log, log_columns = _add_log_features(train, columns)
    predict_log, _ = _add_log_features(predict, columns)
    for column in log_columns:
        if column not in predict_log:
            predict_log[column] = np.nan
    return train_log, predict_log, sorted(set(columns + log_columns))


def _predict(train: pd.DataFrame, predict: pd.DataFrame, target: str, model: str) -> np.ndarray:
    suffix = "future"
    train_log, predict_log, columns = _features(train, predict, target)
    if model == "log_event":
        return _fit_log_delta(train_log, predict_log, target, columns, suffix)
    raw_columns = [column for column in columns if not column.startswith("log1p_")]
    return _fit_ridge(train, predict, target, raw_columns, 10.0, suffix)


def _coverage(calibration: pd.DataFrame, test: pd.DataFrame, target: str, model: str, seed: int) -> dict[str, object]:
    prediction = _predict(calibration, test, target, model)
    # Fit a second prediction on the calibration rows using the fit model is
    # handled by the caller; this helper receives residuals separately.
    truth = test[f"{target}_future"].to_numpy(dtype=np.float64)
    return {"prediction": prediction, "truth": truth}


def _cell_audit(frame: pd.DataFrame, target: str, time_bin: str, model: str, group_column: str, seed: int) -> dict[str, object]:
    lower, upper = next((lo, hi) for name, lo, hi in TIME_BINS if name == time_bin)
    subset = frame[frame["forecast_horizon_hr"].between(lower, upper, inclusive="right")].copy()
    fit_groups, cal_groups, test_groups = _split_groups(subset, group_column, seed)
    if not fit_groups or not cal_groups or not test_groups:
        return {"status": "insufficient_groups", "coverage": None, "calibration_subjects": 0, "test_subjects": 0}
    fit = subset[subset[group_column].isin(fit_groups)].copy()
    calibration = subset[subset[group_column].isin(cal_groups)].copy()
    test = subset[subset[group_column].isin(test_groups)].copy()
    # History features are already attached by the irregular pair builder.
    fit_pred = _predict(fit, calibration, target, model)
    test_pred = _predict(fit, test, target, model)
    cal_truth = calibration[f"{target}_future"].to_numpy(dtype=np.float64)
    test_truth = test[f"{target}_future"].to_numpy(dtype=np.float64)
    cal_residuals = np.abs(cal_truth - fit_pred)
    cal_valid = np.isfinite(cal_residuals)
    cal_residuals = cal_residuals[cal_valid]
    calibration_valid = calibration.iloc[np.flatnonzero(cal_valid)].reset_index(
        drop=True
    )
    if len(cal_residuals) < 100 or not np.isfinite(test_pred).any():
        return {
            "status": "insufficient_calibration",
            "coverage": None,
            "calibration_rows": int(len(cal_residuals)),
            "calibration_subjects": int(calibration["subject_id"].nunique()),
            "test_subjects": int(test["subject_id"].nunique()),
        }
    quantile, _, calibration_metadata = _patient_cluster_conformal_quantile(
        calibration_valid,
        np.arange(len(calibration_valid), dtype=np.int64),
        cal_residuals,
        coverage=LEVEL,
    )
    valid = np.isfinite(test_truth) & np.isfinite(test_pred)
    covered = np.abs(test_truth[valid] - test_pred[valid]) <= quantile
    test_valid = test.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    coverage = (
        _patient_coverage_diagnostics(
            test_valid,
            np.arange(len(test_valid), dtype=np.int64),
            covered,
            seed=seed,
        )
        if valid.any()
        else {
            "row_coverage": float("nan"),
            "patient_equalized_coverage": float("nan"),
            "patient_coverage_ci95": [float("nan"), float("nan")],
            "test_subjects": 0,
        }
    )
    patient_coverage = coverage["patient_equalized_coverage"]
    return {
        "status": "evaluated",
        "coverage": (
            round(patient_coverage, 6)
            if np.isfinite(patient_coverage)
            else None
        ),
        "row_coverage": (
            round(coverage["row_coverage"], 6)
            if np.isfinite(coverage["row_coverage"])
            else None
        ),
        "patient_equalized_coverage": (
            round(patient_coverage, 6)
            if np.isfinite(patient_coverage)
            else None
        ),
        "patient_coverage_ci95": coverage["patient_coverage_ci95"],
        "coverage_gate_passed": bool(
            np.isfinite(patient_coverage)
            and coverage["test_subjects"] >= 20
            and 0.87 <= patient_coverage <= 0.93
        ),
        "interval_level": LEVEL,
        "radius": round(quantile, 6),
        "calibration_method": calibration_metadata,
        "calibration_rows": int(len(cal_residuals)),
        "calibration_subjects": int(calibration["subject_id"].nunique()),
        "test_rows": int(valid.sum()),
        "test_subjects": int(coverage["test_subjects"]),
    }


def audit_cell(frame: pd.DataFrame, target: str, time_bin: str, model: str) -> dict[str, object]:
    patient = [_cell_audit(frame, target, time_bin, model, "subject_id", seed) for seed in SEEDS]
    care = _cell_audit(frame, target, time_bin, model, "first_careunit", 9901) if frame["first_careunit"].nunique(dropna=True) >= 6 else {"status": "insufficient_groups"}
    patient_pass = sum(bool(item.get("coverage_gate_passed")) for item in patient)
    status = "calibrated" if patient_pass == len(SEEDS) and care.get("coverage_gate_passed") else "needs_calibration_audit"
    return {
        "target": target,
        "time_bin": time_bin,
        "model": model,
        "patient_splits": patient,
        "patient_coverage_pass_count": int(patient_pass),
        "care_unit_holdout": care,
        "interval_status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--events-cache", type=Path, default=Path("/private/tmp/mimiciv_sparse_lab_events.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.events_cache.exists():
        events = pd.read_parquet(args.events_cache)
    else:
        events = load_events(args.mimic_dir)
        args.events_cache.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(args.events_cache, index=False)
        print(f"Saved reusable events cache: {args.events_cache}", flush=True)
    for column in ("target", "first_careunit"):
        if column in events:
            events[column] = events[column].astype(object)
    arrays = _event_arrays(events)
    reports = []
    for target, time_bin, model in VALIDATED_CELLS:
        print(f"Conformal audit {target} {time_bin} {model}", flush=True)
        frame = build_irregular_pairs(events, target, arrays)
        frame = add_event_history_features(frame, target, arrays)
        reports.append(audit_cell(frame, target, time_bin, model))
    output = {
        "artifact": "MIMIC-IV sparse irregular-time split-conformal audit",
        "conformal_method": "patient_cluster_equal_weight",
        "interval_level": LEVEL,
        "coverage_gate": [0.87, 0.93],
        "reports": reports,
        "calibrated_cells": [
            {"target": r["target"], "time_bin": r["time_bin"], "model": r["model"]}
            for r in reports if r["interval_status"] == "calibrated"
        ],
        "boundary": {
            "future_lab_used_as_feature": False,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "row_level_outputs_written": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "calibrated_cells": output["calibrated_cells"]}, indent=2))


if __name__ == "__main__":
    main()
