"""Audit whether waveform features improve same-anchor MIMIC nowcasts.

This is deliberately separate from future forecasting.  The candidate uses
only pre-anchor beat features to estimate the value at the anchor, and is
compared with a table-only model and a capacity-matched random-feature model.
All rows remain local-only and the result grants no runtime authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_target_router import _bootstrap_ci, split_subjects
from eicu_nowcasting_audit import nowcast_feature_columns, observed_target_rows
from mimiciv_cross_database_coverage_audit import add_time_holdout_column
from mimiciv_waveform_precision_audit import (
    attach_waveform_features,
    build_segment_index,
    candidate_anchor_rows,
    wave_feature_sets,
)


DEFAULT_TARGETS = ("heart_rate", "map", "respiratory_rate", "o2sat")
DEFAULT_SEEDS = (7, 11, 19, 23, 37, 53, 71)
PLAUSIBLE_TARGETS = {
    "heart_rate": (20.0, 250.0),
    "map": (20.0, 250.0),
    "respiratory_rate": (2.0, 80.0),
    "o2sat": (50.0, 100.0),
}


def _finite_target_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = observed_target_rows(frame, target)
    low, high = PLAUSIBLE_TARGETS[target]
    truth = pd.to_numeric(rows["truth"], errors="coerce")
    return rows[truth.between(low, high)].copy()


def _prepare_features(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return (
            np.zeros((len(train), 0), dtype=np.float64),
            np.zeros((len(predict), 0), dtype=np.float64),
        )
    train_x = train[columns].apply(pd.to_numeric, errors="coerce")
    predict_x = predict[columns].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    predict_x = predict_x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    q01 = train_x.quantile(0.01)
    q99 = train_x.quantile(0.99)
    train_x = train_x.clip(q01, q99, axis=1)
    predict_x = predict_x.clip(q01, q99, axis=1)
    center = train_x.median(axis=0)
    spread = (train_x.quantile(0.75) - train_x.quantile(0.25)).replace(0.0, np.nan)
    spread = spread.fillna(train_x.std(axis=0)).replace(0.0, 1.0).fillna(1.0)
    train_x = (train_x - center) / spread
    predict_x = (predict_x - center) / spread
    return (
        np.nan_to_num(train_x.to_numpy(dtype=np.float64), nan=0.0, posinf=20.0, neginf=-20.0),
        np.nan_to_num(predict_x.to_numpy(dtype=np.float64), nan=0.0, posinf=20.0, neginf=-20.0),
    )


def _fit_ridge(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    features: list[str],
    alpha: float,
    seed: int,
    placebo_count: int = 0,
) -> np.ndarray:
    if train.empty or len(train) < max(30, len(features) + placebo_count + 2):
        return np.full(len(predict), np.nan, dtype=np.float64)
    x_train, x_predict = _prepare_features(train, predict, features)
    if placebo_count:
        rng = np.random.default_rng(seed)
        x_train = np.c_[x_train, rng.normal(size=(len(train), placebo_count))]
        x_predict = np.c_[x_predict, rng.normal(size=(len(predict), placebo_count))]
    y = train["truth"].to_numpy(dtype=np.float64)
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_predict = np.c_[np.ones(len(x_predict)), x_predict]
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y
    return x_predict @ beta


def _evaluate(rows: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, seed: int) -> dict[str, object]:
    truth = rows["truth"].to_numpy(dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(candidate) & np.isfinite(baseline)
    if not finite.any():
        return {"rows": 0, "subjects": 0, "point_delta": None, "ci": [None, None], "significant": False}
    errors = pd.DataFrame({
        "subject_id": rows.loc[finite, "subject_id"].to_numpy(dtype=object),
        "delta": np.abs(candidate[finite] - truth[finite]) - np.abs(baseline[finite] - truth[finite]),
    })
    grouped = errors.groupby("subject_id")["delta"].mean().to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(grouped, seed=seed, samples=300)
    return {
        "rows": int(finite.sum()),
        "subjects": int(len(grouped)),
        "mae_candidate": float(np.abs(candidate[finite] - truth[finite]).mean()),
        "mae_baseline": float(np.abs(baseline[finite] - truth[finite]).mean()),
        "point_delta": float(grouped.mean()),
        "ci": ci,
        "significant": bool(ci and ci[1] < 0.0),
    }


def audit_target(frame: pd.DataFrame, target: str, seeds: tuple[int, ...]) -> dict[str, object]:
    rows = _finite_target_rows(frame, target)
    table_features = nowcast_feature_columns(frame, target)
    wave_features = wave_feature_sets(frame, target, profile="target_group")["wave"]
    candidate_features = sorted(set(table_features + wave_features))
    reports = []
    for seed in seeds:
        discovery_groups, heldout_groups = split_subjects(rows, seed, 0.7)
        discovery = rows[rows["subject_id"].isin(discovery_groups)]
        heldout = rows[rows["subject_id"].isin(heldout_groups)]
        baseline = _fit_ridge(discovery, heldout, table_features, 5.0, seed)
        candidate = _fit_ridge(discovery, heldout, candidate_features, 5.0, seed + 1000)
        placebo = _fit_ridge(discovery, heldout, table_features, 5.0, seed + 2000, len(wave_features))
        candidate_eval = _evaluate(heldout, candidate, baseline, seed + 3000)
        placebo_eval = _evaluate(heldout, candidate, placebo, seed + 4000)
        reports.append({
            "seed": seed,
            "candidate_vs_baseline": candidate_eval,
            "candidate_vs_placebo": placebo_eval,
        })
    external = {}
    for label, column, seed in (
        ("careunit", "first_careunit", 9001),
        ("time", "time_holdout_group", 9901),
    ):
        if column not in rows or rows[column].nunique(dropna=True) < 2:
            external[label] = {"available": False, "passed": False}
            continue
        if label == "time":
            discovery_groups, heldout_groups = {"early"}, {"late"}
        else:
            discovery_groups, heldout_groups = split_subjects(
                rows,
                seed,
                0.7,
                group_column=column,
            )
        discovery = rows[rows[column].isin(discovery_groups)]
        heldout = rows[rows[column].isin(heldout_groups)]
        baseline = _fit_ridge(discovery, heldout, table_features, 5.0, seed)
        candidate = _fit_ridge(discovery, heldout, candidate_features, 5.0, seed + 1000)
        placebo = _fit_ridge(discovery, heldout, table_features, 5.0, seed + 2000, len(wave_features))
        base_eval = _evaluate(heldout, candidate, baseline, seed + 3000)
        placebo_eval = _evaluate(heldout, candidate, placebo, seed + 4000)
        external[label] = {
            "available": True,
            "candidate_vs_baseline": base_eval,
            "candidate_vs_placebo": placebo_eval,
            "passed": bool(base_eval["significant"] and placebo_eval["significant"]),
        }
    random_baseline = sum(bool(x["candidate_vs_baseline"]["significant"]) for x in reports)
    random_placebo = sum(bool(x["candidate_vs_placebo"]["significant"]) for x in reports)
    return {
        "support": {
            "rows": int(len(rows)),
            "subjects": int(rows["subject_id"].nunique()) if len(rows) else 0,
            "table_feature_count": int(len(table_features)),
            "wave_feature_count": int(len(wave_features)),
        },
        "splits": reports,
        "candidate_beats_baseline_splits": sum(
            bool(x["candidate_vs_baseline"]["significant"]) for x in reports
        ),
        "candidate_beats_placebo_splits": sum(
            bool(x["candidate_vs_placebo"]["significant"]) for x in reports
        ),
        "external": external,
        "validated": bool(
            random_baseline == len(seeds)
            and random_placebo == len(seeds)
            and all(value.get("passed", False) for value in external.values() if value.get("available"))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--waveform-root", type=Path, required=True)
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    args = parser.parse_args()

    segments = build_segment_index(args.waveform_root, args.mimic_dir)
    transitions = pd.read_parquet(args.cohort)
    transitions = add_time_holdout_column(transitions, 0.7)
    anchors = candidate_anchor_rows(transitions, segments, args.seconds, 10000, 7105)
    frame, extraction = attach_waveform_features(anchors, args.seconds, feature_mode="beat")
    reports = {target: audit_target(frame, target, DEFAULT_SEEDS) for target in args.targets}
    report = {
        "artifact": "MIMIC-IV Waveform same-anchor nowcast audit",
        "feature_extraction": extraction,
        "targets": reports,
        "boundary": {
            "same_anchor_only": True,
            "future_columns_used": False,
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
            "forecast_authority": False,
            "promotion_allowed": False,
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "targets": {
            target: {
                "support": value["support"],
                "candidate_beats_baseline_splits": value["candidate_beats_baseline_splits"],
                "candidate_beats_placebo_splits": value["candidate_beats_placebo_splits"],
                "external": value["external"],
                "validated": value["validated"],
            }
            for target, value in reports.items()
        }
    }, indent=2))


if __name__ == "__main__":
    main()
