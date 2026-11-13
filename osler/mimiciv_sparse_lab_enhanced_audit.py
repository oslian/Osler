"""Second-pass sparse-lab audit with event-history and log-scale features.

This is a bounded model-design experiment over the same factual MIMIC-IV
anchor/future pairs.  It adds only information available before the anchor:
the target's previous value, previous-to-current slope, observation interval,
and count.  A second candidate predicts log-space changes for heavy-tailed
markers.  Persistence and the original raw ridge remain the comparators.

The report does not promote a model.  A candidate would still need seven
patient-heldout seeds plus the internal care-unit heldout gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_target_router import _bootstrap_ci, _feature_columns, _fit_ridge, _prepare_features, _round, split_subjects
from mimiciv_sparse_lab_long_horizon_audit import (
    HORIZONS,
    MIMIC_DIR,
    _time_ns,
    TARGETS,
    _event_arrays,
    build_target_anchors,
    load_events,
)

SEEDS = (7, 11, 19, 23, 37, 53, 71)


def add_event_history_features(frame: pd.DataFrame, target: str, arrays) -> pd.DataFrame:
    """Add target history features using strictly earlier observations."""

    output = frame.copy().reset_index(drop=True)
    previous_value = np.full(len(output), np.nan, dtype=np.float64)
    previous_age = np.full(len(output), np.nan, dtype=np.float64)
    previous_delta = np.full(len(output), np.nan, dtype=np.float64)
    previous_slope = np.full(len(output), np.nan, dtype=np.float64)
    observation_count = np.zeros(len(output), dtype=np.float64)
    for stay_id, group in output.groupby("stay_id", sort=False):
        event = arrays.get((stay_id, target))
        if event is None:
            continue
        times, values = event
        positions = group.index.to_numpy(dtype=np.int64)
        anchor_times = _time_ns(group["anchor_time"])
        current_index = np.searchsorted(times, anchor_times, side="left") - 1
        valid = current_index >= 0
        if not valid.any():
            continue
        selected = current_index[valid]
        row_positions = positions[valid]
        previous_value[row_positions] = values[selected]
        previous_age[row_positions] = (anchor_times[valid] - times[selected]) / 3.6e12
        observation_count[row_positions] = selected + 1.0
        older = selected - 1
        has_older = older >= 0
        if has_older.any():
            current_values = values[selected[has_older]]
            older_values = values[older[has_older]]
            interval = np.maximum((times[selected[has_older]] - times[older[has_older]]) / 3.6e12, 0.25)
            previous_delta[row_positions[has_older]] = current_values - older_values
            previous_slope[row_positions[has_older]] = (current_values - older_values) / interval
    output["target_previous_value"] = previous_value
    output["target_previous_age_hr"] = previous_age
    output["target_previous_delta"] = previous_delta
    output["target_previous_slope_per_hr"] = previous_slope
    output["target_observation_count"] = observation_count
    return output


def _delta_summary(rows, truth, pred, baseline, seed):
    mask = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(baseline)
    if not mask.any():
        return {"subjects": 0, "point_delta": None, "bootstrap_95_ci": [None, None], "significant": False}
    work = rows.loc[mask, ["subject_id"]].copy()
    work["delta"] = np.abs(pred[mask] - truth[mask]) - np.abs(baseline[mask] - truth[mask])
    values = work.groupby("subject_id")["delta"].mean().to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=300)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_persistence": bool(point < 0.0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def _add_log_features(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    log_features = []
    for column in features:
        if column not in output:
            continue
        values = pd.to_numeric(output[column], errors="coerce")
        if not values.notna().any():
            continue
        # Log transforms are only applied to level/age/count-like features;
        # signed slopes and deltas remain in their native scale.
        if column.endswith("_t") or column.endswith("_age_hr") or column.endswith("_value") or column.endswith("_count"):
            name = f"log1p_{column}"
            output[name] = np.log1p(np.clip(values.to_numpy(dtype=np.float64), 0.0, None))
            log_features.append(name)
    return output, log_features


def _fit_log_delta(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str], suffix: str) -> np.ndarray:
    current = f"{target}_t"
    future = f"{target}_{suffix}"
    usable = train[train[current].notna() & train[future].notna()].copy()
    if len(usable) < max(20, len(features) + 2):
        return np.full(len(test), np.nan, dtype=np.float64)
    x_train, x_test = _prepare_features(usable, test, features)
    y = np.log1p(np.clip(usable[future].to_numpy(dtype=np.float64), 0.0, None)) - np.log1p(np.clip(usable[current].to_numpy(dtype=np.float64), 0.0, None))
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * 10.0
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y
    predicted_log = np.log1p(np.clip(test[current].to_numpy(dtype=np.float64), 0.0, None)) + x_test @ beta
    # Keep a pathological extrapolation from producing inf/NaN predictions;
    # this is a numerical safety bound, not a clinical plausibility claim.
    return np.expm1(np.clip(predicted_log, -20.0, 20.0))


def _run_split(frame: pd.DataFrame, target: str, horizon: int, seed: int, arrays, group_column: str = "subject_id"):
    suffix = f"tp{horizon}"
    discovery, heldout = split_subjects(frame, seed=seed, discovery_fraction=0.67, group_column=group_column)
    train = frame[frame[group_column].isin(discovery)].copy()
    test = frame[frame[group_column].isin(heldout)].copy()
    current = f"{target}_t"
    future = f"{target}_{suffix}"
    train = train[train[current].notna() & train[future].notna()].copy()
    test = test[test[current].notna() & test[future].notna()].copy()
    if train.empty or test.empty:
        return {"status": "insufficient_pairs", "rows": int(len(test))}
    train = add_event_history_features(train, target, arrays)
    test = add_event_history_features(test, target, arrays)
    base = _feature_columns(train, target)
    extras = ["target_previous_value", "target_previous_age_hr", "target_previous_delta", "target_previous_slope_per_hr", "target_observation_count"]
    event_features = sorted(set(base + extras))
    train_log, log_features_train = _add_log_features(train, event_features)
    test_log, _ = _add_log_features(test, event_features)
    log_features = sorted(set(event_features + log_features_train))
    for name in log_features_train:
        if name not in test_log:
            test_log[name] = np.nan
    raw_event = _fit_ridge(train, test, target, event_features, 10.0, suffix)
    log_event = _fit_log_delta(train_log, test_log, target, log_features, suffix)
    baseline = test[current].to_numpy(dtype=np.float64)
    truth = test[future].to_numpy(dtype=np.float64)
    return {
        "status": "evaluated",
        "rows": int(len(test)),
        "subjects": int(test["subject_id"].nunique()),
        "candidate_event": _delta_summary(test, truth, raw_event, baseline, seed + 11),
        "candidate_log_event": _delta_summary(test, truth, log_event, baseline, seed + 23),
    }


def evaluate_pairs(frame: pd.DataFrame, target: str, horizon: int, arrays) -> dict[str, object]:
    reports = [_run_split(frame, target, horizon, seed, arrays) for seed in SEEDS]
    care = None
    if "first_careunit" in frame and frame["first_careunit"].nunique(dropna=True) >= 2:
        care = _run_split(frame, target, horizon, 9901, arrays, "first_careunit")
    statuses = {}
    for model in ("candidate_event", "candidate_log_event"):
        patient_pass = all(item.get(model, {}).get("significant", False) for item in reports)
        care_pass = bool(care and care.get(model, {}).get("significant", False))
        statuses[model] = "validated" if patient_pass and care_pass else "candidate_or_fallback"
    return {
        "target": target,
        "horizon_hours": horizon,
        "pairs": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()) if len(frame) else 0,
        "patient_splits": reports,
        "care_unit_holdout": care,
        "promotion_status": statuses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--horizons", default=",".join(str(value) for value in HORIZONS))
    parser.add_argument("--events-cache", type=Path, default=Path("/private/tmp/mimiciv_sparse_lab_events.parquet"))
    args = parser.parse_args()
    if args.events_cache.exists():
        print(f"Loading cached sparse events: {args.events_cache}", flush=True)
        events = pd.read_parquet(args.events_cache)
    else:
        events = load_events(args.mimic_dir)
        args.events_cache.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(args.events_cache, index=False)
        print(f"Cached sparse events: {args.events_cache}", flush=True)
    for column in ("target", "first_careunit"):
        if column in events:
            events[column] = events[column].astype(object)
    arrays = _event_arrays(events)
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    horizons = tuple(int(item.strip()) for item in args.horizons.split(",") if item.strip())
    results = []
    for target in targets:
        anchors = build_target_anchors(events, target, horizons, arrays, endpoint=True)
        for horizon in horizons:
            future = f"{target}_tp{horizon}"
            age_column = f"{target}_future_age_hr_{horizon}"
            lower = max(0.5, 0.75 * horizon)
            upper = 1.25 * horizon
            if not anchors.empty:
                pairs = anchors[
                    anchors[future].notna()
                    & anchors[age_column].between(lower, upper, inclusive="both")
                ].copy()
            else:
                pairs = anchors
            if not pairs.empty:
                pairs = add_event_history_features(pairs, target, arrays)
            report = evaluate_pairs(pairs, target, horizon, arrays)
            results.append(report)
            print(target, horizon, report["pairs"], report["promotion_status"], flush=True)
    output = {
        "artifact": "MIMIC-IV sparse laboratory enhanced factual audit",
        "model_comparison": (
            "persistence vs raw event-history ridge vs log-space event-history ridge"
        ),
        "horizon_definition": "future observation must fall in [0.75h, 1.25h], with a 0.5h minimum",
        "results": results,
        "validated_cells": [
            {"target": item["target"], "horizon_hours": item["horizon_hours"], "model": model}
            for item in results
            for model, status in item["promotion_status"].items()
            if status == "validated"
        ],
        "boundary": {
            "anchor_safe": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "row_level_outputs_written": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "validated_cells": output["validated_cells"]}, indent=2))


if __name__ == "__main__":
    main()
