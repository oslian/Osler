"""Irregular-time sparse-lab forecast audit.

Fixed 24/48-hour endpoint labels are too sparse for many MIMIC-IV labs. This
audit treats the requested forecast horizon as an explicit query variable and
uses the first future observation between 0.5 and 72 hours as the factual
target. Results are stratified into 0--6h, 6--24h, and 24--72h bins.

The elapsed horizon is a query condition, not a patient outcome feature: the
model is asked for a value at a specified time. The audit never uses the
future lab value or any post-anchor measurement as a feature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_target_router import _bootstrap_ci, _feature_columns, _fit_ridge, _prepare_features, _round, split_subjects
from mimiciv_sparse_lab_enhanced_audit import (
    SEEDS,
    _add_log_features,
    _delta_summary,
    _event_arrays,
    _fit_log_delta,
    add_event_history_features,
)
from mimiciv_sparse_lab_long_horizon_audit import MIMIC_DIR, TARGETS, _time_ns, load_events


TIME_BINS = (
    ("0_6h", 0.5, 6.0),
    ("6_24h", 6.0, 24.0),
    ("24_72h", 24.0, 72.0),
)


def build_irregular_pairs(events: pd.DataFrame, target: str, arrays) -> pd.DataFrame:
    rows = []
    target_events = events[events["target"].eq(target)]
    for stay_id, group in target_events.groupby("stay_id", sort=False):
        ordered = group.sort_values("charttime")
        times = _time_ns(ordered["charttime"])
        values = ordered["value"].to_numpy(dtype=np.float64)
        for index, (anchor_ns, current) in enumerate(zip(times, values)):
            first_future = int(np.searchsorted(times, anchor_ns + int(0.5 * 3.6e12), side="left"))
            last_future = int(np.searchsorted(times, anchor_ns + int(72.0 * 3.6e12), side="right"))
            if first_future >= last_future:
                continue
            for future_index in range(first_future, last_future):
                elapsed = float((times[future_index] - anchor_ns) / 3.6e12)
                rows.append({
                    "subject_id": ordered["subject_id"].iloc[0],
                    "stay_id": stay_id,
                    "first_careunit": ordered["first_careunit"].iloc[0],
                    "anchor_time": pd.Timestamp(anchor_ns),
                    "hours_since_intime": float((pd.Timestamp(anchor_ns) - ordered["intime"].iloc[0]).total_seconds() / 3600.0),
                    "target": target,
                    f"{target}_t": float(current),
                    "forecast_horizon_hr": elapsed,
                    f"{target}_future": float(values[future_index]),
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Add the latest anchor-time state values using the shared future-safe
    # anchor mechanism, then keep only pre-anchor features.
    state = []
    for source in sorted({key[1] for key in arrays}):
        if f"{source}_t" in frame:
            continue
        values = np.full(len(frame), np.nan, dtype=np.float64)
        ages = np.full(len(frame), np.nan, dtype=np.float64)
        for stay_id, group in frame.groupby("stay_id", sort=False):
            event = arrays.get((stay_id, source))
            if event is None:
                continue
            event_times, event_values = event
            positions = group.index.to_numpy(dtype=np.int64)
            anchor_times = _time_ns(group["anchor_time"])
            selected = np.searchsorted(event_times, anchor_times, side="right") - 1
            valid = selected >= 0
            if valid.any():
                values[positions[valid]] = event_values[selected[valid]]
                ages[positions[valid]] = (anchor_times[valid] - event_times[selected[valid]]) / 3.6e12
        state.extend([pd.Series(values, name=f"{source}_t"), pd.Series(ages, name=f"{source}_age_hr")])
    if state:
        frame = pd.concat([frame.reset_index(drop=True), pd.concat(state, axis=1)], axis=1)
    else:
        frame = frame.reset_index(drop=True)
    return add_event_history_features(frame, target, arrays)


def _run_split(frame: pd.DataFrame, target: str, seed: int, group_column: str = "subject_id") -> dict[str, object]:
    discovery, heldout = split_subjects(frame, seed=seed, discovery_fraction=0.67, group_column=group_column)
    train = frame[frame[group_column].isin(discovery)].copy()
    test = frame[frame[group_column].isin(heldout)].copy()
    if train.empty or test.empty:
        return {"status": "insufficient_pairs", "rows": int(len(test))}
    current = f"{target}_t"
    future = f"{target}_future"
    base = _feature_columns(train, target)
    extra = ["target_previous_value", "target_previous_age_hr", "target_previous_delta", "target_previous_slope_per_hr", "target_observation_count", "forecast_horizon_hr"]
    features = sorted(set(base + extra))
    train_log, log_features_train = _add_log_features(train, features)
    test_log, _ = _add_log_features(test, features)
    for column in log_features_train:
        if column not in test_log:
            test_log[column] = np.nan
    log_features = sorted(set(features + log_features_train))
    raw = _fit_ridge(train, test, target, features, 10.0, "future")
    log = _fit_log_delta(train_log, test_log, target, log_features, "future")
    truth = test[future].to_numpy(dtype=np.float64)
    baseline = test[current].to_numpy(dtype=np.float64)
    return {
        "status": "evaluated",
        "rows": int(len(test)),
        "subjects": int(test["subject_id"].nunique()),
        "raw_event": _delta_summary(test, truth, raw, baseline, seed + 11),
        "log_event": _delta_summary(test, truth, log, baseline, seed + 23),
    }


def _run_forward_trajectory_split(
    frame: pd.DataFrame,
    target: str,
    *,
    discovery_fraction: float = 0.75,
) -> dict[str, object]:
    """Train on each stay's earlier events and evaluate its later events.

    MIMIC calendar dates are independently shifted by patient, so a global
    calendar cutoff is not a valid deployment-time split. This split instead
    asks the online question we can identify: after learning from earlier
    measurements in a stay, does the model generalize to later measurements?
    """

    train_parts = []
    test_parts = []
    for _, group in frame.groupby("stay_id", sort=False):
        ordered = group.sort_values(["anchor_time", "forecast_horizon_hr"])
        if len(ordered) < 4:
            continue
        cut = max(1, min(len(ordered) - 1, int(np.floor(len(ordered) * discovery_fraction))))
        train_parts.append(ordered.iloc[:cut])
        test_parts.append(ordered.iloc[cut:])
    if not train_parts or not test_parts:
        return {"status": "insufficient_pairs", "rows": 0}
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    current = f"{target}_t"
    future = f"{target}_future"
    base = _feature_columns(train, target)
    extra = [
        "target_previous_value",
        "target_previous_age_hr",
        "target_previous_delta",
        "target_previous_slope_per_hr",
        "target_observation_count",
        "forecast_horizon_hr",
    ]
    features = sorted(set(base + extra))
    train_log, log_features_train = _add_log_features(train, features)
    test_log, _ = _add_log_features(test, features)
    for column in log_features_train:
        if column not in test_log:
            test_log[column] = np.nan
    log_features = sorted(set(features + log_features_train))
    raw = _fit_ridge(train, test, target, features, 10.0, "future")
    log = _fit_log_delta(train_log, test_log, target, log_features, "future")
    truth = test[future].to_numpy(dtype=np.float64)
    baseline = test[current].to_numpy(dtype=np.float64)
    return {
        "status": "evaluated",
        "definition": "within-stay earlier 75% -> later 25%; no global calendar claim",
        "rows": int(len(test)),
        "subjects": int(test["subject_id"].nunique()),
        "stays": int(test["stay_id"].nunique()),
        "raw_event": _delta_summary(test, truth, raw, baseline, 9953),
        "log_event": _delta_summary(test, truth, log, baseline, 9967),
    }


def evaluate_bin(frame: pd.DataFrame, target: str, lower: float, upper: float, arrays) -> dict[str, object]:
    subset = frame[frame["forecast_horizon_hr"].between(lower, upper, inclusive="right")].copy()
    reports = [_run_split(subset, target, seed) for seed in SEEDS] if not subset.empty else []
    care = None
    if not subset.empty and subset["first_careunit"].nunique(dropna=True) >= 2:
        care = _run_split(subset, target, 9901, "first_careunit")
    forward = (
        _run_forward_trajectory_split(subset, target)
        if not subset.empty
        else None
    )
    promotion = {}
    for model in ("raw_event", "log_event"):
        promotion[model] = "validated" if (
            reports and all(item.get(model, {}).get("significant", False) for item in reports)
            and care and care.get(model, {}).get("significant", False)
            and forward and forward.get(model, {}).get("significant", False)
        ) else "candidate_or_fallback"
    horizons = subset["forecast_horizon_hr"].to_numpy(dtype=np.float64) if not subset.empty else np.asarray([])
    return {
        "pairs": int(len(subset)),
        "subjects": int(subset["subject_id"].nunique()) if not subset.empty else 0,
        "observed_horizon_hr": {
            "median": _round(float(np.median(horizons))) if len(horizons) else None,
            "q10": _round(float(np.quantile(horizons, 0.10))) if len(horizons) else None,
            "q90": _round(float(np.quantile(horizons, 0.90))) if len(horizons) else None,
        },
        "patient_splits": reports,
        "care_unit_holdout": care,
        "forward_trajectory_holdout": forward,
        "promotion_status": promotion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--events-cache", type=Path, default=Path("/private/tmp/mimiciv_sparse_lab_events.parquet"))
    parser.add_argument("--targets", default=",".join(TARGETS))
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
    results = []
    for target in tuple(item.strip() for item in args.targets.split(",") if item.strip()):
        print(f"Building irregular pairs: {target}", flush=True)
        frame = build_irregular_pairs(events, target, arrays)
        for name, lower, upper in TIME_BINS:
            report = evaluate_bin(frame, target, lower, upper, arrays)
            report.update({"target": target, "time_bin": name, "lower_hr": lower, "upper_hr": upper})
            results.append(report)
            print(target, name, report["pairs"], report["promotion_status"], flush=True)
    output = {
        "artifact": "MIMIC-IV sparse laboratory irregular-time factual audit",
        "horizon_definition": "forecast horizon is an explicit query variable; labels are first future labs in 0.5-72h",
        "results": results,
        "validated_cells": [
            {"target": item["target"], "time_bin": item["time_bin"], "model": model}
            for item in results
            for model, status in item["promotion_status"].items()
            if status == "validated"
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
    print(json.dumps({"output": str(args.output), "validated_cells": output["validated_cells"]}, indent=2))


if __name__ == "__main__":
    main()
