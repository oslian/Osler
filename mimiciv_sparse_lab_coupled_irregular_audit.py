"""Coupled irregular-time audit for sparse MIMIC-IV laboratory targets.

The existing sparse audit uses the target's own event history.  This bounded
experiment adds strictly pre-anchor history summaries for the other measured
physiology streams as well: latest value/age are already present, while this
script adds each source's previous delta, slope, and observation count.

It is still factual prediction only.  A coupled candidate is promoted only if
it beats persistence and the target-history-only candidate on every patient
split and the internal care-unit holdout.  No causal or hidden-state claim is
made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_target_router import _feature_columns, _fit_ridge, split_subjects
from mimiciv_sparse_lab_enhanced_audit import (
    SEEDS,
    _add_log_features,
    _fit_log_delta,
    _delta_summary,
    add_event_history_features,
)
from mimiciv_sparse_lab_irregular_time_audit import (
    TIME_BINS,
    build_irregular_pairs,
)
from mimiciv_sparse_lab_long_horizon_audit import (
    MIMIC_DIR,
    TARGETS,
    _event_arrays,
    _time_ns,
    load_events,
)


COUPLED_SOURCE_MAP: dict[str, tuple[str, ...]] = {
    "cortisol": ("glucose", "sodium", "potassium", "bicarbonate", "anion_gap", "creatinine", "bun", "albumin"),
    "pth": ("calcium", "phosphate", "creatinine", "bun", "albumin"),
    "tsh": ("sodium", "potassium", "glucose", "bicarbonate", "creatinine", "bun", "albumin"),
    "total_t4": ("tsh", "albumin", "bilirubin", "creatinine", "bun", "sodium"),
    "free_t4": ("tsh", "albumin", "bilirubin", "creatinine", "bun", "sodium"),
    "troponin_i": ("troponin_t", "ck_mb", "cpk", "bnp", "creatinine", "bun", "potassium"),
    "troponin_t": ("troponin_i", "ck_mb", "cpk", "bnp", "creatinine", "bun", "potassium"),
    "bnp": ("troponin_t", "troponin_i", "ck_mb", "creatinine", "bun", "albumin", "sodium"),
    "ck_mb": ("troponin_t", "troponin_i", "cpk", "bnp", "creatinine", "bun", "potassium"),
    "cpk": ("ck_mb", "troponin_t", "creatinine", "bun", "potassium", "phosphate"),
    "inr": ("bilirubin", "bilirubin_direct", "albumin", "hemoglobin", "fibrinogen", "ptt", "creatinine", "bun"),
    "ptt": ("inr", "fibrinogen", "hemoglobin", "creatinine", "bun", "bilirubin"),
    "fibrinogen": ("inr", "ptt", "crp", "bilirubin", "albumin", "hemoglobin"),
    "crp": ("fibrinogen", "albumin", "bilirubin", "creatinine", "bun", "cortisol"),
    "ferritin": ("hemoglobin", "bilirubin", "crp", "albumin", "creatinine", "bun"),
    "bilirubin": ("bilirubin_direct", "albumin", "inr", "fibrinogen", "hemoglobin", "creatinine", "bun"),
    "bilirubin_direct": ("bilirubin", "albumin", "inr", "hemoglobin", "creatinine", "bun"),
}


def add_coupled_history_features(
    frame: pd.DataFrame,
    target: str,
    arrays: dict[tuple[object, str], tuple[np.ndarray, np.ndarray]],
    sources: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Add other-target history using observations strictly before anchor."""

    output = frame.copy().reset_index(drop=True)
    sources = tuple(
        source for source in (sources or COUPLED_SOURCE_MAP.get(target, ()))
        if source != target and any(name == source for _, name in arrays)
    )
    anchor_by_stay = {
        stay_id: (group.index.to_numpy(dtype=np.int64), _time_ns(group["anchor_time"]))
        for stay_id, group in output.groupby("stay_id", sort=False)
    }

    for source in sources:
        delta = np.full(len(output), np.nan, dtype=np.float64)
        slope = np.full(len(output), np.nan, dtype=np.float64)
        count = np.zeros(len(output), dtype=np.float64)
        age = np.full(len(output), np.nan, dtype=np.float64)
        for stay_id, (positions, anchor_times) in anchor_by_stay.items():
            event = arrays.get((stay_id, source))
            if event is None:
                continue
            times, values = event
            current_index = np.searchsorted(times, anchor_times, side="right") - 1
            valid = current_index >= 0
            if not valid.any():
                continue
            selected = current_index[valid]
            row_positions = positions[valid]
            count[row_positions] = selected + 1.0
            age[row_positions] = (anchor_times[valid] - times[selected]) / 3.6e12
            older = selected - 1
            has_older = older >= 0
            if has_older.any():
                intervals = np.maximum(
                    (times[selected[has_older]] - times[older[has_older]]) / 3.6e12,
                    0.25,
                )
                delta[row_positions[has_older]] = (
                    values[selected[has_older]] - values[older[has_older]]
                )
                slope[row_positions[has_older]] = delta[row_positions[has_older]] / intervals
        output[f"history_{source}_age_hr"] = age
        output[f"history_{source}_delta"] = delta
        output[f"history_{source}_slope_per_hr"] = slope
        output[f"history_{source}_count"] = count
    return output


def _model_features(train: pd.DataFrame, target: str) -> list[str]:
    base = _feature_columns(train, target)
    extras = [
        "target_previous_value",
        "target_previous_age_hr",
        "target_previous_delta",
        "target_previous_slope_per_hr",
        "target_observation_count",
        "forecast_horizon_hr",
    ]
    coupled = [
        column
        for column in train.columns
        if column.startswith("history_")
        and (
            column.endswith("_age_hr")
            or column.endswith("_delta")
            or column.endswith("_slope_per_hr")
            or column.endswith("_count")
        )
    ]
    return sorted(set(base + extras + coupled))


def _fit_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = _model_features(train, target)
    train_log, log_columns = _add_log_features(train, features)
    test_log, _ = _add_log_features(test, features)
    for column in log_columns:
        if column not in test_log:
            test_log[column] = np.nan
    log_features = sorted(set(features + log_columns))

    coupled_raw = _fit_ridge(train, test, target, features, 10.0, "future")
    coupled_log = _fit_log_delta(train_log, test_log, target, log_features, "future")

    # Refit the target-history-only candidate on the same rows as a direct
    # increment comparator.  It is not used to select on the final test set.
    history_features = [
        column for column in features
        if not column.startswith("history_")
    ]
    history_raw = _fit_ridge(train, test, target, history_features, 10.0, "future")
    history_train_log, history_log_columns = _add_log_features(train, history_features)
    history_test_log, _ = _add_log_features(test, history_features)
    for column in history_log_columns:
        if column not in history_test_log:
            history_test_log[column] = np.nan
    history_log_features = sorted(set(history_features + history_log_columns))
    history_log = _fit_log_delta(
        history_train_log,
        history_test_log,
        target,
        history_log_features,
        "future",
    )
    return coupled_raw, coupled_log, history_raw, history_log


def _run_split(
    frame: pd.DataFrame,
    target: str,
    seed: int,
    group_column: str = "subject_id",
) -> dict[str, object]:
    discovery, heldout = split_subjects(
        frame,
        seed=seed,
        discovery_fraction=0.67,
        group_column=group_column,
    )
    train = frame[frame[group_column].isin(discovery)].copy()
    test = frame[frame[group_column].isin(heldout)].copy()
    current = f"{target}_t"
    future = f"{target}_future"
    train = train[train[current].notna() & train[future].notna()].copy()
    test = test[test[current].notna() & test[future].notna()].copy()
    if train.empty or test.empty:
        return {"status": "insufficient_pairs", "rows": int(len(test)), "subjects": 0}

    train = add_event_history_features(train, target, _EVENT_ARRAYS)
    test = add_event_history_features(test, target, _EVENT_ARRAYS)
    coupled_raw, coupled_log, history_raw, history_log = _fit_candidates(train, test, target)
    truth = test[future].to_numpy(dtype=np.float64)
    baseline = test[current].to_numpy(dtype=np.float64)
    return {
        "status": "evaluated",
        "rows": int(len(test)),
        "subjects": int(test["subject_id"].nunique()),
        "coupled_raw": _delta_summary(test, truth, coupled_raw, baseline, seed + 11),
        "coupled_log": _delta_summary(test, truth, coupled_log, baseline, seed + 23),
        "coupled_raw_vs_history": _delta_summary(test, truth, coupled_raw, history_raw, seed + 31),
        "coupled_log_vs_history": _delta_summary(test, truth, coupled_log, history_log, seed + 37),
    }


def evaluate_bin(
    frame: pd.DataFrame,
    target: str,
    lower: float,
    upper: float,
    arrays: dict[tuple[object, str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    subset = frame[frame["forecast_horizon_hr"].between(lower, upper, inclusive="right")].copy()
    reports = []
    global _EVENT_ARRAYS
    _EVENT_ARRAYS = arrays
    if not subset.empty:
        reports = [_run_split(subset, target, seed) for seed in SEEDS]
    care = None
    if not subset.empty and subset["first_careunit"].nunique(dropna=True) >= 6:
        care = _run_split(subset, target, 9901, "first_careunit")

    statuses = {}
    for model in ("coupled_raw", "coupled_log"):
        baseline_pass = all(item.get(model, {}).get("significant", False) for item in reports)
        increment_key = f"{model}_vs_history"
        history_pass = all(item.get(increment_key, {}).get("significant", False) for item in reports)
        care_pass = bool(
            care
            and care.get(model, {}).get("significant", False)
            and care.get(increment_key, {}).get("significant", False)
        )
        statuses[model] = "validated" if baseline_pass and history_pass and care_pass else "candidate_or_fallback"

    return {
        "pairs": int(len(subset)),
        "subjects": int(subset["subject_id"].nunique()) if not subset.empty else 0,
        "patient_splits": reports,
        "care_unit_holdout": care,
        "promotion_status": statuses,
    }


_EVENT_ARRAYS: dict[tuple[object, str], tuple[np.ndarray, np.ndarray]] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument(
        "--events-cache",
        type=Path,
        default=Path("/private/tmp/mimiciv_sparse_lab_events.parquet"),
    )
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = pd.read_parquet(args.events_cache) if args.events_cache.exists() else load_events(args.mimic_dir)
    for column in ("target", "first_careunit"):
        if column in events:
            events[column] = events[column].astype(object)
    arrays = _event_arrays(events)
    results = []
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    for target in targets:
        print(f"Building coupled irregular pairs: {target}", flush=True)
        frame = build_irregular_pairs(events, target, arrays)
        frame = add_coupled_history_features(
            frame,
            target,
            arrays,
            COUPLED_SOURCE_MAP.get(target, ()),
        )
        for name, lower, upper in TIME_BINS:
            report = evaluate_bin(frame, target, lower, upper, arrays)
            report.update({"target": target, "time_bin": name, "lower_hr": lower, "upper_hr": upper})
            results.append(report)
            print(target, name, report["pairs"], report["promotion_status"], flush=True)

    output = {
        "artifact": "MIMIC-IV coupled irregular-time sparse laboratory audit",
        "history_definition": "all source history features use observations strictly before anchor",
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
            "hidden_state_accuracy_claim_allowed": False,
            "row_level_outputs_written": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "validated_cells": output["validated_cells"]}, indent=2))


if __name__ == "__main__":
    main()
