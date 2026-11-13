"""Patient-held-out audit for future abnormal biomarker risk.

The task is deliberately narrow: given information available at an anchor and
an explicit requested elapsed horizon, estimate whether the next factual value
at that horizon is outside its recorded MIMIC-IV reference range.  The script
does not predict when clinicians will order the next measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eicu_sepsis_target_router import split_subjects
from mimiciv_biomarker_prediction_support_audit import load_events


SEEDS = (17, 29, 43, 59, 71, 89, 101)
DEFAULT_TARGETS = (
    "total_cholesterol",
    "hdl_cholesterol",
    "hba1c",
    "iron",
    "tibc",
    "transferrin",
)
HISTORY_COLUMNS = (
    "previous_value",
    "previous_age_hr",
    "previous_delta",
    "observation_count",
)
FEATURE_COLUMNS = (
    "current_value",
    "current_abnormal",
    "requested_horizon_hr",
    *HISTORY_COLUMNS,
)


def _abnormal(value: pd.Series, low: pd.Series, high: pd.Series) -> pd.Series:
    return (
        (low.notna() & value.lt(low))
        | (high.notna() & value.gt(high))
    ).astype(np.int8)


def build_pairs(events: pd.DataFrame, target: str) -> pd.DataFrame:
    source = events[events["target"].eq(target)].copy()
    if source.empty:
        return source
    source = source.sort_values(["stay_id", "charttime"], kind="stable")
    group = source.groupby("stay_id", sort=False)
    source["previous_value"] = group["value"].shift(1)
    source["previous_time"] = group["charttime"].shift(1)
    source["previous_age_hr"] = (
        source["charttime"] - source["previous_time"]
    ).dt.total_seconds() / 3600.0
    source["previous_delta"] = source["value"] - source["previous_value"]
    source["observation_count"] = group.cumcount().astype(np.float64) + 1.0
    source["future_time"] = group["charttime"].shift(-1)
    source["future_value"] = group["value"].shift(-1)
    source["future_low"] = group["reference_low"].shift(-1)
    source["future_high"] = group["reference_high"].shift(-1)
    source["requested_horizon_hr"] = (
        source["future_time"] - source["charttime"]
    ).dt.total_seconds() / 3600.0
    pairs = source[
        source["requested_horizon_hr"].between(0.5, 72.0, inclusive="both")
        & source["future_value"].notna()
        & (source["reference_low"].notna() | source["reference_high"].notna())
        & (source["future_low"].notna() | source["future_high"].notna())
    ].copy()
    if pairs.empty:
        return pairs
    pairs["current_value"] = pairs["value"]
    pairs["current_abnormal"] = _abnormal(
        pairs["value"], pairs["reference_low"], pairs["reference_high"]
    )
    pairs["future_abnormal"] = _abnormal(
        pairs["future_value"], pairs["future_low"], pairs["future_high"]
    )
    return pairs.reset_index(drop=True)


def _pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=1000, solver="lbfgs")),
    ])


def _patient_losses(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    comparator: np.ndarray,
) -> np.ndarray:
    truth = frame["future_abnormal"].to_numpy(dtype=np.float64)
    work = pd.DataFrame({
        "subject_id": frame["subject_id"].to_numpy(),
        "delta": (candidate - truth) ** 2 - (comparator - truth) ** 2,
    })
    return work.groupby("subject_id", sort=False)["delta"].mean().to_numpy(dtype=np.float64)


def _bootstrap(values: np.ndarray, seed: int, samples: int = 2000) -> dict[str, object]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 2:
        return {"mean_delta": None, "ci95": None, "pass": False}
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(clean, size=len(clean), replace=True).mean()
        for _ in range(samples)
    ])
    interval = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "mean_delta": float(clean.mean()),
        "ci95": interval,
        "pass": bool(interval[1] < 0.0),
    }


def _evaluate(
    frame: pd.DataFrame,
    train_groups: set[object],
    test_groups: set[object],
    group_column: str,
    seed: int,
) -> dict[str, object]:
    train = frame[frame[group_column].isin(train_groups)].copy()
    test = frame[frame[group_column].isin(test_groups)].copy()
    if train.empty or test.empty or train["future_abnormal"].nunique() < 2:
        return {"status": "insufficient_split_support", "pass": False}
    candidate = _pipeline()
    candidate.fit(train[list(FEATURE_COLUMNS)], train["future_abnormal"])
    candidate_probability = candidate.predict_proba(test[list(FEATURE_COLUMNS)])[:, 1]

    rng = np.random.default_rng(seed)
    placebo_train = train[list(FEATURE_COLUMNS)].copy()
    placebo_test = test[list(FEATURE_COLUMNS)].copy()
    for column in HISTORY_COLUMNS:
        placebo_train[column] = rng.permutation(placebo_train[column].to_numpy())
        placebo_test[column] = rng.permutation(placebo_test[column].to_numpy())
    placebo = _pipeline()
    placebo.fit(placebo_train, train["future_abnormal"])
    placebo_probability = placebo.predict_proba(placebo_test)[:, 1]

    persistence_probability = test["current_abnormal"].to_numpy(dtype=np.float64)
    prevalence_probability = np.full(len(test), train["future_abnormal"].mean(), dtype=np.float64)
    truth = test["future_abnormal"].to_numpy(dtype=np.float64)
    versus_persistence = _bootstrap(
        _patient_losses(test, candidate_probability, persistence_probability), seed + 11
    )
    versus_prevalence = _bootstrap(
        _patient_losses(test, candidate_probability, prevalence_probability), seed + 23
    )
    versus_placebo = _bootstrap(
        _patient_losses(test, candidate_probability, placebo_probability), seed + 37
    )
    return {
        "status": "evaluated",
        "rows": int(len(test)),
        "subjects": int(test["subject_id"].nunique()),
        "event_prevalence": float(truth.mean()),
        "candidate_brier": float(np.mean((candidate_probability - truth) ** 2)),
        "persistence_brier": float(np.mean((persistence_probability - truth) ** 2)),
        "prevalence_brier": float(np.mean((prevalence_probability - truth) ** 2)),
        "placebo_brier": float(np.mean((placebo_probability - truth) ** 2)),
        "versus_persistence": versus_persistence,
        "versus_prevalence": versus_prevalence,
        "versus_placebo": versus_placebo,
        "pass": bool(
            versus_persistence["pass"]
            and versus_prevalence["pass"]
            and versus_placebo["pass"]
        ),
    }


def evaluate_target(frame: pd.DataFrame, target: str) -> dict[str, object]:
    pairs = build_pairs(frame, target)
    positives = int(pairs["future_abnormal"].sum()) if not pairs.empty else 0
    negatives = int(len(pairs) - positives)
    if pairs.empty or positives < 30 or negatives < 30:
        return {
            "target": target,
            "status": "insufficient_class_support",
            "pairs": int(len(pairs)),
            "positive": positives,
            "negative": negatives,
            "promotion_status": "fallback",
        }
    patient_splits = []
    for seed in SEEDS:
        train, test = split_subjects(pairs, seed, 0.67, "subject_id")
        patient_splits.append(_evaluate(pairs, train, test, "subject_id", seed))
    care_unit = None
    if pairs["first_careunit"].nunique(dropna=True) >= 2:
        train, test = split_subjects(pairs, 9901, 0.67, "first_careunit")
        care_unit = _evaluate(pairs, train, test, "first_careunit", 9901)
    validated = bool(
        all(item.get("pass", False) for item in patient_splits)
        and care_unit is not None
        and care_unit.get("pass", False)
    )
    return {
        "target": target,
        "status": "evaluated",
        "pairs": int(len(pairs)),
        "subjects": int(pairs["subject_id"].nunique()),
        "positive": positives,
        "negative": negatives,
        "patient_splits": patient_splits,
        "care_unit_holdout": care_unit,
        "promotion_status": "validated" if validated else "fallback",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = load_events(args.mimic_dir)
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    results = []
    for target in targets:
        report = evaluate_target(events, target)
        results.append(report)
        print(target, report["promotion_status"], flush=True)
    output = {
        "schema": "mimiciv_biomarker_abnormal_risk.v1",
        "task": "future abnormal risk at an explicit requested elapsed horizon",
        "results": results,
        "validated_targets": sorted(
            item["target"] for item in results if item["promotion_status"] == "validated"
        ),
        "policy": {
            "next_measurement_time_modeled": False,
            "future_information_used_as_feature": False,
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
            "row_level_data_written": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "validated_targets": output["validated_targets"],
        "next_measurement_time_modeled": False,
    }, indent=2))


if __name__ == "__main__":
    main()
