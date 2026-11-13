"""Downstream audit for hidden physiology belief proxies.

The audit asks a narrow question: do the two online hidden-state proxies add
held-out predictive information for measurable downstream variables?  It does
not attempt to score the unobservable state itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_target_router import _bootstrap_ci, _feature_columns, _fit_ridge, _round, split_subjects
from hidden_physiology_beliefs import (
    ALL_HIDDEN_PHYSIOLOGY_COLUMNS,
    hidden_physiology_belief_features,
    placebo_hidden_physiology_features,
)


SEEDS = (7, 11, 19, 23, 37, 53, 71)


def _subject_column(frame: pd.DataFrame) -> str:
    return "subject_id" if "subject_id" in frame else "stay_id"


def _target_rows(frame: pd.DataFrame, target: str, future_suffix: str) -> pd.DataFrame:
    current = f"{target}_t"
    future = f"{target}_{future_suffix}"
    if current not in frame or future not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame[current].notna() & frame[future].notna()].copy()


def _delta_summary(rows: pd.DataFrame, truth: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, seed: int) -> dict[str, object]:
    mask = np.isfinite(truth) & np.isfinite(candidate) & np.isfinite(baseline)
    if not mask.any():
        return {"subjects": 0, "point_delta": None, "bootstrap_95_ci": [None, None], "significant": False}
    subject = _subject_column(rows)
    work = rows.loc[mask, [subject]].copy()
    work["delta"] = np.abs(candidate[mask] - truth[mask]) - np.abs(baseline[mask] - truth[mask])
    values = work.groupby(subject)["delta"].mean().to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(values, seed=seed, samples=300)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_baseline": bool(point < 0.0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def _run_split(frame: pd.DataFrame, target: str, suffix: str, seed: int, group_column: str | None = None) -> dict[str, object]:
    group_column = group_column or _subject_column(frame)
    discovery, heldout = split_subjects(frame, seed=seed, discovery_fraction=0.67, group_column=group_column)
    train = _target_rows(frame[frame[group_column].isin(discovery)], target, suffix)
    test = _target_rows(frame[frame[group_column].isin(heldout)], target, suffix)
    if train.empty or test.empty:
        return {"status": "insufficient_pairs", "rows": int(len(test)), "subjects": 0}
    base = [column for column in _feature_columns(train, target) if column not in ALL_HIDDEN_PHYSIOLOGY_COLUMNS]
    hidden = [column for column in ALL_HIDDEN_PHYSIOLOGY_COLUMNS if column in train and column in test]
    candidate = _fit_ridge(train, test, target, sorted(set(base + hidden)), 10.0, suffix)
    baseline = _fit_ridge(train, test, target, base, 10.0, suffix)
    placebo_train = pd.concat([train, placebo_hidden_physiology_features(train, seed=seed + 100)], axis=1)
    placebo_test = pd.concat([test, placebo_hidden_physiology_features(test, seed=seed + 200)], axis=1)
    placebo = _fit_ridge(
        placebo_train,
        placebo_test,
        target,
        sorted(set(base + [f"placebo_{column}" for column in hidden])),
        10.0,
        suffix,
    )
    truth = test[f"{target}_{suffix}"].to_numpy(dtype=np.float64)
    return {
        "status": "evaluated",
        "group_column": group_column,
        "rows": int(len(test)),
        "subjects": int(test[_subject_column(test)].nunique()),
        "feature_counts": {"baseline": len(base), "hidden": len(hidden)},
        "candidate_vs_baseline": _delta_summary(test, truth, candidate, baseline, seed + 11),
        "candidate_vs_placebo": _delta_summary(test, truth, candidate, placebo, seed + 23),
    }


def audit(cohort: Path, future_suffix: str, targets: tuple[str, ...]) -> dict[str, object]:
    frame = pd.read_parquet(cohort).reset_index(drop=True)
    hidden = hidden_physiology_belief_features(frame).reset_index(drop=True)
    combined = pd.concat([frame, hidden], axis=1)
    results = {}
    for target in targets:
        print(f"Auditing hidden beliefs -> {target}", flush=True)
        random_reports = [_run_split(combined, target, future_suffix, seed) for seed in SEEDS]
        care = None
        if "hospitalid" in combined and combined["hospitalid"].nunique(dropna=True) >= 3:
            care = _run_split(combined, target, future_suffix, 9901, "hospitalid")
        pass_count = sum(
            bool(item.get("candidate_vs_baseline", {}).get("significant"))
            and bool(item.get("candidate_vs_placebo", {}).get("significant"))
            for item in random_reports
        )
        results[target] = {
            "patient_split_pass_both_count": int(pass_count),
            "patient_split_reports": random_reports,
            "hospital_holdout": care,
            "validated": bool(
                pass_count == len(SEEDS)
                and care
                and care.get("candidate_vs_baseline", {}).get("significant")
                and care.get("candidate_vs_placebo", {}).get("significant")
            ),
        }
    return {
        "artifact": "hidden physiology downstream observable audit",
        "cohort": str(cohort),
        "future_suffix": future_suffix,
        "rows": int(len(frame)),
        "subjects": int(frame[_subject_column(frame)].nunique()),
        "targets": results,
        "belief_columns": list(ALL_HIDDEN_PHYSIOLOGY_COLUMNS),
        "boundary": {
            "hidden_state_accuracy_claim_allowed": False,
            "direct_hormone_concentration_claim_allowed": False,
            "insulin_sensitivity_causal_claim_allowed": False,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--future-suffix", default="tp6")
    parser.add_argument("--targets", default="glucose,anion_gap,bicarbonate,ph,sodium,potassium,map")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.cohort,
        args.future_suffix,
        tuple(item.strip() for item in args.targets.split(",") if item.strip()),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "validated": [name for name, item in report["targets"].items() if item["validated"]]}, indent=2))


if __name__ == "__main__":
    main()

