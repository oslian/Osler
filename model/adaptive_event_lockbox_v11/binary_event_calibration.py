"""Calibration of the static binary event cells on the development databases.

Logistic recalibration (Cox 1958; Van Calster et al. 2016, 2019):

- calibration slope: coefficient of logit(p) in a logistic model of the outcome;
- calibration-in-the-large (CITL): intercept with logit(p) as an offset;
- observed / expected ratio and AUROC are reported, not gated.

Confidence intervals are percentile cluster bootstraps over the point-gate axes
of ``GATE_AXES`` (patients everywhere; hospitals too on eICU).  Predictions are
rebuilt exactly as the frozen ``evaluate_event`` builds them and must reproduce
the Brier score of ``binary_event_metrics_report.py`` before anything is scored.

Candidate rule proposed before this check (user decision 2026-10-28, option B):
a sample passes when the slope is within [0.8, 1.2] and every axis' CITL 95% CI
contains 0; a cell passes in >= 80% of samples.  Development data: in-sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from external_value_conditional_interval_audit import EXTERNAL_PREFIX, GATE_AXES, external_cohort
from mimic_to_eicu_49_cell_external_audit import (
    _feature_frame, _fit_hypoxemia, _safe_float, feature_subset, normalize_cohort, stable_seed,
)
from multi_sample_external_audit import ROBUST_SHARE, mimic_sample
from osler_jepa.event_calibration import BOOTSTRAP_SAMPLES, calibration, cluster_bootstrap

THRESHOLD = 92.0
SLOPE_BOUNDS = (0.8, 1.2)
CLUSTER_COLUMN = {"patient": "subject_id", "hospital": "hospitalid"}


def event_predictions(mimic: pd.DataFrame, external: pd.DataFrame, *, cell: str, database: str) -> pd.DataFrame:
    """External rows, predicted probability and outcome, as ``evaluate_event`` computes them."""

    frame = pd.concat([mimic, external], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    subject = frame["subject_id"].astype(str)
    train_rows = np.flatnonzero(subject.str.startswith("MIMIC:"))
    test_rows = np.flatnonzero(subject.str.startswith(EXTERNAL_PREFIX[database]))
    features = feature_subset(_feature_frame(frame), "measurement_pure_ridge", "o2sat")
    rows, candidate, *_ = _fit_hypoxemia(frame, features, train_rows, test_rows,
                                         seed=stable_seed(cell, "mimic_to_eicu"), threshold=THRESHOLD)
    return pd.DataFrame({
        "subject_id": frame["subject_id"].to_numpy()[rows],
        "hospitalid": frame["hospitalid"].to_numpy()[rows],
        "probability": candidate,
        "outcome": (_safe_float(frame, "future_o2sat")[rows] <= THRESHOLD).astype(float),
    })


def score_sample(predictions: pd.DataFrame, database: str, *, seed: int) -> dict:
    estimate = calibration(predictions["probability"].to_numpy(), predictions["outcome"].to_numpy())
    intervals = {axis: cluster_bootstrap(predictions, CLUSTER_COLUMN[axis], seed=seed + i)
                 for i, axis in enumerate(GATE_AXES[database]["point"])}
    slope_ok = SLOPE_BOUNDS[0] <= estimate["slope"] <= SLOPE_BOUNDS[1]
    citl_ok = all(ci["citl"] is not None and ci["citl"][0] <= 0.0 <= ci["citl"][1] for ci in intervals.values())
    return {**estimate, "auroc": float(roc_auc_score(predictions["outcome"], predictions["probability"])),
            "intervals": intervals, "slope_pass": slope_ok, "citl_pass": citl_ok, "pass": bool(slope_ok and citl_ok)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="binary_event_metrics_report.py output")
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_harmonized_20261022"))
    parser.add_argument("--external-template", default=None)
    parser.add_argument("--max-rows-per-database", type=int, default=15000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.database = report["external_database"]
    cells = []
    for entry in report["cells"]:
        if not entry["evaluable"]:
            cells.append(entry)
            continue
        target, horizon = entry["cell"].rsplit("@", 1)
        horizon = int(horizon.rstrip("h"))
        external, _ = external_cohort(args, target, horizon)
        full, _ = normalize_cohort(pd.read_parquet(args.mimic_template.format(horizon=horizon)),
                                   database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
        samples = []
        for frozen in entry["per_sample"]:
            mimic = mimic_sample(full, target=target, horizon=horizon, sample=frozen["sample"], rows=args.max_rows_per_database)
            predictions = event_predictions(mimic, external, cell=entry["cell"], database=args.database)
            brier = float(np.mean((predictions["probability"] - predictions["outcome"]) ** 2))
            if abs(brier - frozen["candidate_brier"]) > 1e-12:
                raise RuntimeError(f"{entry['cell']} sample {frozen['sample']}: rebuilt predictions differ ({brier} vs {frozen['candidate_brier']})")
            samples.append({"sample": frozen["sample"], "event_rate": float(predictions["outcome"].mean()),
                            "mean_probability": float(predictions["probability"].mean()),
                            **score_sample(predictions, args.database, seed=stable_seed(entry["cell"], "calibration", frozen["sample"]))})
        share = float(np.mean([s["pass"] for s in samples]))
        cells.append({"cell": entry["cell"], "evaluable": True, "pass_share": share, "robust": share >= ROBUST_SHARE,
                      "per_sample": samples})
        print(json.dumps({"cell": entry["cell"], "slopes": [round(s["slope"], 3) for s in samples],
                          "citl": [round(s["citl"], 3) for s in samples], "pass_share": share}), flush=True)
    args.output.write_text(json.dumps({
        "schema": "binary_event_calibration.v1",
        "metrics": str(args.metrics),
        "external_database": args.database,
        "rule": {"slope_bounds": SLOPE_BOUNDS, "citl_ci_contains_zero_on_axes": GATE_AXES[args.database]["point"],
                 "bootstrap_samples": BOOTSTRAP_SAMPLES, "robust_share": ROBUST_SHARE},
        "independent_validation": False,
        "cells": cells,
        "causal_claim_allowed": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
