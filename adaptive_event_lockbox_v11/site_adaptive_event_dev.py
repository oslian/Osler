"""MIMIC-only development of online calibration for the binary event cells.

Same deployment scenarios and shared-clock replay as ``site_adaptive_interval_dev``
(S1 time, S2 cardiac care units; warm-up stays are not scored).  The frozen
static hypoxemia model (``_fit_hypoxemia``, as ``evaluate_event`` fits it) is
fitted on the source patients and replayed on the target stream.

Candidates: static; online Platt scaling with the published settings
(``osler_jepa.online_platt``, Gupta & Ramdas 2023).  The MIMIC-prevalence
comparator receives the same recalibration (it then tracks the site's event
rate), so the Brier gate stays a like-for-like test of the model; persistence
(current SpO2 <= 92) is unchanged.

Predeclared rule (written before any candidate was scored): a stream passes
when the Brier score beats persistence and the comparator on the patient axis
(``paired_comparison``) and calibration passes ``osler_jepa.event_calibration``
(CITL in [-0.2, 0.2], slope in [0.8, 1.25]).  Online Platt scaling is adopted
only if it passes more streams than static and its median Brier ratio to
static is <= 1.  No external data is read.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import (
    _feature_frame, _fit_hypoxemia, cap_by_subject, feature_subset, normalize_cohort, paired_comparison, stable_seed,
)
from osler_jepa.event_calibration import calibration, calibration_pass, valid_probability
from osler_jepa.online_platt import OnlinePlattConfig, online_platt_scaling
from site_adaptive_interval_dev import SCENARIOS, STREAM_ROWS, Replay, replay
from target_specific_domain_conformal_audit import _safe_float

THRESHOLD = 92.0
CANDIDATES = {"static": None, "online_platt": OnlinePlattConfig()}


def static_model(frame: pd.DataFrame, train: np.ndarray, test: np.ndarray, cell: str) -> dict[str, pd.Series]:
    """Frozen static model fitted on ``train``: probability, MIMIC prevalence and persistence of each test row."""

    features = feature_subset(_feature_frame(frame), "measurement_pure_ridge", "o2sat")
    rows, candidate, persistence, prevalence, _ = _fit_hypoxemia(
        frame, features, train, test, seed=stable_seed(cell, "mimic_to_eicu"), threshold=THRESHOLD)
    return {name: pd.Series(values, index=rows) for name, values in
            (("probability", candidate), ("persistence", persistence), ("prevalence", prevalence))}


def recalibrate(deployment: Replay, values: np.ndarray, observed: np.ndarray, config: OnlinePlattConfig | None) -> np.ndarray:
    if config is None:
        return values
    return online_platt_scaling(anchor_hours=deployment.clock, resolve_hours=deployment.resolve, probability=values,
                                outcome=observed, config=config)


def score_stream(frame: pd.DataFrame, rows: np.ndarray, observed: np.ndarray, probability: np.ndarray,
                 persistence: np.ndarray, comparator: np.ndarray, *, axes: tuple[str, ...], seed: int) -> dict:
    """Brier gate on ``axes`` against persistence and the comparator, plus the calibration gate.

    Any invalid probability (non-finite or outside [0, 1]) in the candidate or the comparator fails the stream."""

    invalid = int(np.sum(~valid_probability(probability)) + np.sum(~valid_probability(comparator)))
    if invalid:
        return {"invalid_predictions": invalid, "point_pass": False, "calibration_pass": False, "full_pass": False}
    loss = (probability - observed) ** 2
    comparisons = {name: paired_comparison(frame, rows, loss, (other - observed) ** 2, seed=seed + i)
                   for i, (name, other) in enumerate((("persistence", persistence), ("comparator", comparator)))}
    point = all(report[axis]["pass"] for report in comparisons.values() for axis in axes)
    estimate = calibration(probability, observed)
    calibrated = calibration_pass(estimate)
    return {"invalid_predictions": 0, "brier": float(loss.mean()), "event_rate": float(observed.mean()), "mean_probability": float(probability.mean()),
            **estimate, "point_pass": bool(point), "calibration_pass": calibrated, "full_pass": bool(point and calibrated),
            "comparisons": {name: {axis: report[axis] for axis in axes} for name, report in comparisons.items()}}


def run_cell(full: pd.DataFrame, years: pd.Series, *, cell: str) -> dict:
    full = full.assign(anchor_year_group=full["subject_id"].map(years).fillna("unknown"))
    results = {}
    for name, (column, source_values, target_values) in SCENARIOS.items():
        in_target = full[column].isin(target_values).to_numpy()
        in_source = full[column].isin(source_values).to_numpy() if source_values else ~in_target
        source_frame = cap_by_subject(full[in_source], 15000, stable_seed(cell, name, "source"))
        target_frame = cap_by_subject(full[in_target], STREAM_ROWS, stable_seed(cell, name, "target"))
        source_frame = source_frame.assign(subject_id="SRC:" + source_frame["subject_id"].astype(str))
        target_frame = target_frame.assign(subject_id="TGT:" + target_frame["subject_id"].astype(str))
        frame = pd.concat([source_frame, target_frame], ignore_index=True, sort=False)
        frame.sort_values(["subject_id", "anchor_time"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        subject = frame["subject_id"].astype(str)
        model = static_model(frame, np.flatnonzero(subject.str.startswith("SRC:")), np.flatnonzero(subject.str.startswith("TGT:")), cell)
        deployment = replay(frame, model["probability"].index.to_numpy(), "o2sat", stable_seed(cell, name, "clock"))
        rows, scored = deployment.rows, ~deployment.warm
        observed = (_safe_float(frame, "future_o2sat")[rows] <= THRESHOLD).astype(float)
        scenario, reference = {}, None
        for label, config in CANDIDATES.items():
            probability = recalibrate(deployment, model["probability"][rows].to_numpy(), observed, config)
            comparator = recalibrate(deployment, model["prevalence"][rows].to_numpy(), observed, config)
            scenario[label] = score_stream(frame, rows[scored], observed[scored], probability[scored],
                                           model["persistence"][rows].to_numpy()[scored], comparator[scored],
                                           axes=("patient",), seed=stable_seed(cell, name, "brier"))
            reference = reference or scenario[label]["brier"]
            scenario[label]["brier_ratio_vs_static"] = scenario[label]["brier"] / reference
        results[name] = {"stream_rows": int(len(rows)), "scored_rows": int(scored.sum()), "candidates": scenario}
        print(json.dumps({"cell": cell, "scenario": name, **{label: {k: round(v, 3) if isinstance(v, float) else v
              for k, v in result.items() if k in ("citl", "slope", "brier", "point_pass", "full_pass")}
              for label, result in scenario.items()}}), flush=True)
    return results


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = [row for row in development["cells"] if row["task_type"] == "binary_event_probability"]
    patients = pd.read_csv(args.patients, usecols=["subject_id", "anchor_year_group"])
    years = pd.Series(patients["anchor_year_group"].to_numpy(), index="MIMIC:" + patients["subject_id"].astype(str))
    cells = []
    for contract in sorted(contracts, key=lambda row: int(row["horizon_hours"])):
        horizon = int(contract["horizon_hours"])
        full, _ = normalize_cohort(pd.read_parquet(args.mimic_template.format(horizon=horizon)), database="MIMIC-IV",
                                   target=contract["target"], horizon=horizon, max_rows=0)
        full = full.assign(first_careunit=full["careunit"])
        cells.append({"cell": contract["cell"], **run_cell(full, years, cell=contract["cell"])})
    summary = {}
    for label in CANDIDATES:
        streams = [cell[s]["candidates"][label] for cell in cells for s in SCENARIOS]
        summary[label] = {"full_pass_streams": int(sum(item["full_pass"] for item in streams)),
                          "point_pass_streams": int(sum(item["point_pass"] for item in streams)),
                          "calibration_pass_streams": int(sum(item["calibration_pass"] for item in streams)),
                          "streams": len(streams),
                          "median_brier_ratio_vs_static": float(np.median([item["brier_ratio_vs_static"] for item in streams]))}
    ops = summary["online_platt"]
    adopted = ops["full_pass_streams"] > summary["static"]["full_pass_streams"] and ops["median_brier_ratio_vs_static"] <= 1.0
    output = {
        "schema": "site_adaptive_event_dev.v1",
        "database": "MIMIC-IV only",
        "mimic_template": args.mimic_template,
        "selection_rule": "online Platt scaling adopted only if more full-pass streams than static and median Brier ratio <= 1",
        "summary": summary,
        "selected": "online_platt" if adopted else "static",
        "cells": cells,
        "external_data_read": False,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", type=Path, default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"))
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--patients", type=Path, default=Path.home() / "Downloads/mimic-iv-3.1/hosp/patients.csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"selected": result["selected"], "summary": result["summary"]}, indent=1))
