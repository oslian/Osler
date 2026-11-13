"""External observation-only audit on the full MIMIC-IV ICU transition table.

This is deliberately separate from the DKA audit.  MIMIC-IV is a single
hospital, so it cannot provide a true hospital-held-out split.  We therefore
report patient-held-out and ICU-careunit-held-out results and mark the missing
hospital axis explicitly instead of inventing one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_aware_pure_jepa import (
    _split_subjects,
    conformal_calibration,
    evaluate,
    fit_one,
)
from nonlinear_latent_rollout import _time_hours, state_variables


SEEDS = (7, 19, 31, 43, 59, 71, 83)


def _prepare(path: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = pd.read_parquet(path)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=2026).sort_index()
    variables = state_variables(frame)
    for variable in variables:
        future = f"{variable}_tp6"
        if future in frame.columns:
            frame.rename(columns={future: f"future_{variable}"}, inplace=True)
        elif f"{variable}_t" in frame.columns:
            frame[f"future_{variable}"] = np.nan
    frame["_source_horizon"] = 6
    if "hours_since_onset" not in frame:
        frame["hours_since_onset"] = 0.0
    # MIMIC-IV is BIDMC only.  Care unit is a useful secondary split, but is
    # not a hospital identifier and is never reported as one.
    frame["hospitalid"] = -1
    if "first_careunit" not in frame:
        frame["first_careunit"] = "unknown"
    frame["first_careunit"] = frame["first_careunit"].fillna("unknown").astype(str)
    frame["_time_hours"] = _time_hours(frame)
    frame.sort_values(["stay_id", "t"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame, variables


def _careunit_split(frame: pd.DataFrame, seed: int, fraction: float = 0.25):
    units = frame["first_careunit"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    held = set(units[:max(1, int(round(len(units) * fraction)))].tolist())
    test = frame["first_careunit"].isin(held).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def _run_fit(frame, variables, train_rows, test_rows, seed: int, epochs: int):
    fit_local, cal_local = _split_subjects(frame.iloc[train_rows].reset_index(drop=True), seed + 1001, 0.20)
    calibration_rows = train_rows[cal_local]
    fit_rows = train_rows[fit_local]
    model, arrays, losses = fit_one(frame, variables, fit_rows, seed=seed, epochs=epochs, batch_size=2048)
    metrics = evaluate(model, arrays, frame, variables, test_rows)
    conformal = conformal_calibration(model, arrays, variables, calibration_rows, test_rows)
    return {"loss_history": losses, "metrics": metrics, "conformal": conformal}


def _summary(results):
    summary = {}
    keys = sorted(set().union(*(r["metrics"].keys() for r in results)))
    for target in keys:
        cells = [r["metrics"].get(target, {}) for r in results]
        cells = [cell for cell in cells if "delta_vs_persistence" in cell]
        if not cells:
            continue
        summary[target] = {
            "n_splits": len(cells),
            "median_delta_vs_persistence": float(np.median([c["delta_vs_persistence"] for c in cells])),
            "pass_splits": int(sum(c["delta_vs_persistence"] < 0 for c in cells)),
            "pass_all_patient_splits": bool(len(cells) == len(results) and all(c["delta_vs_persistence"] < 0 for c in cells)),
            "median_n": int(np.median([c["n"] for c in cells])),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=-1)
    args = parser.parse_args()
    max_rows = None if args.max_rows < 0 else args.max_rows
    frame, variables = _prepare(args.transition, max_rows)
    patient_results = []
    for seed in SEEDS:
        train_rows, test_rows = _split_subjects(frame, seed)
        result = _run_fit(frame, variables, train_rows, test_rows, seed, args.epochs)
        patient_results.append(result)
        print(json.dumps({"scope": "patient", "seed": seed, "targets": len(result["metrics"]), "loss": result["loss_history"][-1]}), flush=True)
    train_rows, test_rows = _careunit_split(frame, 2026)
    careunit_result = _run_fit(frame, variables, train_rows, test_rows, 2026, args.epochs)
    report = {
        "schema": "mimiciv_full_observation_audit.v1",
        "transition": str(args.transition.resolve()),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "careunits": int(frame["first_careunit"].nunique()),
        "variables": list(variables),
        "patient_heldout": {"seeds": list(SEEDS), "summary": _summary(patient_results)},
        "careunit_heldout": careunit_result,
        "true_hospital_heldout": {"status": "not_available", "reason": "MIMIC-IV is single-center BIDMC; first_careunit is not a hospital identifier"},
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "row_level_outputs_committed": False,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rows": report["rows"], "targets": len(variables), "careunits": report["careunits"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
