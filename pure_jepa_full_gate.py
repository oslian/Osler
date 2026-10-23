"""Full factual gate for pure-JEPA body-system checkpoints.

Runs patient-held-out seeds, hospital-held-out evaluation when available, and
split-conformal coverage. Unsupported sparse targets are explicitly rejected
instead of being forced into a forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nonlinear_latent_rollout import load_multihorizon_cohort, set_seed
from pure_jepa_body_system import (
    PureJEPABodyModel,
    _arrays,
    _metrics,
    _predict,
)


SEEDS = (7, 11, 19, 23, 37, 53, 71)
MODULES = (
    "aki", "acute_neuro", "cardiac_injury", "cardiovascular_instability",
    "coagulopathy_heme", "electrolyte_acid_base", "endocrine_stress",
    "gi_pancreatic_nutrition", "hepatic_failure", "immune_inflammatory",
    "integumentary_skin_wound", "musculoskeletal_rhabdo", "respiratory",
    "sepsis", "toxic_metabolic",
)


def _split(frame, column, seed, fraction):
    groups = frame[column].dropna().drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    count = max(1, int(round(len(groups) * fraction)))
    selected = set(groups[:count].tolist())
    test = frame[column].isin(selected).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def _fit(frame, variables, rows, seed, epochs, batch_size):
    set_seed(seed)
    arrays = _arrays(frame, variables, rows)
    _, input_matrix, current, future, future_mask, history, horizon = arrays
    model = PureJEPABodyModel(len(variables), input_matrix.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(rows)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            observations = torch.from_numpy(input_matrix[history[batch]]).float()
            state = torch.from_numpy(current[batch]).float()
            dt = torch.from_numpy(horizon[batch]).float()
            target = torch.from_numpy(future[batch]).float()
            mask = torch.from_numpy(future_mask[batch]).float()
            prediction = model(observations, state, dt)
            loss_values = torch.nn.functional.smooth_l1_loss(
                prediction, target, reduction="none", beta=0.5
            )
            loss = (loss_values * mask).sum() / mask.sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model, arrays


def _raw_predictions(model, arrays, rows):
    scaler = arrays[0]
    normalized = _predict(model, arrays, rows)
    return normalized * scaler.scales[None, :] + scaler.medians[None, :]


def _truth_current(arrays, rows):
    scaler, _, current, future, future_mask, _, _ = arrays
    current_raw = current[rows] * scaler.scales[None, :] + scaler.medians[None, :]
    truth_raw = future[rows] * scaler.scales[None, :] + scaler.medians[None, :]
    return truth_raw, current_raw, future_mask[rows].astype(bool)


def _bootstrap_ci(groups, delta, seed, samples=300):
    unique = np.unique(groups)
    if len(unique) < 5:
        return None
    by_group = np.asarray([delta[groups == group].mean() for group in unique])
    rng = np.random.default_rng(seed)
    draws = [rng.choice(by_group, len(by_group), replace=True).mean() for _ in range(samples)]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _coverage(model, arrays, calibration_rows, test_rows, frame, variables):
    calibration = _raw_predictions(model, arrays, calibration_rows)
    test = _raw_predictions(model, arrays, test_rows)
    truth_cal, _, mask_cal = _truth_current(arrays, calibration_rows)
    truth_test, _, mask_test = _truth_current(arrays, test_rows)
    quantiles = {}
    coverage = {}
    for index, target in enumerate(variables):
        cal_valid = mask_cal[:, index] & np.isfinite(calibration[:, index])
        test_valid = mask_test[:, index] & np.isfinite(test[:, index])
        if cal_valid.sum() < 30 or test_valid.sum() < 30:
            coverage[target] = {"status": "needs_calibration_audit", "n": int(test_valid.sum())}
            continue
        # Finite-sample split-conformal absolute residual quantile for nominal 90%.
        residual = np.abs(calibration[cal_valid, index] - truth_cal[cal_valid, index])
        q = float(np.quantile(residual, 0.90, method="higher"))
        lower = test[test_valid, index] - q
        upper = test[test_valid, index] + q
        observed = truth_test[test_valid, index]
        rate = float(((observed >= lower) & (observed <= upper)).mean())
        coverage[target] = {
            "n": int(test_valid.sum()),
            "coverage": rate,
            "interval_width": float(2.0 * q),
            "status": "PASS" if 0.87 <= rate <= 0.93 else "REJECT",
        }
        quantiles[target] = q
    return coverage


def _evaluate(model, arrays, frame, variables, rows, seed):
    prediction = _raw_predictions(model, arrays, rows)
    truth, current, mask = _truth_current(arrays, rows)
    groups = frame.iloc[rows]["subject_id"].to_numpy()
    output = {}
    for index, target in enumerate(variables):
        valid = mask[:, index] & np.isfinite(prediction[:, index])
        if valid.sum() < 30:
            output[target] = {"status": "insufficient_support", "n": int(valid.sum())}
            continue
        candidate_error = np.abs(prediction[valid, index] - truth[valid, index])
        persistence_error = np.abs(current[valid, index] - truth[valid, index])
        delta = candidate_error - persistence_error
        ci = _bootstrap_ci(groups[valid], delta, seed + index)
        output[target] = {
            "n": int(valid.sum()),
            "candidate_mae": float(candidate_error.mean()),
            "persistence_mae": float(persistence_error.mean()),
            "delta_vs_persistence": float(delta.mean()),
            "bootstrap_95_ci_delta": ci,
            "significant_vs_persistence": bool(ci is not None and ci[1] < 0.0),
        }
    return output


def _run_module(data_root, module, horizons, epochs, batch_size, max_stays):
    frame, variables = load_multihorizon_cohort(
        data_root, module, horizons, max_stays_per_horizon=max_stays, history=5, seed=0
    )
    patient_reports = {}
    for seed in SEEDS:
        train, test = _split(frame, "subject_id", seed, 0.25)
        fit, calibration = _split(frame.iloc[train].reset_index(drop=True), "subject_id", seed + 1000, 0.20)
        # Map calibration indices back to the original frame.
        calibration = train[calibration]
        fit = train[fit]
        model, arrays = _fit(frame, variables, fit, seed, epochs, batch_size)
        patient_reports[str(seed)] = {
            "metrics": _evaluate(model, arrays, frame, variables, test, seed),
            "conformal": _coverage(model, arrays, calibration, test, frame, variables),
        }
    hospital_report = {"status": "not_available"}
    valid_hospitals = frame["hospitalid"].dropna().nunique() if "hospitalid" in frame else 0
    if valid_hospitals >= 4:
        train, test = _split(frame, "hospitalid", 991, 0.25)
        fit, calibration = _split(frame.iloc[train].reset_index(drop=True), "subject_id", 1991, 0.20)
        calibration = train[calibration]
        fit = train[fit]
        model, arrays = _fit(frame, variables, fit, 991, epochs, batch_size)
        hospital_report = {
            "status": "run",
            "hospital_count": int(valid_hospitals),
            "metrics": _evaluate(model, arrays, frame, variables, test, 991),
            "conformal": _coverage(model, arrays, calibration, test, frame, variables),
        }
    aggregate = {}
    for target in variables:
        rows = [
            report["metrics"][target]
            for report in patient_reports.values()
            if target in report["metrics"] and report["metrics"][target].get("candidate_mae") is not None
        ]
        aggregate[target] = {
            "seeds": len(rows),
            "significant_seeds": sum(row["significant_vs_persistence"] for row in rows),
            "mean_delta_vs_persistence": float(np.mean([row["delta_vs_persistence"] for row in rows])) if rows else None,
            "validated_patient_gate": bool(
                len(rows) == len(SEEDS)
                and all(row["significant_vs_persistence"] for row in rows)
            ),
        }
    return {
        "module": module,
        "variables": list(variables),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "architecture": {
            "measurement_pure": True,
            "history_encoder": "GRU",
            "target_specific_heads": len(variables),
            "hand_beliefs_used": False,
        },
        "patient_heldout": patient_reports,
        "aggregate": aggregate,
        "hospital_heldout": hospital_report,
        "promotion_status": "candidate_only",
        "causal_claim_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--modules", default=",".join(MODULES))
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--output", default="/private/tmp/pure_jepa_full_gate.json")
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    results = {}
    for module in [value.strip() for value in args.modules.split(",") if value.strip()]:
        print(f"running full factual gate: {module}")
        try:
            results[module] = _run_module(
                args.data_root, module, horizons, args.epochs, args.batch_size, args.max_stays
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            results[module] = {"status": "not_available", "error": str(exc)}
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

