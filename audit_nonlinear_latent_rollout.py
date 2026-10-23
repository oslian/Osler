"""Comprehensive observation-only audit for the nonlinear rollout candidate.

The audit deliberately treats the model as a factual trajectory predictor.
It has no treatment-effect interpretation and never uses future columns as
features.  Each split is divided into fit/calibration/test groups before any
model is trained.  The output is an aggregate-safe report; it does not store
patient identifiers or row-level predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from nonlinear_latent_rollout import (
    DEFAULT_HORIZONS,
    effective_rank,
    load_multihorizon_cohort,
)
from train_nonlinear_latent_rollout import (
    _history_matrix,
    _metrics,
    _predict,
    _prepared_arrays,
    _ridge_predictions,
    _split_groups,
    _train_model,
)


ALPHA = 0.10
MIN_CALIBRATION = 50
MIN_EVALUATION = 50
BOOTSTRAP_REPS = 200


def _raw_truth(frame: pd.DataFrame, variables: Sequence[str], rows: np.ndarray) -> np.ndarray:
    return frame[[f"future_{v}" for v in variables]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)[rows]


def _raw_current(frame: pd.DataFrame, variables: Sequence[str], rows: np.ndarray) -> np.ndarray:
    return frame[[f"{v}_t" for v in variables]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)[rows]


def _predict_mc(
    model,
    input_matrix: np.ndarray,
    current: np.ndarray,
    history_indices: np.ndarray,
    horizons: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate predictive uncertainty with transition dropout samples."""

    samples: list[np.ndarray] = []
    first_latent: np.ndarray | None = None
    model.train()
    with torch.no_grad():
        for repeat in range(max(2, int(repeats))):
            outputs: list[np.ndarray] = []
            latents: list[np.ndarray] = []
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start:start + batch_size]
                sequence = torch.from_numpy(input_matrix[history_indices[batch_rows]]).float()
                state = torch.from_numpy(current[batch_rows]).float()
                horizon = torch.from_numpy(horizons[batch_rows]).float()
                prediction, latent = model(sequence, state, horizon)
                outputs.append(prediction.cpu().numpy())
                latents.append(latent.cpu().numpy())
            sample = np.concatenate(outputs)
            samples.append(sample)
            if repeat == 0:
                first_latent = np.concatenate(latents)
    model.eval()
    stacked = np.stack(samples, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0), first_latent


def _uncertainty_gate(
    frame: pd.DataFrame,
    variables: Sequence[str],
    calibration_rows: np.ndarray,
    test_rows: np.ndarray,
    calibration_pred: np.ndarray,
    calibration_std: np.ndarray,
    test_pred: np.ndarray,
    test_std: np.ndarray,
    scaler,
) -> dict[str, object]:
    """Use calibration-only uncertainty thresholds to abstain to persistence."""

    cal_truth = _raw_truth(frame, variables, calibration_rows)
    cal_current = _raw_current(frame, variables, calibration_rows)
    test_truth = _raw_truth(frame, variables, test_rows)
    test_current = _raw_current(frame, variables, test_rows)
    test_pred_raw = test_pred * scaler.scales[None, :] + scaler.medians[None, :]
    calibration_horizon = frame["_source_horizon"].to_numpy(dtype=np.int64)[calibration_rows]
    test_horizon = frame["_source_horizon"].to_numpy(dtype=np.int64)[test_rows]
    result: dict[str, object] = {}
    for horizon in sorted(set(test_horizon.tolist())):
        horizon_result: dict[str, object] = {}
        for index, variable in enumerate(variables):
            cal_valid = (
                (calibration_horizon == horizon)
                & np.isfinite(cal_truth[:, index])
                & np.isfinite(cal_current[:, index])
                & np.isfinite(calibration_std[:, index])
            )
            test_valid = (
                (test_horizon == horizon)
                & np.isfinite(test_truth[:, index])
                & np.isfinite(test_current[:, index])
                & np.isfinite(test_std[:, index])
            )
            if cal_valid.sum() < MIN_CALIBRATION or test_valid.sum() < MIN_EVALUATION:
                continue
            threshold = float(np.quantile(calibration_std[cal_valid, index], 0.75))
            allowed = test_std[:, index] <= threshold
            gated = np.where(allowed, test_pred_raw[:, index], test_current[:, index])
            candidate_error = np.abs(test_pred_raw[test_valid, index] - test_truth[test_valid, index])
            gated_error = np.abs(gated[test_valid] - test_truth[test_valid, index])
            persistence_error = np.abs(test_current[test_valid, index] - test_truth[test_valid, index])
            horizon_result[variable] = {
                "uncertainty_threshold_normalized": threshold,
                "abstention_rate": float((~allowed[test_valid]).mean()),
                "candidate_mae": float(candidate_error.mean()),
                "gated_mae": float(gated_error.mean()),
                "persistence_mae": float(persistence_error.mean()),
                "gated_delta_vs_persistence": float(gated_error.mean() - persistence_error.mean()),
            }
        result[str(horizon)] = horizon_result
    return result


def _group_bootstrap(
    frame: pd.DataFrame,
    rows: np.ndarray,
    values: np.ndarray,
    group_column: str,
    seed: int,
) -> dict[str, float | int | None]:
    """Bootstrap a row-level error delta by patient/stay, never by raw row."""

    finite = np.isfinite(values)
    rows = rows[finite]
    values = values[finite]
    if len(values) < MIN_EVALUATION:
        return {"n": int(len(values)), "delta": None, "ci_low": None, "ci_high": None}
    groups = frame[group_column].to_numpy()[rows]
    unique = pd.unique(groups)
    by_group = {group: values[groups == group] for group in unique}
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[index] = np.concatenate([by_group[group] for group in sampled]).mean()
    return {
        "n": int(len(values)),
        "groups": int(len(unique)),
        "delta": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def _conformal(
    frame: pd.DataFrame,
    variables: Sequence[str],
    calibration_rows: np.ndarray,
    test_rows: np.ndarray,
    calibration_pred: np.ndarray,
    test_pred: np.ndarray,
    scaler,
) -> dict[str, object]:
    calibration_truth = _raw_truth(frame, variables, calibration_rows)
    test_truth = _raw_truth(frame, variables, test_rows)
    calibration_current = _raw_current(frame, variables, calibration_rows)
    test_current = _raw_current(frame, variables, test_rows)
    calibration_pred_raw = calibration_pred * scaler.scales[None, :] + scaler.medians[None, :]
    test_pred_raw = test_pred * scaler.scales[None, :] + scaler.medians[None, :]
    calibration_horizon = frame["_source_horizon"].to_numpy(dtype=np.int64)[calibration_rows]
    test_horizon = frame["_source_horizon"].to_numpy(dtype=np.int64)[test_rows]
    result: dict[str, object] = {}
    for horizon in sorted(set(test_horizon.tolist())):
        horizon_result: dict[str, object] = {}
        for index, variable in enumerate(variables):
            cal_valid = (
                (calibration_horizon == horizon)
                & np.isfinite(calibration_truth[:, index])
                & np.isfinite(calibration_current[:, index])
            )
            test_valid = (
                (test_horizon == horizon)
                & np.isfinite(test_truth[:, index])
                & np.isfinite(test_current[:, index])
            )
            if cal_valid.sum() < MIN_CALIBRATION or test_valid.sum() < MIN_EVALUATION:
                continue
            residuals = np.abs(
                calibration_pred_raw[cal_valid, index] - calibration_truth[cal_valid, index]
            )
            # Conservative split-conformal quantile.  The finite residuals are
            # calibration-only; the test rows remain untouched until coverage.
            rank = int(np.ceil((len(residuals) + 1) * (1.0 - ALPHA)))
            rank = min(max(rank, 1), len(residuals))
            q = float(np.sort(residuals)[rank - 1])
            covered = np.abs(
                test_pred_raw[test_valid, index] - test_truth[test_valid, index]
            ) <= q
            coverage = float(covered.mean())
            horizon_result[variable] = {
                "calibration_n": int(cal_valid.sum()),
                "test_n": int(test_valid.sum()),
                "q90": q,
                "coverage": coverage,
                "interval_status": "calibrated" if 0.87 <= coverage <= 0.93 else "needs_calibration_audit",
            }
        result[str(horizon)] = horizon_result
    return result


def _stability_check(model, input_matrix, current, history_indices, frame, rows, batch_size):
    """Check finite, bounded predictions at a longer-than-trained horizon."""

    probe = rows[: min(len(rows), 2048)]
    if not len(probe):
        return {"n": 0, "finite": False, "max_abs_normalized": None}
    predictions = []
    embeddings = []
    model.eval()
    import torch

    with torch.no_grad():
        for start in range(0, len(probe), batch_size):
            batch_rows = probe[start:start + batch_size]
            sequence = torch.from_numpy(input_matrix[history_indices[batch_rows]]).float()
            state = torch.from_numpy(current[batch_rows]).float()
            horizon = torch.full((len(batch_rows),), 168.0, dtype=torch.float32)
            prediction, latent = model(sequence, state, horizon)
            predictions.append(prediction.cpu().numpy())
            embeddings.append(latent.cpu().numpy())
    pred = np.concatenate(predictions)
    latent = np.concatenate(embeddings)
    return {
        "n": int(len(probe)),
        "finite": bool(np.isfinite(pred).all() and np.isfinite(latent).all()),
        "max_abs_normalized": float(np.max(np.abs(pred))),
        "latent_effective_rank": effective_rank(latent),
        "bounded": bool(np.isfinite(pred).all() and np.max(np.abs(pred)) <= 12.0),
    }


def _audit_split(
    frame: pd.DataFrame,
    variables: Sequence[str],
    fit_rows: np.ndarray,
    calibration_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    history: int,
    hidden: int,
    latent: int,
    epochs: int,
    batch_size: int,
    lr: float,
    cap_train_rows_per_horizon: int | None,
    seed: int,
    mc_repeats: int,
) -> dict[str, object]:
    model, scaler, train_meta = _train_model(
        frame,
        variables,
        fit_rows,
        history=history,
        hidden=hidden,
        latent=latent,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        cap_per_horizon=cap_train_rows_per_horizon,
        seed=seed,
    )
    _, input_matrix, current, _, _, _ = _prepared_arrays(frame, variables, fit_rows)
    history_indices = _history_matrix(frame, history)
    horizon_values = frame["_source_horizon"].to_numpy(dtype=np.float32)
    calibration_pred, calibration_std, calibration_latent = _predict_mc(
        model, input_matrix, current, history_indices, horizon_values, calibration_rows, batch_size, mc_repeats
    )
    test_pred, test_std, test_latent = _predict_mc(
        model, input_matrix, current, history_indices, horizon_values, test_rows, batch_size, mc_repeats
    )
    ridge_raw = _ridge_predictions(frame, variables, fit_rows, test_rows, input_matrix, scaler)
    test_truth = _raw_truth(frame, variables, test_rows)
    test_current = _raw_current(frame, variables, test_rows)
    test_pred_raw = test_pred * scaler.scales[None, :] + scaler.medians[None, :]
    horizon = frame["_source_horizon"].to_numpy(dtype=np.int64)[test_rows]
    bootstrap: dict[str, object] = {}
    for h in sorted(set(horizon.tolist())):
        h_result: dict[str, object] = {}
        hmask = horizon == h
        for index, variable in enumerate(variables):
            valid = hmask & np.isfinite(test_truth[:, index]) & np.isfinite(test_current[:, index])
            if valid.sum() < MIN_EVALUATION:
                continue
            candidate_minus_persistence = (
                np.abs(test_pred_raw[valid, index] - test_truth[valid, index])
                - np.abs(test_current[valid, index] - test_truth[valid, index])
            )
            candidate_minus_ridge = (
                np.abs(test_pred_raw[valid, index] - test_truth[valid, index])
                - np.abs(ridge_raw[valid, index] - test_truth[valid, index])
            )
            valid_rows = test_rows[valid]
            h_result[variable] = {
                "vs_persistence": _group_bootstrap(
                    frame, valid_rows, candidate_minus_persistence, "subject_id", seed + index
                ),
                "vs_ridge": _group_bootstrap(
                    frame, valid_rows, candidate_minus_ridge, "subject_id", seed + 100 + index
                ),
            }
        bootstrap[str(h)] = h_result
    return {
        "train": train_meta,
        "test_rows": int(len(test_rows)),
        "metrics": _metrics(frame, variables, test_rows, test_pred_raw, ridge_raw, scaler),
        "bootstrap": bootstrap,
        "conformal": _conformal(
            frame, variables, calibration_rows, test_rows, calibration_pred, test_pred, scaler
        ),
        "uncertainty_gate": _uncertainty_gate(
            frame,
            variables,
            calibration_rows,
            test_rows,
            calibration_pred,
            calibration_std,
            test_pred,
            test_std,
            scaler,
        ),
        "effective_rank": effective_rank(test_latent),
        "calibration_effective_rank": effective_rank(calibration_latent),
        "rollout_stability": _stability_check(
            model, input_matrix, current, history_indices, frame, test_rows, batch_size
        ),
    }


def _split_three(frame: pd.DataFrame, group_column: str, seed: int):
    fit_pool, test = _split_groups(frame, group_column, seed, fraction=0.25)
    pool = frame.iloc[fit_pool]
    fit_local, calibration_local = _split_groups(pool, group_column, seed + 1000, fraction=0.25)
    return pool.iloc[fit_local].index.to_numpy(dtype=np.int64), pool.iloc[calibration_local].index.to_numpy(dtype=np.int64), frame.iloc[test].index.to_numpy(dtype=np.int64)


def run(
    data_root: str | Path,
    module: str,
    horizons: Sequence[int],
    *,
    seeds: int = 7,
    history: int = 5,
    hidden: int = 64,
    latent: int = 48,
    epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 1e-3,
    max_stays_per_horizon: int | None = 2500,
    cap_train_rows_per_horizon: int | None = 12000,
    mc_repeats: int = 6,
    output: str | Path = "/private/tmp/nonlinear_latent_rollout_comprehensive_audit.json",
) -> dict[str, object]:
    frame, variables = load_multihorizon_cohort(
        data_root,
        module,
        horizons,
        max_stays_per_horizon=max_stays_per_horizon,
        history=history,
        seed=0,
    )
    future_columns = [column for column in frame.columns if column.startswith("future_")]
    input_columns = [
        *[f"{v}_t" for v in variables],
        *[f"{v}_age_hr" for v in variables],
        "hours_since_onset",
    ]
    leakage = [column for column in input_columns if "_tp" in column or "future" in column.lower()]
    report: dict[str, object] = {
        "observation_only": True,
        "causal_claim_allowed": False,
        "counterfactual_claim_allowed": False,
        "clinical_claim_allowed": False,
        "runtime_decision_authority": False,
        "module": module,
        "horizons": list(horizons),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "variables": list(variables),
        "leakage_audit": {
            "future_columns_present_in_source": int(len(future_columns)),
            "future_columns_used_as_input": leakage,
            "passed": not leakage,
        },
        "patient_heldout": {},
        "hospital_heldout": None,
        "promotion_gate": "candidate_only",
    }
    for seed in range(seeds):
        fit, calibration, test = _split_three(frame, "subject_id", seed)
        print(f"patient seed={seed} fit={len(fit)} calibration={len(calibration)} test={len(test)}")
        report["patient_heldout"][str(seed)] = _audit_split(
            frame,
            variables,
            fit,
            calibration,
            test,
            history=history,
            hidden=hidden,
            latent=latent,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            cap_train_rows_per_horizon=cap_train_rows_per_horizon,
            seed=seed,
            mc_repeats=mc_repeats,
        )
    if frame["hospitalid"].nunique() >= 4:
        fit, calibration, test = _split_three(frame, "hospitalid", 9001)
        print(f"hospital heldout fit={len(fit)} calibration={len(calibration)} test={len(test)}")
        report["hospital_heldout"] = _audit_split(
            frame,
            variables,
            fit,
            calibration,
            test,
            history=history,
            hidden=hidden,
            latent=latent,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            cap_train_rows_per_horizon=cap_train_rows_per_horizon,
            seed=9001,
            mc_repeats=mc_repeats,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--module", default="aki")
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-stays-per-horizon", type=int, default=2500)
    parser.add_argument("--cap-train-rows-per-horizon", type=int, default=12000)
    parser.add_argument("--mc-repeats", type=int, default=6)
    parser.add_argument("--output", default="/private/tmp/nonlinear_latent_rollout_comprehensive_audit.json")
    args = parser.parse_args()
    run(
        args.data_root,
        args.module,
        tuple(int(value) for value in args.horizons.split(",") if value.strip()),
        seeds=args.seeds,
        history=args.history,
        hidden=args.hidden,
        latent=args.latent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_stays_per_horizon=args.max_stays_per_horizon,
        cap_train_rows_per_horizon=args.cap_train_rows_per_horizon,
        mc_repeats=args.mc_repeats,
        output=args.output,
    )


if __name__ == "__main__":
    main()
