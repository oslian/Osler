"""Patient-balanced scoring audit for the joint whole-body JEPA.

The production gate intentionally stays conservative.  This diagnostic asks a
different question: did row-weighted MAE hide useful patient-level signal?
It evaluates the same held-out splits with patient-equalized error,
target/horizon-normalized error, direction accuracy, delta correlation,
patient-cluster bootstrap intervals, own-target ablation, and a held-out SVD
stability check.  It never promotes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from osler_jepa.cohort_signature import cohort_signature

from whole_body_joint_gate import (
    SEEDS,
    _ablate_target,
    _forward,
    _forward_many,
    _split_by_column,
    _split_train_calibration,
)
from whole_body_joint_jepa import (
    _build_supervision_strata,
    fit_joint,
    load_joint_cohort,
    load_joint_event_examples,
)


TARGETS = ("glucose", "map", "bicarbonate", "creatinine", "potassium", "heart_rate")
BOOTSTRAP_SAMPLES = 1000


def _raw_arrays(arrays):
    scaler, _, current, _, _, future, future_mask, _, horizon, _ = arrays[:10]
    current_raw = current * scaler.scales + scaler.medians
    future_raw = future * scaler.scales + scaler.medians
    return current_raw, future_raw, future_mask, horizon


def _cell_rows(arrays, rows, target_index, horizon_value):
    _, _, future_mask, horizon = _raw_arrays(arrays)
    rows = np.asarray(rows, dtype=np.int64)
    keep = (
        (horizon[rows] == float(horizon_value))
        & np.asarray(future_mask[rows, target_index], dtype=bool)
    )
    return rows[keep], np.flatnonzero(keep)


def _subject_mae(frame, rows, error):
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    values = {}
    for subject in np.unique(subjects):
        values[subject] = float(np.mean(error[subjects == subject]))
    return values


def _bootstrap_ci(values, samples=BOOTSTRAP_SAMPLES, seed=20260718):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return {"n_subjects": int(len(values)), "median": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "n_subjects": int(len(values)),
        "median": float(np.mean(values)),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def _metric_cell(frame, arrays, prediction, baseline_prediction, rows, target_index, horizon_value, train_rows):
    scaler = arrays[0]
    current_raw, future_raw, future_mask, horizon = _raw_arrays(arrays)
    selected, local = _cell_rows(arrays, rows, target_index, horizon_value)
    if len(selected) < 30:
        return None
    current = current_raw[selected, target_index]
    truth = future_raw[selected, target_index]
    # The joint model emits normalized values; all score comparisons must be
    # made in the target's original clinical units.
    pred = prediction[local, target_index] * scaler.scales[target_index] + scaler.medians[target_index]
    baseline = baseline_prediction[selected, target_index]
    candidate_error = np.abs(pred - truth)
    baseline_error = np.abs(baseline - truth)
    candidate_delta = pred - current
    truth_delta = truth - current
    nonzero = np.abs(truth_delta) > 1e-8
    target_name = scaler.variables[target_index]
    observed_training = pd.to_numeric(
        frame.iloc[np.asarray(train_rows)][f"{target_name}_t"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    observed_training = observed_training[np.isfinite(observed_training)]
    if len(observed_training) >= 4:
        target_scale = float(
            np.subtract(*np.quantile(observed_training, [0.75, 0.25]))
        )
        if target_scale < 1e-6:
            target_scale = float(np.std(observed_training))
    else:
        target_scale = 0.0
    target_scale = max(target_scale, float(scaler.scales[target_index]), 1e-6)
    candidate_by_subject = _subject_mae(frame, selected, candidate_error)
    baseline_by_subject = _subject_mae(frame, selected, baseline_error)
    difference_by_subject = {
        subject: candidate_by_subject[subject] - baseline_by_subject[subject]
        for subject in candidate_by_subject
        if subject in baseline_by_subject
    }
    direction = float(np.mean(np.sign(candidate_delta[nonzero]) == np.sign(truth_delta[nonzero]))) if nonzero.any() else None
    correlation = float(np.corrcoef(candidate_delta, truth_delta)[0, 1]) if np.std(candidate_delta) > 1e-9 and np.std(truth_delta) > 1e-9 else None
    row_delta = float(candidate_error.mean() - baseline_error.mean())
    patient_delta = float(np.mean(list(difference_by_subject.values()))) if difference_by_subject else None
    return {
        "target": str(target_index),
        "horizon_hours": float(horizon_value),
        "n_rows": int(len(selected)),
        "n_subjects": int(len(difference_by_subject)),
        "row_mae": float(candidate_error.mean()),
        "persistence_mae": float(baseline_error.mean()),
        "row_delta_vs_persistence": row_delta,
        "patient_equal_mae": float(np.mean(list(candidate_by_subject.values()))) if candidate_by_subject else None,
        "patient_equal_persistence_mae": float(np.mean(list(baseline_by_subject.values()))) if baseline_by_subject else None,
        "patient_equal_delta_vs_persistence": patient_delta,
        "patient_delta_bootstrap_ci": _bootstrap_ci(difference_by_subject.values()),
        "normalized_row_delta": float(row_delta / target_scale),
        "direction_accuracy": direction,
        "delta_correlation": correlation,
        "nonzero_delta_rows": int(nonzero.sum()),
    }


def _add_ablation_metrics(frame, arrays, full_prediction, own_prediction, rows, target_index, horizon_value):
    scaler = arrays[0]
    current_raw, future_raw, future_mask, horizon = _raw_arrays(arrays)
    selected, local = _cell_rows(arrays, rows, target_index, horizon_value)
    if len(selected) < 30:
        return None
    truth = future_raw[selected, target_index]
    full_raw = full_prediction[local, target_index] * scaler.scales[target_index] + scaler.medians[target_index]
    own_raw = own_prediction[local, target_index] * scaler.scales[target_index] + scaler.medians[target_index]
    full_error = np.abs(full_raw - truth)
    own_error = np.abs(own_raw - truth)
    full_by_subject = _subject_mae(frame, selected, full_error)
    own_by_subject = _subject_mae(frame, selected, own_error)
    differences = {
        subject: full_by_subject[subject] - own_by_subject[subject]
        for subject in full_by_subject
        if subject in own_by_subject
    }
    return {
        "full_row_mae": float(full_error.mean()),
        "own_target_row_mae": float(own_error.mean()),
        "row_delta_full_vs_own": float(full_error.mean() - own_error.mean()),
        "patient_equal_delta_full_vs_own": float(np.mean(list(differences.values()))) if differences else None,
        "patient_delta_bootstrap_ci_full_vs_own": _bootstrap_ci(differences.values(), seed=20260719),
    }


def _measurement_time_metrics(frame, arrays, prediction_log, rows, train_rows):
    """Score the observation process separately from physiological values."""

    if len(arrays) < 17 or prediction_log is None:
        return {}
    actual = np.asarray(arrays[15], dtype=np.float64)
    observed = np.asarray(arrays[16], dtype=bool)
    if not bool(observed.any()):
        return {}
    raw_prediction = np.asarray(prediction_log, dtype=np.float64)
    positive_log1p = np.logaddexp(0.0, raw_prediction)
    prediction_log = np.log1p(np.clip(np.expm1(positive_log1p), 0.0, 168.0))
    reports = {}
    train_rows = np.asarray(train_rows, dtype=np.int64)
    for index, target in enumerate(arrays[0].variables):
        test_mask = observed[np.asarray(rows, dtype=np.int64), index]
        if int(test_mask.sum()) < 20:
            continue
        test_rows_local = np.asarray(rows, dtype=np.int64)[test_mask]
        truth_log = np.log1p(actual[test_rows_local, index])
        candidate_log = prediction_log[test_mask, index]
        train_actual = actual[train_rows, index]
        train_valid = np.isfinite(train_actual)
        if int(train_valid.sum()) < 10:
            continue
        baseline_log = float(np.median(np.log1p(train_actual[train_valid])))
        candidate_error = np.abs(candidate_log - truth_log)
        baseline_error = np.abs(baseline_log - truth_log)
        candidate_by_subject = _subject_mae(frame, test_rows_local, candidate_error)
        baseline_by_subject = _subject_mae(frame, test_rows_local, baseline_error)
        differences = {
            subject: candidate_by_subject[subject] - baseline_by_subject[subject]
            for subject in candidate_by_subject
            if subject in baseline_by_subject
        }
        reports[target] = {
            "n_rows": int(test_mask.sum()),
            "n_subjects": int(len(differences)),
            "candidate_log1p_mae": float(candidate_error.mean()),
            "baseline_log1p_mae": float(baseline_error.mean()),
            "patient_equal_delta_vs_baseline": (
                float(np.mean(list(differences.values()))) if differences else None
            ),
            "patient_delta_bootstrap_ci": _bootstrap_ci(
                differences.values(), seed=20260721 + index
            ),
            "candidate_median_hours": float(np.median(np.expm1(candidate_log))),
            "truth_median_hours": float(np.median(np.expm1(truth_log))),
            "status": "candidate_only",
        }
    return reports


def _fit_svd_matrix(arrays, rows, n_state):
    scaler, _, current, _, _, future, future_mask, _, _, _ = arrays[:10]
    rows = np.asarray(rows, dtype=np.int64)
    x = current[rows]
    matrix = np.zeros((n_state, n_state), dtype=np.float64)
    for target_index in range(n_state):
        valid = np.asarray(future_mask[rows, target_index], dtype=bool)
        if valid.sum() < 30:
            continue
        y = future[rows[valid], target_index] - current[rows[valid], target_index]
        x_valid = x[valid]
        y = (y - y.mean()) / max(float(y.std()), 1e-8)
        # The state matrix intentionally contains correlated physiological
        # variables. LSQR preserves the ridge objective without the unstable
        # normal-equation solve used by the default dense solver.
        matrix[target_index] = Ridge(alpha=1.0, solver="lsqr").fit(
            x_valid, y
        ).coef_
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    probabilities = singular / max(float(singular.sum()), 1e-12)
    effective_rank = float(np.exp(-(probabilities * np.log(probabilities + 1e-12)).sum()))
    return matrix, singular, vt, effective_rank


def _svd_stability(arrays, fit_rows, test_rows, n_state):
    fit_matrix, fit_singular, fit_vt, fit_rank = _fit_svd_matrix(arrays, fit_rows, n_state)
    test_matrix, test_singular, test_vt, test_rank = _fit_svd_matrix(arrays, test_rows, n_state)
    mode_cosines = [
        float(abs(np.dot(fit_vt[index], test_vt[index])))
        for index in range(min(5, len(fit_vt), len(test_vt)))
    ]
    mode_count = min(5, len(fit_vt), len(test_vt))
    principal_cosines = (
        np.linalg.svd(
            fit_vt[:mode_count] @ test_vt[:mode_count].T,
            compute_uv=False,
        )
        if mode_count
        else np.asarray([], dtype=np.float64)
    )
    return {
        "fit_effective_rank": fit_rank,
        "test_effective_rank": test_rank,
        "fit_singular_values": [float(value) for value in fit_singular[:5]],
        "test_singular_values": [float(value) for value in test_singular[:5]],
        "top_mode_absolute_cosines": mode_cosines,
        "mean_top_mode_cosine": float(np.mean(mode_cosines)) if mode_cosines else None,
        "top_subspace_principal_cosines": [
            float(value) for value in principal_cosines
        ],
        "mean_top_subspace_cosine": (
            float(np.mean(principal_cosines)) if len(principal_cosines) else None
        ),
    }


def _latent_svd_summary(latent):
    latent = np.asarray(latent, dtype=np.float64)
    latent = latent[np.isfinite(latent).all(axis=1)]
    if len(latent) < 3 or latent.shape[1] < 2:
        return None
    centered = latent - latent.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    probabilities = singular / max(float(singular.sum()), 1e-12)
    effective_rank = float(
        np.exp(-(probabilities * np.log(probabilities + 1e-12)).sum())
    )
    return {
        "effective_rank": effective_rank,
        "singular_values": [float(value) for value in singular[:8]],
        "modes": vt[: min(5, len(vt))],
    }


def _latent_svd_stability(model, arrays, fit_rows, test_rows):
    """Audit the learned JEPA latent itself, not only the data Jacobian."""

    fit = _latent_svd_summary(
        _forward(model, arrays, np.asarray(fit_rows, dtype=np.int64), output_key="latent")
    )
    test = _latent_svd_summary(
        _forward(model, arrays, np.asarray(test_rows, dtype=np.int64), output_key="latent")
    )
    if fit is None or test is None:
        return {
            "fit_effective_rank": None,
            "test_effective_rank": None,
            "mean_top_mode_cosine": None,
            "pass": False,
        }
    count = min(len(fit["modes"]), len(test["modes"]))
    cosines = [
        float(abs(np.dot(fit["modes"][index], test["modes"][index])))
        for index in range(count)
    ]
    principal_cosines = (
        np.linalg.svd(
            fit["modes"][:count] @ test["modes"][:count].T,
            compute_uv=False,
        )
        if count
        else np.asarray([], dtype=np.float64)
    )
    subspace_similarity = (
        float(np.mean(principal_cosines)) if len(principal_cosines) else None
    )
    ratio = test["effective_rank"] / max(fit["effective_rank"], 1e-12)
    passed = bool(
        fit["effective_rank"] >= 2.0
        and test["effective_rank"] >= 2.0
        and 0.50 <= ratio <= 2.0
        and subspace_similarity is not None
        and subspace_similarity >= 0.50
    )
    return {
        "fit_effective_rank": fit["effective_rank"],
        "test_effective_rank": test["effective_rank"],
        "rank_ratio": ratio,
        "fit_singular_values": fit["singular_values"],
        "test_singular_values": test["singular_values"],
        "top_mode_absolute_cosines": cosines,
        "mean_top_mode_cosine": float(np.mean(cosines)) if cosines else None,
        "top_subspace_principal_cosines": [
            float(value) for value in principal_cosines
        ],
        "mean_top_subspace_cosine": subspace_similarity,
        "mode_comparison_policy": (
            "promotion uses rotation-invariant principal-subspace overlap; "
            "same-index vector cosine is descriptive only"
        ),
        "pass": passed,
    }


def run(args):
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    if args.event_examples is not None:
        frame, variables, module_names = load_joint_event_examples(
            args.event_examples,
            modules,
            horizons,
            args.max_stays,
            min_modules=args.min_modules,
            organ_contract_mode=args.organ_contract_mode,
            history_mode=args.history_mode,
            history_steps=args.history_steps,
            history_lags_hours=args._history_lags_hours,
        )
    else:
        frame, variables, module_names = load_joint_cohort(
            args.data_root,
            modules,
            horizons,
            args.max_stays,
            min_modules=args.min_modules,
            alignment=args.alignment,
            asof_max_age_hours=args.asof_max_age_hours,
            asof_age_policy=args.asof_age_policy,
            include_treatment_context=args.include_treatment_context,
        )
    unknown = sorted(set(targets) - set(variables))
    if unknown:
        raise ValueError(f"Targets absent from cohort: {unknown}")
    target_indices = [variables.index(target) for target in targets]
    reports = []
    for seed in SEEDS:
        train_rows, test_rows = _split_by_column(frame, "subject_id", seed)
        fit_rows, _ = _split_train_calibration(frame, train_rows, seed + 1001)
        model, arrays, losses = fit_joint(
            frame,
            variables,
            module_names,
            fit_rows,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            measurement_process_mode=args.measurement_process_mode,
            cross_forecast_weight=args.cross_forecast_weight,
            masked_forecast_probability=args.masked_forecast_probability,
            module_mask_probability=args.module_mask_probability,
            world_model=args.world_model,
            world_model_weight=args.world_model_weight,
            target_momentum=args.target_momentum,
            future_target_mode=args.future_target_mode,
            target_min_observations=args.target_min_observations,
            regime_balanced_loss=args.regime_balanced_loss,
            regime_count=args.regime_count,
            state_normalization=args.state_normalization,
            hospital_invariance_weight=args.hospital_invariance_weight,
            target_encoder_mode=args.target_encoder_mode,
            target_horizon_regime_adapter=args.target_horizon_regime_adapter,
            patient_residual_adapter=args.patient_residual_adapter,
            future_task_balanced_sampling=args.future_task_balanced_sampling,
            coupling_preservation_weight=args.coupling_preservation_weight,
            uncertainty_gate=args.uncertainty_gate,
            validated_edge_adapters=args.validated_edge_adapters,
            validated_edge_weight=args.validated_edge_weight,
            validated_edge_stage_epochs=args.validated_edge_stage_epochs,
            target_head_stage_epochs=args.target_head_stage_epochs,
            measurement_time_head=args.measurement_time_head,
            measurement_time_weight=args.measurement_time_weight,
            uncertainty_calibration_weight=args.uncertainty_calibration_weight,
            include_treatment_context=args.include_treatment_context,
            belief_input_mode=args.belief_input_mode,
        )
        prediction = _forward(model, arrays, test_rows)
        measurement_time_prediction = None
        if model.measurement_time_head:
            measurement_time_prediction = _forward(
                model,
                arrays,
                test_rows,
                output_key="next_measurement_time_log",
            )
        current_raw, _, _, _ = _raw_arrays(arrays)
        ablated_views = [
            (
                index,
                _ablate_target(
                    arrays,
                    index,
                    len(variables),
                    len(module_names),
                    None,
                ),
            )
            for index in target_indices
        ]
        ablated = _forward_many(model, ablated_views, test_rows, output_keys=("value",))
        cells = {}
        for target, target_index in zip(targets, target_indices):
            for horizon_value in horizons:
                key = f"{target}@{horizon_value}h"
                cell = _metric_cell(
                    frame,
                    arrays,
                    prediction,
                    current_raw,
                    test_rows,
                    target_index,
                    horizon_value,
                    fit_rows,
                )
                if cell is None:
                    continue
                cell["target"] = target
                cell["own_target_ablation"] = _add_ablation_metrics(
                    frame,
                    arrays,
                    prediction,
                    ablated[target_index]["value"],
                    test_rows,
                    target_index,
                    horizon_value,
                )
                cells[key] = cell
        reports.append({
            "seed": int(seed),
            "cells": cells,
            "measurement_time": _measurement_time_metrics(
                frame,
                arrays,
                measurement_time_prediction,
                test_rows,
                fit_rows,
            ),
            "svd": _svd_stability(arrays, fit_rows, test_rows, len(variables)),
            "latent_svd": _latent_svd_stability(
                model, arrays, fit_rows, test_rows
            ),
            "loss": float(losses[-1]),
        })
        print(json.dumps({"seed": seed, "cells": len(cells), "loss": losses[-1]}), flush=True)
    source_contract = (
        "canonical_event_ledger"
        if args.event_examples is not None
        else "legacy_module_transition"
    )
    return {
        "schema": "whole_body_scoring_audit.v1",
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "hospitals": int(frame["hospitalid"].nunique()),
        "modules": list(module_names),
        "variables": list(variables),
        "targets": list(targets),
        "reports": reports,
        "source_contract": source_contract,
        "model_configuration": {
            "cohort_signature": cohort_signature(frame, variables),
            "source_contract": source_contract,
            "organ_contract_mode": str(args.organ_contract_mode),
            "history_mode": str(args.history_mode),
            "history_steps": int(args.history_steps),
            "history_lags_hours": list(args._history_lags_hours),
            "modules": list(module_names),
            "horizons": list(horizons),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "cross_forecast_weight": float(args.cross_forecast_weight),
            "masked_forecast_probability": float(
                args.masked_forecast_probability
            ),
            "module_mask_probability": float(args.module_mask_probability),
            "world_model": bool(args.world_model),
            "world_model_weight": float(args.world_model_weight),
            "target_momentum": float(args.target_momentum),
            "future_target_mode": str(args.future_target_mode),
            "target_min_observations": int(args.target_min_observations),
            "regime_balanced_loss": bool(args.regime_balanced_loss),
            "regime_count": int(args.regime_count),
            "measurement_process_mode": str(args.measurement_process_mode),
            "measurement_time_head": bool(args.measurement_time_head),
            "measurement_time_weight": float(args.measurement_time_weight),
            "uncertainty_calibration_weight": float(
                args.uncertainty_calibration_weight
            ),
            "include_treatment_context": bool(args.include_treatment_context),
            "state_normalization": str(args.state_normalization),
            "hospital_invariance_weight": float(args.hospital_invariance_weight),
            "target_encoder_mode": str(args.target_encoder_mode),
            "target_horizon_regime_adapter": bool(args.target_horizon_regime_adapter),
            "patient_residual_adapter": bool(args.patient_residual_adapter),
            "future_task_balanced_sampling": bool(args.future_task_balanced_sampling),
            "coupling_preservation_weight": float(args.coupling_preservation_weight),
            "uncertainty_gate": bool(args.uncertainty_gate),
            "validated_edge_adapters": bool(args.validated_edge_adapters),
            "validated_edge_weight": float(args.validated_edge_weight),
            "validated_edge_stage_epochs": int(args.validated_edge_stage_epochs),
            "target_head_stage_epochs": int(args.target_head_stage_epochs),
            "belief_input_mode": str(args.belief_input_mode),
        },
        "metric_policy": {
            "primary": "patient_equalized MAE and patient-cluster bootstrap delta CI",
            "secondary": ["row_mae", "target_horizon_normalized_delta", "direction_accuracy", "delta_correlation", "own_target_ablation"],
            "svd": "data-Jacobian and learned-latent fit/test singular-mode stability",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--event-examples", type=Path)
    parser.add_argument(
        "--organ-contract-mode",
        choices=("canonical_organs", "overlap"),
        default="canonical_organs",
    )
    parser.add_argument(
        "--history-mode",
        choices=("multiscale", "recent"),
        default="multiscale",
    )
    parser.add_argument("--history-steps", type=int, default=5)
    parser.add_argument(
        "--history-lags-hours",
        default="48,24,12,8,6,4,3,2,1,0",
    )
    parser.add_argument("--modules", required=True)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=200)
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument("--alignment", choices=("exact", "asof"), default="asof")
    parser.add_argument("--asof-max-age-hours", type=float, default=6.0)
    parser.add_argument("--asof-age-policy", choices=("fixed", "horizon_scaled"), default="horizon_scaled")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cross-forecast-weight", type=float, default=1.0)
    parser.add_argument("--masked-forecast-probability", type=float, default=0.50)
    parser.add_argument("--module-mask-probability", type=float, default=0.75)
    parser.add_argument("--world-model", action="store_true")
    parser.add_argument("--world-model-weight", type=float, default=1.0)
    parser.add_argument("--target-momentum", type=float, default=0.99)
    parser.add_argument("--future-target-mode", choices=("window", "single"), default="window")
    parser.add_argument("--target-min-observations", type=int, default=2)
    parser.add_argument("--regime-balanced-loss", action="store_true")
    parser.add_argument("--regime-count", type=int, default=3)
    parser.add_argument("--state-normalization", choices=("standard", "robust"), default="robust")
    parser.add_argument("--measurement-process-mode", choices=("observed", "neutralized"), default="neutralized")
    parser.add_argument("--hospital-invariance-weight", type=float, default=0.05)
    parser.add_argument("--target-encoder-mode", choices=("full", "fast"), default="fast")
    parser.add_argument("--target-horizon-regime-adapter", action="store_true")
    parser.add_argument("--patient-residual-adapter", action="store_true")
    parser.add_argument("--future-task-balanced-sampling", action="store_true")
    parser.add_argument("--coupling-preservation-weight", type=float, default=0.0)
    parser.add_argument("--uncertainty-gate", action="store_true")
    parser.add_argument("--validated-edge-adapters", action="store_true")
    parser.add_argument("--validated-edge-weight", type=float, default=0.0)
    parser.add_argument("--validated-edge-stage-epochs", type=int, default=0)
    parser.add_argument("--target-head-stage-epochs", type=int, default=0)
    parser.add_argument("--measurement-time-head", action="store_true")
    parser.add_argument("--measurement-time-weight", type=float, default=0.10)
    parser.add_argument(
        "--uncertainty-calibration-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument("--include-treatment-context", action="store_true")
    parser.add_argument(
        "--belief-input-mode",
        choices=("observed", "current_only", "placebo_time_only"),
        default="observed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args._history_lags_hours = tuple(
        float(value)
        for value in args.history_lags_hours.split(",")
        if value.strip()
    )
    output = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rows": output["rows"], "seeds": len(output["reports"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
