"""Audit one shared renal patient state across validated target-horizon cells.

The shared candidate uses one causal kidney-organ history encoder and one renal
belief state.  Each target-horizon keeps an independent point and scale head.
Promotion is per cell and requires the shared model to beat the population
anchor, a capacity-matched patient-mismatched placebo, and an independently
trained patient-state model on the same held-out rows.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from osler_jepa.interval_personalization import (
    mondrian_online_adaptive_interval_widths,
)
from osler_jepa.patient_state import (
    GatedSharedPrivateRenalForecaster,
    SharedRenalPatientStateForecaster,
)
from patient_state_adapter_audit import (
    PreparedData,
    _balanced_rows,
    _comparison,
    _device,
    _feature_columns,
    _fit_scaler,
    _history_indices,
    _interval_narrowing_claim_allowed,
    _mismatched_patient_rows,
    _model_arrays,
    _predict,
    _select_residual_gain,
    _split_rows,
    _summary,
    _temporal_regimes,
    _train_model,
    _width_comparison,
)
from aki_renal_belief import renal_belief_state_v3_features


TASKS = (
    ("creatinine", 3),
    ("creatinine", 12),
    ("creatinine", 24),
    ("bun", 3),
    ("bun", 6),
    ("bun", 12),
    ("bun", 24),
    ("bun", 48),
    ("urine_output", 6),
    ("urine_output", 12),
    ("map", 3),
    ("map", 6),
)


def task_key(target: str, horizon: int) -> str:
    return f"{target}@{horizon}h"


@dataclass
class SharedRenalData:
    prepared: PreparedData
    labels: np.ndarray
    current_targets: np.ndarray


def prepare_shared_data(
    root: Path,
    *,
    history_steps: int,
    max_rows_per_subject: int,
    max_subjects: int | None,
    seed: int,
) -> SharedRenalData:
    base_path = root / "eicu_aki_transitions_3h.parquet"
    metadata_columns = (
        "subject_id",
        "stay_id",
        "t",
        "hospitalid",
        "unittype",
        "hours_since_onset",
    )
    renal_variables = (
        "creatinine",
        "bun",
        "urine_output",
        "potassium",
        "bicarbonate",
        "sodium",
        "map",
    )
    measurement_columns = tuple(
        column
        for variable in renal_variables
        for column in (f"{variable}_t", f"{variable}_age_hr")
    )
    treatment_columns = (
        "hist_fluids",
        "hist_vasopressor",
        "hist_diuretics",
        "hist_renal_replacement",
        "hist_nephrotoxin",
    )
    base_future_columns = tuple(
        f"{target}_tp3" for target, horizon in TASKS if horizon == 3
    )
    base_columns = tuple(
        dict.fromkeys(
            (*metadata_columns, *measurement_columns, *treatment_columns, *base_future_columns)
        )
    )
    frame = pd.read_parquet(base_path, columns=list(base_columns)).reset_index(drop=True)
    frame = _balanced_rows(
        frame,
        max_rows_per_subject=max_rows_per_subject,
        max_subjects=max_subjects,
        seed=seed,
    )
    if frame.duplicated(["stay_id", "t"]).any():
        raise ValueError("base renal cohort contains duplicate anchors")

    tasks_by_horizon: dict[int, list[str]] = {}
    for target, horizon in TASKS:
        tasks_by_horizon.setdefault(horizon, []).append(target)
    for horizon, targets in tasks_by_horizon.items():
        if horizon == 3:
            continue
        future_columns = [f"{target}_tp{horizon}" for target in targets]
        source = pd.read_parquet(
            root / f"eicu_aki_transitions_{horizon}h.parquet",
            columns=["stay_id", "t", *future_columns],
        )
        if source.duplicated(["stay_id", "t"]).any():
            raise ValueError(f"{horizon}h renal cohort contains duplicate anchors")
        frame = frame.merge(
            source,
            on=["stay_id", "t"],
            how="left",
            validate="one_to_one",
        )
        del source
        gc.collect()

    feature_columns = _feature_columns(frame)
    current = frame[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    beliefs = renal_belief_state_v3_features(frame).apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    indices, history_mask = _history_indices(frame, history_steps)
    labels = np.column_stack(
        [
            pd.to_numeric(frame[f"{target}_tp{horizon}"], errors="coerce")
            for target, horizon in TASKS
        ]
    ).astype(np.float64)
    current_targets = np.column_stack(
        [pd.to_numeric(frame[f"{target}_t"], errors="coerce") for target, _ in TASKS]
    ).astype(np.float64)
    labels[~np.isfinite(current_targets)] = np.nan
    support = np.isfinite(labels).sum(axis=0)
    if np.any(support < 100):
        missing = [task_key(*TASKS[i]) for i in np.flatnonzero(support < 100)]
        raise ValueError(f"shared renal tasks lack label support: {missing}")
    prepared = PreparedData(
        frame=frame,
        current=current,
        history_indices=indices,
        history_mask=history_mask,
        beliefs=beliefs,
        belief_name="renal_belief_state_v3",
        feature_columns=feature_columns,
    )
    return SharedRenalData(prepared, labels, current_targets)


def _global_split(
    data: SharedRenalData,
    *,
    seed: int,
    group_column: str,
    late_stage_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = data.prepared.frame
    eligible = np.isfinite(data.labels).any(axis=1)
    eligible_rows = np.flatnonzero(eligible)
    local_frame = frame.loc[eligible_rows].reset_index(drop=True)
    local_train, local_test = _split_rows(
        local_frame,
        seed=seed,
        group_column=group_column,
        late_stage_only=late_stage_only,
    )
    train = eligible_rows[local_train]
    test = eligible_rows[local_test]
    subjects = frame.loc[train, "subject_id"].astype(str).unique()
    rng = np.random.default_rng(seed + 101)
    rng.shuffle(subjects)
    count = max(1, int(round(0.20 * len(subjects))))
    calibration_subjects = set(subjects[:count])
    is_calibration = frame.loc[train, "subject_id"].astype(str).isin(
        calibration_subjects
    ).to_numpy()
    return train[~is_calibration], train[is_calibration], test


def _task_normalization(
    data: SharedRenalData,
    fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros(len(TASKS), dtype=np.float32)
    scales = np.ones(len(TASKS), dtype=np.float32)
    for task_index in range(len(TASKS)):
        values = data.labels[fit, task_index]
        values = values[np.isfinite(values)]
        centers[task_index] = np.median(values)
        scale = (np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349
        scales[task_index] = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
    return centers, scales


def _population_models(
    data: SharedRenalData,
    fit: np.ndarray,
) -> list[object]:
    models = []
    for task_index in range(len(TASKS)):
        rows = fit[np.isfinite(data.labels[fit, task_index])]
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
        model.fit(data.prepared.current[rows], data.labels[rows, task_index])
        models.append(model)
    return models


def _task_matrices(
    data: SharedRenalData,
    rows: np.ndarray,
    population_models: list[object],
    centers: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(rows)
    anchors = np.zeros((count, len(TASKS)), dtype=np.float32)
    current_targets = np.zeros_like(anchors)
    truth = np.zeros_like(anchors)
    mask = np.isfinite(data.labels[rows]) & np.isfinite(data.current_targets[rows])
    physical_anchors = np.zeros((count, len(TASKS)), dtype=np.float64)
    for task_index, model in enumerate(population_models):
        physical_anchors[:, task_index] = model.predict(data.prepared.current[rows])
        anchors[:, task_index] = (
            physical_anchors[:, task_index] - centers[task_index]
        ) / scales[task_index]
        observed = data.current_targets[rows, task_index]
        values = data.labels[rows, task_index]
        current_targets[:, task_index] = np.where(
            np.isfinite(observed), (observed - centers[task_index]) / scales[task_index], 0.0
        )
        truth[:, task_index] = np.where(
            np.isfinite(values), (values - centers[task_index]) / scales[task_index], 0.0
        )
    return anchors, current_targets, truth, mask, physical_anchors


def _train_shared(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    anchors: np.ndarray,
    current_targets: np.ndarray,
    truth: np.ndarray,
    task_mask: np.ndarray,
    *,
    seed: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    variant: str,
    disable_shared: bool = False,
) -> nn.Module:
    torch.manual_seed(seed)
    current, history, history_mask, belief = arrays
    model_classes = {
        "pure_shared": SharedRenalPatientStateForecaster,
        "gated_shared_private": GatedSharedPrivateRenalForecaster,
    }
    model = model_classes[variant](
        feature_dim=current.shape[1], belief_dim=belief.shape[1],
        task_count=len(TASKS), hidden=hidden,
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(current),
        torch.from_numpy(history),
        torch.from_numpy(history_mask),
        torch.from_numpy(belief),
        torch.from_numpy(anchors),
        torch.from_numpy(current_targets),
        torch.from_numpy(truth),
        torch.from_numpy(task_mask),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    model.train()
    for _ in range(epochs):
        for batch in loader:
            current_b, history_b, history_mask_b, belief_b, anchor_b, current_b_t, truth_b, task_mask_b = [
                value.to(device) for value in batch
            ]
            model_args = (
                current_b, history_b, history_mask_b, belief_b, anchor_b, current_b_t
            )
            if variant == "gated_shared_private":
                point, scale = model(*model_args, disable_shared=disable_shared)
            else:
                point, scale = model(*model_args)
            observed = task_mask_b.to(point.dtype)
            count = observed.sum(dim=0).clamp_min(1.0)
            point_terms = nn.functional.smooth_l1_loss(
                point, truth_b, reduction="none"
            )
            point_loss = ((point_terms * observed).sum(dim=0) / count).mean()
            error = (point - truth_b).detach().abs()
            scale_terms = error / scale + torch.log(scale)
            scale_loss = ((scale_terms * observed).sum(dim=0) / count).mean()
            loss = point_loss + 0.10 * scale_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


@torch.no_grad()
def _predict_shared(
    model: nn.Module,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    anchors: np.ndarray,
    current_targets: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    variant: str,
    disable_shared: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    current, history, history_mask, belief = arrays
    dataset = TensorDataset(
        torch.from_numpy(current),
        torch.from_numpy(history),
        torch.from_numpy(history_mask),
        torch.from_numpy(belief),
        torch.from_numpy(anchors),
        torch.from_numpy(current_targets),
    )
    points, scales = [], []
    model.eval()
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        values = [value.to(device) for value in batch]
        if variant == "gated_shared_private":
            point, scale = model(*values, disable_shared=disable_shared)
        else:
            point, scale = model(*values)
        points.append(point.cpu().numpy())
        scales.append(scale.cpu().numpy())
    return np.concatenate(points), np.concatenate(scales)


def _run_split(
    data: SharedRenalData,
    *,
    seed: int,
    group_column: str,
    late_stage_only: bool,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    bootstrap_samples: int,
    device: torch.device,
    independent_ablation: bool,
    variant: str,
    calibrate_coupling_gain: bool,
) -> dict[str, object]:
    prepared = data.prepared
    frame = prepared.frame
    fit, calibration, test = _global_split(
        data,
        seed=seed,
        group_column=group_column,
        late_stage_only=late_stage_only,
    )
    current_center, current_scale = _fit_scaler(prepared.current, fit)
    belief_center, belief_scale = _fit_scaler(prepared.beliefs, fit)
    population_models = _population_models(data, fit)
    centers, target_scales = _task_normalization(data, fit)
    fit_task = _task_matrices(data, fit, population_models, centers, target_scales)
    cal_task = _task_matrices(data, calibration, population_models, centers, target_scales)
    test_task = _task_matrices(data, test, population_models, centers, target_scales)
    fit_arrays = _model_arrays(
        prepared, fit, current_center, current_scale, belief_center, belief_scale
    )
    cal_arrays = _model_arrays(
        prepared, calibration, current_center, current_scale, belief_center, belief_scale
    )
    test_arrays = _model_arrays(
        prepared, test, current_center, current_scale, belief_center, belief_scale
    )
    candidate = _train_shared(
        fit_arrays,
        fit_task[0],
        fit_task[1],
        fit_task[2],
        fit_task[3],
        seed=seed,
        hidden=hidden,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        variant=variant,
    )
    mismatched_fit = _mismatched_patient_rows(frame, fit, seed + 201)
    mismatched_fit_raw = _model_arrays(
        prepared,
        mismatched_fit,
        current_center,
        current_scale,
        belief_center,
        belief_scale,
    )
    placebo_arrays = (fit_arrays[0],) + mismatched_fit_raw[1:]
    placebo = _train_shared(
        placebo_arrays,
        fit_task[0],
        fit_task[1],
        fit_task[2],
        fit_task[3],
        seed=seed,
        hidden=hidden,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        variant=variant,
    )
    candidate_calibration, scale_calibration = _predict_shared(
        candidate,
        cal_arrays,
        cal_task[0],
        cal_task[1],
        device=device,
        batch_size=batch_size,
        variant=variant,
    )
    candidate_test, scale_test = _predict_shared(
        candidate,
        test_arrays,
        test_task[0],
        test_task[1],
        device=device,
        batch_size=batch_size,
        variant=variant,
    )
    mismatched_calibration = _mismatched_patient_rows(frame, calibration, seed + 302)
    mismatch_cal_raw = _model_arrays(
        prepared,
        mismatched_calibration,
        current_center,
        current_scale,
        belief_center,
        belief_scale,
    )
    mismatch_test = _mismatched_patient_rows(frame, test, seed + 303)
    mismatch_test_raw = _model_arrays(
        prepared,
        mismatch_test,
        current_center,
        current_scale,
        belief_center,
        belief_scale,
    )
    placebo_calibration, _ = _predict_shared(
        placebo,
        (cal_arrays[0],) + mismatch_cal_raw[1:],
        cal_task[0],
        cal_task[1],
        device=device,
        batch_size=batch_size,
        variant=variant,
    )
    placebo_test, _ = _predict_shared(
        placebo,
        (test_arrays[0],) + mismatch_test_raw[1:],
        test_task[0],
        test_task[1],
        device=device,
        batch_size=batch_size,
        variant=variant,
    )

    independent_predictions: dict[int, np.ndarray] = {}
    independent_calibration_predictions: dict[int, np.ndarray] = {}
    independent_scales: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if independent_ablation and variant == "gated_shared_private":
        no_coupling = _train_shared(
            fit_arrays,
            fit_task[0],
            fit_task[1],
            fit_task[2],
            fit_task[3],
            seed=seed,
            hidden=hidden,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            variant=variant,
            disable_shared=True,
        )
        no_coupling_test, _ = _predict_shared(
            no_coupling,
            test_arrays,
            test_task[0],
            test_task[1],
            device=device,
            batch_size=batch_size,
            variant=variant,
            disable_shared=True,
        )
        no_coupling_calibration, no_coupling_calibration_scale = _predict_shared(
            no_coupling,
            cal_arrays,
            cal_task[0],
            cal_task[1],
            device=device,
            batch_size=batch_size,
            variant=variant,
            disable_shared=True,
        )
        _, no_coupling_test_scale = _predict_shared(
            no_coupling,
            test_arrays,
            test_task[0],
            test_task[1],
            device=device,
            batch_size=batch_size,
            variant=variant,
            disable_shared=True,
        )
        for task_index in range(len(TASKS)):
            cal_local = np.flatnonzero(cal_task[3][:, task_index])
            test_local = np.flatnonzero(test_task[3][:, task_index])
            independent_predictions[task_index] = no_coupling_test[
                test_local, task_index
            ]
            independent_calibration_predictions[task_index] = no_coupling_calibration[
                cal_local, task_index
            ]
            independent_scales[task_index] = (
                no_coupling_calibration_scale[cal_local, task_index],
                no_coupling_test_scale[test_local, task_index],
            )
    elif independent_ablation:
        for task_index in range(len(TASKS)):
            fit_local = np.flatnonzero(fit_task[3][:, task_index])
            cal_local = np.flatnonzero(cal_task[3][:, task_index])
            test_local = np.flatnonzero(test_task[3][:, task_index])
            independent = _train_model(
                tuple(array[fit_local] for array in fit_arrays),
                fit_task[0][fit_local, task_index],
                fit_task[1][fit_local, task_index],
                fit_task[2][fit_local, task_index],
                seed=seed,
                hidden=hidden,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
            )
            prediction, _ = _predict(
                independent,
                tuple(array[test_local] for array in test_arrays),
                test_task[0][test_local, task_index],
                test_task[1][test_local, task_index],
                device,
                batch_size,
            )
            independent_predictions[task_index] = prediction
            calibration_prediction, calibration_independent_scale = _predict(
                independent,
                tuple(array[cal_local] for array in cal_arrays),
                cal_task[0][cal_local, task_index],
                cal_task[1][cal_local, task_index],
                device,
                batch_size,
            )
            _, test_independent_scale = _predict(
                independent,
                tuple(array[test_local] for array in test_arrays),
                test_task[0][test_local, task_index],
                test_task[1][test_local, task_index],
                device,
                batch_size,
            )
            independent_calibration_predictions[task_index] = calibration_prediction
            independent_scales[task_index] = (
                calibration_independent_scale,
                test_independent_scale,
            )

    results: dict[str, object] = {}
    for task_index, (target, horizon) in enumerate(TASKS):
        cal_local = np.flatnonzero(cal_task[3][:, task_index])
        test_local = np.flatnonzero(test_task[3][:, task_index])
        truth_cal = data.labels[calibration[cal_local], task_index]
        truth_test = data.labels[test[test_local], task_index]
        population_cal = cal_task[4][cal_local, task_index]
        population_test = test_task[4][test_local, task_index]
        shared_candidate_cal = (
            candidate_calibration[cal_local, task_index] * target_scales[task_index]
            + centers[task_index]
        )
        shared_candidate_eval = (
            candidate_test[test_local, task_index] * target_scales[task_index]
            + centers[task_index]
        )
        shared_placebo_cal = (
            placebo_calibration[cal_local, task_index] * target_scales[task_index]
            + centers[task_index]
        )
        shared_placebo_eval = (
            placebo_test[test_local, task_index] * target_scales[task_index]
            + centers[task_index]
        )
        shared_scale_cal = np.maximum(
            scale_calibration[cal_local, task_index] * target_scales[task_index], 1e-6
        )
        shared_scale_test = np.maximum(
            scale_test[test_local, task_index] * target_scales[task_index], 1e-6
        )
        subjects_cal = frame.loc[calibration[cal_local], "subject_id"].astype(str).to_numpy()
        subjects_test = frame.loc[test[test_local], "subject_id"].astype(str).to_numpy()
        coupling_gain = 1.0
        placebo_gain = 1.0
        independent_eval = None
        independent_cal = None
        independent_scale_cal = None
        independent_scale_test = None
        if independent_ablation:
            independent_cal = (
                independent_calibration_predictions[task_index]
                * target_scales[task_index]
                + centers[task_index]
            )
            independent_eval = (
                independent_predictions[task_index] * target_scales[task_index]
                + centers[task_index]
            )
            independent_scale_cal, independent_scale_test = independent_scales[task_index]
            independent_scale_cal = np.maximum(
                independent_scale_cal * target_scales[task_index], 1e-6
            )
            independent_scale_test = np.maximum(
                independent_scale_test * target_scales[task_index], 1e-6
            )
            if calibrate_coupling_gain:
                coupling_gain = _select_residual_gain(
                    subjects_cal,
                    truth_cal,
                    independent_cal,
                    shared_candidate_cal,
                )
                placebo_gain = _select_residual_gain(
                    subjects_cal,
                    truth_cal,
                    independent_cal,
                    shared_placebo_cal,
                )
            candidate_cal = independent_cal + coupling_gain * (
                shared_candidate_cal - independent_cal
            )
            candidate_eval = independent_eval + coupling_gain * (
                shared_candidate_eval - independent_eval
            )
            placebo_eval = independent_eval + placebo_gain * (
                shared_placebo_eval - independent_eval
            )
            candidate_scale_cal = (
                (1.0 - coupling_gain) * independent_scale_cal
                + coupling_gain * shared_scale_cal
            )
            candidate_scale_test = (
                (1.0 - coupling_gain) * independent_scale_test
                + coupling_gain * shared_scale_test
            )
        else:
            candidate_cal = shared_candidate_cal
            candidate_eval = shared_candidate_eval
            placebo_eval = shared_placebo_eval
            candidate_scale_cal = shared_scale_cal
            candidate_scale_test = shared_scale_test
        scores = np.abs(candidate_cal - truth_cal) / candidate_scale_cal
        calibration_regimes = _temporal_regimes(
            frame.loc[calibration[cal_local], "hours_since_onset"]
        )
        test_regimes = _temporal_regimes(
            frame.loc[test[test_local], "hours_since_onset"]
        )
        test_time = pd.to_datetime(frame.loc[test[test_local], "t"])
        future_time = test_time + pd.to_timedelta(horizon, unit="h")
        online = mondrian_online_adaptive_interval_widths(
            scores,
            subjects_cal,
            calibration_regimes,
            subjects_test,
            test_regimes,
            test_time,
            future_time,
            candidate_scale_test,
            candidate_eval,
            truth_test,
            coverage=0.90,
        )
        population_scores = np.abs(population_cal - truth_cal)
        population_online = mondrian_online_adaptive_interval_widths(
            population_scores,
            subjects_cal,
            calibration_regimes,
            subjects_test,
            test_regimes,
            test_time,
            future_time,
            np.ones(len(test_local), dtype=np.float64),
            population_test,
            truth_test,
            coverage=0.90,
        )
        covered = np.abs(candidate_eval - truth_test) <= online.widths
        population_covered = np.abs(population_test - truth_test) <= population_online.widths
        width = _width_comparison(
            subjects_test,
            online.widths,
            population_online.widths,
            seed=seed + 607 + task_index,
            bootstrap_samples=bootstrap_samples,
        )
        width["both_coverage_gates_pass"] = bool(
            0.87 <= covered.mean() <= 0.93
            and 0.87 <= population_covered.mean() <= 0.93
        )
        width["narrower_interval_claim_allowed"] = bool(
            _interval_narrowing_claim_allowed(
                width,
                candidate_coverage=float(covered.mean()),
                population_coverage=float(population_covered.mean()),
            )
        )
        coupled_interval = None
        coupled_interval_pass = not independent_ablation
        if independent_ablation:
            independent_scores = np.abs(independent_cal - truth_cal) / independent_scale_cal
            independent_online = mondrian_online_adaptive_interval_widths(
                independent_scores,
                subjects_cal,
                calibration_regimes,
                subjects_test,
                test_regimes,
                test_time,
                future_time,
                independent_scale_test,
                independent_eval,
                truth_test,
                coverage=0.90,
            )
            independent_covered = (
                np.abs(independent_eval - truth_test) <= independent_online.widths
            )
            coupled_interval = _width_comparison(
                subjects_test,
                online.widths,
                independent_online.widths,
                seed=seed + 809 + task_index,
                bootstrap_samples=bootstrap_samples,
            )
            coupled_interval["candidate_coverage"] = round(float(covered.mean()), 6)
            coupled_interval["independent_coverage"] = round(
                float(independent_covered.mean()), 6
            )
            coupled_interval["both_coverage_gates_pass"] = bool(
                0.87 <= covered.mean() <= 0.93
                and 0.87 <= independent_covered.mean() <= 0.93
            )
            coupled_interval["narrower_interval_claim_allowed"] = bool(
                _interval_narrowing_claim_allowed(
                    coupled_interval,
                    candidate_coverage=float(covered.mean()),
                    population_coverage=float(independent_covered.mean()),
                )
            )
            coupled_interval_pass = bool(
                coupled_interval["narrower_interval_claim_allowed"]
            )
        vs_population = _comparison(
            subjects_test,
            truth_test,
            candidate_eval,
            population_test,
            seed + 401 + task_index,
            bootstrap_samples,
        )
        vs_placebo = _comparison(
            subjects_test,
            truth_test,
            candidate_eval,
            placebo_eval,
            seed + 503 + task_index,
            bootstrap_samples,
        )
        vs_independent = None
        coupling_pass = not independent_ablation
        if independent_ablation:
            vs_independent = _comparison(
                subjects_test,
                truth_test,
                candidate_eval,
                independent_eval,
                seed + 701 + task_index,
                bootstrap_samples,
            )
            coupling_pass = bool(
                coupling_gain > 0.0 and vs_independent["significant"]
            )
        results[task_key(target, horizon)] = {
            "target": target,
            "horizon_hours": horizon,
            "rows": int(len(test_local)),
            "subjects": int(len(np.unique(subjects_test))),
            "mae": {
                "population": round(float(np.mean(np.abs(population_test - truth_test))), 6),
                "shared_renal": round(float(np.mean(np.abs(candidate_eval - truth_test))), 6),
                "placebo": round(float(np.mean(np.abs(placebo_eval - truth_test))), 6),
            },
            "candidate_vs_population": vs_population,
            "candidate_vs_placebo": vs_placebo,
            "candidate_vs_independent": vs_independent,
            "coupling_gain": round(float(coupling_gain), 6),
            "placebo_coupling_gain": round(float(placebo_gain), 6),
            "passes_coupling_gate": coupling_pass,
            "passes_coupled_interval_gate": coupled_interval_pass,
            "passes_point_gate": bool(
                vs_population["significant"]
                and vs_placebo["significant"]
                and coupling_pass
            ),
            "adaptive_conformal": {
                "coverage": round(float(covered.mean()), 6),
                "mean_half_width": round(float(np.mean(online.widths)), 6),
                "passes_0_87_to_0_93": bool(0.87 <= covered.mean() <= 0.93),
            },
            "matched_population_conformal": {
                "coverage": round(float(population_covered.mean()), 6),
                "mean_half_width": round(float(np.mean(population_online.widths)), 6),
                "median_half_width": round(float(np.median(population_online.widths)), 6),
                "passes_0_87_to_0_93": bool(
                    0.87 <= population_covered.mean() <= 0.93
                ),
            },
            "matched_interval_width": width,
            "matched_independent_interval_width": coupled_interval,
        }
    output = {
        "seed": seed,
        "group_column": group_column,
        "late_stage_only": late_stage_only,
        "shared_parameter_count": int(sum(p.numel() for p in candidate.parameters())),
        "placebo_parameter_count": int(sum(p.numel() for p in placebo.parameters())),
        "variant": variant,
        "tasks": results,
    }
    del candidate, placebo
    if independent_ablation and variant == "gated_shared_private":
        del no_coupling
    elif independent_ablation:
        del independent
    del fit_arrays, cal_arrays, test_arrays
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _task_summary(runs: list[dict[str, object]], key: str, expected: int) -> dict[str, object]:
    task_runs = [run["tasks"][key] for run in runs]
    return _summary(task_runs, expected) | {
        "coupling_gate_passes": int(sum(run["passes_coupling_gate"] for run in task_runs)),
        "all_coupling_gates_pass": bool(
            len(task_runs) == expected and all(run["passes_coupling_gate"] for run in task_runs)
        ),
        "coupled_interval_gate_passes": int(
            sum(run["passes_coupled_interval_gate"] for run in task_runs)
        ),
        "all_coupled_interval_gates_pass": bool(
            len(task_runs) == expected
            and all(run["passes_coupled_interval_gate"] for run in task_runs)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("renal_shared_patient_state_audit.json"))
    parser.add_argument("--seeds", default="7,11,19,23,37,53,71")
    parser.add_argument("--history-steps", type=int, default=6)
    parser.add_argument("--max-rows-per-subject", type=int, default=8)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--variant",
        choices=("pure_shared", "gated_shared_private"),
        default="gated_shared_private",
    )
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-patient", action="store_true")
    parser.add_argument(
        "--external-gate",
        choices=("all", "hospital", "care_unit", "time"),
        default="all",
        help=(
            "Run one external gate as a fail-closed partial audit, or all three. "
            "A partial audit can never promote a cell."
        ),
    )
    parser.add_argument("--skip-independent-ablation", action="store_true")
    parser.add_argument("--calibrate-coupling-gain", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    data = prepare_shared_data(
        args.root.resolve(),
        history_steps=args.history_steps,
        max_rows_per_subject=args.max_rows_per_subject,
        max_subjects=args.max_subjects,
        seed=17,
    )
    common = {
        "data": data,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "bootstrap_samples": args.bootstrap_samples,
        "device": _device(args.device),
        "independent_ablation": not args.skip_independent_ablation,
        "variant": args.variant,
        "calibrate_coupling_gain": args.calibrate_coupling_gain,
    }
    patient_runs = []
    if not args.skip_patient:
        patient_runs = [
            _run_split(
                seed=seed,
                group_column="subject_id",
                late_stage_only=False,
                **common,
            )
            for seed in seeds
        ]
    external_runs = []
    if not args.skip_external:
        external_specs = {
            "hospital": (9001, "hospitalid", False),
            "care_unit": (9002, "unittype", False),
            "time": (9003, "subject_id", True),
        }
        requested_gates = (
            tuple(external_specs)
            if args.external_gate == "all"
            else (args.external_gate,)
        )
        external_runs = [
            _run_split(
                seed=external_specs[gate][0],
                group_column=external_specs[gate][1],
                late_stage_only=external_specs[gate][2],
                **common,
            )
            for gate in requested_gates
        ]
    cells = {}
    for target, horizon in TASKS:
        key = task_key(target, horizon)
        patient = (
            _task_summary(patient_runs, key, len(seeds)) if patient_runs else None
        )
        external = _task_summary(external_runs, key, 3) if external_runs else None
        promoted = bool(
            patient is not None
            and patient["all_point_gates_pass"]
            and patient["all_conformal_gates_pass"]
            and patient["all_matched_interval_narrowing_gates_pass"]
            and patient["all_coupling_gates_pass"]
            and patient["all_coupled_interval_gates_pass"]
            and external is not None
            and external["all_point_gates_pass"]
            and external["all_conformal_gates_pass"]
            and external["all_matched_interval_narrowing_gates_pass"]
            and external["all_coupling_gates_pass"]
            and external["all_coupled_interval_gates_pass"]
        )
        cells[key] = {
            "patient_heldout": patient,
            "external_heldout": external,
            "coupled_precision_promoted": promoted,
            "external_full_interval_width_reduction": (
                round(
                    -2.0
                    * np.median(
                        [
                            run["tasks"][key]["matched_independent_interval_width"][
                                "patient_equalized_delta_half_width"
                            ]
                            for run in external_runs
                        ]
                    ),
                    6,
                )
                if promoted else None
            ),
        }
    report = {
        "schema": "shared_renal_patient_state_audit.v1",
        "architecture": {
            "variant": args.variant,
            "shared": "one causal kidney-organ patient state",
            "heads": [task_key(*task) for task in TASKS],
            "matched_ablation": (
                "same shared-private model with shared gate disabled"
                if args.variant == "gated_shared_private"
                else "independent patient-state encoder per task"
            ),
            "coupling_gain_policy": (
                "calibration-patient selected shrinkage toward shared residual"
                if args.calibrate_coupling_gain
                else "full shared residual"
            ),
            "capacity_matched_placebo": "shared model with mismatched patient history and belief",
        },
        "cohort": {
            "rows": int(len(data.prepared.frame)),
            "subjects": int(data.prepared.frame["subject_id"].nunique()),
            "hospitals": int(data.prepared.frame["hospitalid"].nunique()),
            "care_units": int(data.prepared.frame["unittype"].nunique()),
            "task_support": {
                task_key(*task): int(np.isfinite(data.labels[:, index]).sum())
                for index, task in enumerate(TASKS)
            },
        },
        "cells": cells,
        "coupled_precision_promoted_cells": [
            key for key, value in cells.items() if value["coupled_precision_promoted"]
        ],
        "patient_runs": patient_runs,
        "external_runs": external_runs,
        "external_gate_scope": args.external_gate,
        "boundary": {
            "factual_only": True,
            "causal_claim_allowed": False,
            "existing_independent_artifacts_remain_active_until_cell_promotion": True,
        },
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "coupled_precision_promoted_cells": report["coupled_precision_promoted_cells"],
        "task_support": report["cohort"]["task_support"],
    }, indent=2))


if __name__ == "__main__":
    main()
