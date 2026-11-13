"""External promotion gates for the persistent whole-body physiology state.

The development database is used for every fitted quantity: normalization,
base JEPA, persistent candidate, and the 311-feature cell expert.  The lockbox
database is used only for frozen evaluation and a disjoint conformal
calibration/test split.  This script never modifies the active runtime.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.validate_persistent_physiology_state import (
    MODULES,
    _base_feature_cache,
    _bootstrap_delta,
    _fit_cell_experts,
    _model_call,
    _predict_values,
)
from whole_body_joint_jepa import fit_joint, load_joint_event_examples


TRAINING_SEED = 20260829
DEVELOPMENT_RESIDUAL_GATE_FRACTION = 0.20
RENAL_TREATMENT_CLASSES = ("fluids", "diuretics", "vasopressor", "nephrotoxin")
TARGET_DELTA_TOLERANCE_HOURS = {
    "heart_rate": 0.5,
    "map": 0.5,
    "urine_output": 0.5,
    "urine_output_per_kg_hour": 0.5,
    "glucose": 1.0,
    "bicarbonate": 1.0,
    "potassium": 1.0,
}
PHYSIOLOGICAL_ACTIVE_RANGES = {
    "heart_rate": (60.0, 100.0),
    "bicarbonate": (22.0, 29.0),
    "glucose": (70.0, 180.0),
}


def _treatment_classes_for_target(target):
    if target == "bicarbonate":
        return ("bicarbonate", "fluids", "ventilation")
    if target in {"heart_rate", "map"}:
        return ("fluids", "vasopressor", "ventilation")
    if target == "glucose":
        return ("dextrose", "fluids", "insulin")
    return RENAL_TREATMENT_CLASSES


def _prefix_provenance(frame, prefix):
    frame = frame.copy()
    for column in ("subject_id", "stay_id", "encounter_id", "patient_id"):
        if column in frame:
            frame[column] = prefix + ":" + frame[column].astype(str)
    return frame


def _target_delta_bounds(target, horizon_hours):
    tolerance = TARGET_DELTA_TOLERANCE_HOURS.get(
        str(target), max(0.5, 0.20 * float(horizon_hours))
    )
    return (
        max(0.05, float(horizon_hours) - float(tolerance)),
        float(horizon_hours) + float(tolerance),
    )


def _apply_target_delta_contract(frame, target, bounds=None, horizon_hours=3.0):
    """Remove nominal-horizon labels outside the target's actual-time contract."""

    import pandas as pd

    frame = frame.copy()
    if bounds is None:
        bounds = _target_delta_bounds(target, horizon_hours)
    delta_column = f"future_delta_t_hr_{target}"
    future_column = f"future_{target}"
    if delta_column not in frame or future_column not in frame:
        raise ValueError(f"missing future contract columns for {target}")
    delta = pd.to_numeric(frame[delta_column], errors="coerce")
    future_present = frame[future_column].notna()
    weight_column = f"future_label_weight_{target}"
    if weight_column in frame:
        effective_before = future_present & (
            pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0) > 0.0
        )
    else:
        effective_before = future_present
    in_window = delta.between(bounds[0], bounds[1], inclusive="both")
    eligible = effective_before & in_window
    rejected = effective_before & ~in_window
    invalid_future = future_present & ~in_window
    frame.loc[invalid_future, future_column] = np.nan
    if weight_column in frame:
        frame.loc[invalid_future, weight_column] = 0.0
    return frame, {
        "target": target,
        "actual_delta_hours": [float(bounds[0]), float(bounds[1])],
        "target_tolerance_hours": float(
            max(float(horizon_hours) - bounds[0], bounds[1] - float(horizon_hours))
        ),
        "labels_before": int(effective_before.sum()),
        "labels_retained": int(eligible.sum()),
        "labels_rejected": int(rejected.sum()),
        "future_values_already_zero_weight": int(
            (future_present & ~effective_before).sum()
        ),
    }


def _configure_binary_treatment_context(frame, treatment_classes):
    """Use cross-database treatment presence without chart-frequency counts."""

    import pandas as pd

    frame = frame.copy()
    columns = []
    support = {}
    for treatment in treatment_classes:
        source = f"hist_{treatment}_evidence_count"
        if source not in frame:
            raise ValueError(f"missing canonical treatment field: {source}")
        target = f"hist_{treatment}_present"
        frame[target] = (
            pd.to_numeric(frame[source], errors="coerce").fillna(0.0) > 0.0
        ).astype(np.float32)
        columns.append(target)
        support[treatment] = float(frame[target].mean())
    frame.attrs["treatment_history_columns"] = columns
    return frame, support


def _combine_development_lockbox(development, lockbox):
    """Combine frames while preserving database-local precomputed histories."""

    development = _prefix_provenance(development, "development")
    lockbox = _prefix_provenance(lockbox, "lockbox")
    offset = len(development)
    history_columns = sorted(
        column for column in development if column.startswith("_history_")
    )
    if history_columns != sorted(
        column for column in lockbox if column.startswith("_history_")
    ):
        raise ValueError("development and lockbox history contracts differ")
    for column in history_columns:
        lockbox[column] = lockbox[column].astype(np.int64) + offset
    import pandas as pd

    combined = pd.concat((development, lockbox), ignore_index=True, sort=False)
    combined.attrs.update(copy.deepcopy(development.attrs))
    return combined, np.arange(len(development)), np.arange(offset, len(combined))


def _split_development_residual_gate(frame, development_rows, seed=TRAINING_SEED):
    """Reserve patients that cannot influence model fitting or residual training."""

    rows = np.asarray(development_rows, dtype=np.int64)
    patients = np.asarray(
        sorted(frame.iloc[rows]["subject_id"].astype(str).unique())
    )
    if len(patients) < 10:
        raise ValueError("development cohort is too small for a residual gate")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    gate_count = max(1, int(round(len(patients) * DEVELOPMENT_RESIDUAL_GATE_FRACTION)))
    gate_patients = set(patients[:gate_count])
    row_patients = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    is_gate = np.asarray(
        [patient in gate_patients for patient in row_patients], dtype=bool
    )
    fit_rows = rows[~is_gate]
    gate_rows = rows[is_gate]
    if not len(fit_rows) or not len(gate_rows):
        raise ValueError("development residual split produced an empty partition")
    return fit_rows, gate_rows


def _patient_errors(frame, rows, truth, predictions):
    patients = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    output = []
    for patient in np.unique(patients):
        selected = patients == patient
        output.append(
            {
                "patient": patient,
                **{
                    name: float(np.mean(np.abs(values[selected] - truth[selected])))
                    for name, values in predictions.items()
                },
            }
        )
    return output


def _comparison(patient_errors, candidate, reference, seed=TRAINING_SEED):
    renamed = [
        {
            "patient": row["patient"],
            "persistent": row[candidate],
            "reference": row[reference],
        }
        for row in patient_errors
    ]
    return _bootstrap_delta(renamed, "reference", seed=seed)


def _development_residual_gate(
    frame,
    candidate,
    base,
    arrays,
    base_arrays,
    rows,
    target_index,
    horizon_hours,
    target,
    owner_only=False,
    physiological_regime=False,
):
    """Decide on untouched development patients whether a residual may activate."""

    valid_rows = _valid_cell_rows(arrays, rows, target_index, horizon_hours)
    patient_count = int(frame.iloc[valid_rows]["subject_id"].nunique())
    if len(valid_rows) < 100 or patient_count < 20:
        return {
            "rows": int(len(valid_rows)),
            "patients": patient_count,
            "minimum_rows": 100,
            "minimum_patients": 20,
            "enabled": False,
            "reason": "insufficient untouched development support",
        }

    candidate_view = (
        _owner_only_arrays(candidate, arrays, valid_rows, target_index)
        if owner_only
        else arrays
    )
    base_view = (
        _owner_only_arrays(base, base_arrays, valid_rows, target_index)
        if owner_only
        else base_arrays
    )
    candidate_prediction = _predict_values(
        candidate, candidate_view, valid_rows
    )[0][:, target_index]
    base_prediction = _predict_values(base, base_view, valid_rows)[0][:, target_index]
    truth = arrays[5][valid_rows, target_index]
    current = arrays[2][valid_rows, target_index]
    active = None
    if physiological_regime:
        active = _active_regime(frame, valid_rows, target)
        candidate_prediction = np.where(active, candidate_prediction, current)
        base_prediction = np.where(active, base_prediction, current)

    errors = _patient_errors(
        frame,
        valid_rows,
        truth,
        {"candidate": candidate_prediction, "base": base_prediction},
    )
    comparison = _comparison(errors, "candidate", "base")
    candidate_mae = float(np.mean([row["candidate"] for row in errors]))
    base_mae = float(np.mean([row["base"] for row in errors]))
    relative_gain = float((base_mae - candidate_mae) / max(base_mae, 1e-12))
    minimum_gain = 0.001
    enabled = bool(comparison["pass"] and relative_gain >= minimum_gain)
    return {
        "rows": int(len(valid_rows)),
        "patients": patient_count,
        "owner_organ_input_policy": bool(owner_only),
        "physiological_regime_gate": (
            {
                "enabled": True,
                "active_rows": int(active.sum()),
                "active_fraction": float(active.mean()),
            }
            if active is not None
            else {"enabled": False}
        ),
        "patient_equalized_mae": {
            "candidate": candidate_mae,
            "base": base_mae,
        },
        "relative_mae_gain": relative_gain,
        "minimum_relative_mae_gain": minimum_gain,
        "paired_patient_bootstrap": comparison,
        "enabled": enabled,
        "reason": (
            "target residual passed untouched development patients"
            if enabled
            else "target residual failed untouched development patients"
        ),
    }


def _finite_sample_quantile(values, coverage=0.90):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("conformal calibration has no finite residuals")
    level = min(1.0, np.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(values, level, method="higher"))


def _conformal_gate(frame, rows, truth, prediction, seed=TRAINING_SEED):
    patients = np.asarray(sorted(frame.iloc[rows]["subject_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    split = max(1, len(patients) // 2)
    calibration_patients = set(patients[:split])
    row_patients = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    calibration = np.asarray(
        [patient in calibration_patients for patient in row_patients], dtype=bool
    )
    test = ~calibration
    radius = _finite_sample_quantile(
        np.abs(prediction[calibration] - truth[calibration])
    )
    covered = np.abs(prediction[test] - truth[test]) <= radius
    test_patients = row_patients[test]
    patient_coverage = [
        float(np.mean(covered[test_patients == patient]))
        for patient in np.unique(test_patients)
    ]
    row_coverage = float(np.mean(covered))
    equalized = float(np.mean(patient_coverage))
    return {
        "calibration_patients": int(len(calibration_patients)),
        "test_patients": int(len(np.unique(test_patients))),
        "normalized_radius": radius,
        "row_coverage": row_coverage,
        "patient_equalized_coverage": equalized,
        "target_coverage_range": [0.87, 0.93],
        "pass": bool(0.87 <= row_coverage <= 0.93 and 0.87 <= equalized <= 0.93),
    }


def _masked_arrays(arrays, rows, target_index):
    masked = list(arrays)
    input_matrix = arrays[1].copy()
    current = arrays[2].copy()
    mask = arrays[3].copy()
    ages = arrays[4].copy()
    history = arrays[7]
    history_rows = np.unique(history[np.asarray(rows, dtype=np.int64)].reshape(-1))
    variable_count = current.shape[1]
    input_matrix[history_rows, target_index] = 0.0
    input_matrix[history_rows, variable_count + target_index] = 0.0
    input_matrix[history_rows, 2 * variable_count + target_index] = 7.0
    current[rows, target_index] = 0.0
    mask[rows, target_index] = 0.0
    ages[rows, target_index] = 7.0
    masked[1] = input_matrix
    masked[2] = current
    masked[3] = mask
    masked[4] = ages
    return tuple(masked)


def _predict_anchor_masked(model, arrays, rows, target_index, batch_size=1024):
    """Hide the anchor target but retain strictly earlier target history."""

    _, input_matrix, current, mask, ages, _, _, history, horizon, presence, *_ = arrays
    variable_count = current.shape[1]
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = np.asarray(rows[start : start + batch_size], dtype=np.int64)
            observations = input_matrix[history[batch]].copy()
            for position, source_row in enumerate(batch):
                anchor_slots = history[source_row] == source_row
                observations[position, anchor_slots, target_index] = 0.0
                observations[
                    position, anchor_slots, variable_count + target_index
                ] = 0.0
                observations[
                    position, anchor_slots, 2 * variable_count + target_index
                ] = 7.0
            batch_current = current[batch].copy()
            batch_mask = mask[batch].copy()
            batch_ages = ages[batch].copy()
            batch_current[:, target_index] = 0.0
            batch_mask[:, target_index] = 0.0
            batch_ages[:, target_index] = 7.0
            output = model(
                torch.from_numpy(observations).float(),
                torch.from_numpy(batch_current).float(),
                torch.from_numpy(batch_mask).float(),
                torch.from_numpy(batch_ages).float(),
                torch.from_numpy(presence[batch]).float(),
                torch.from_numpy(horizon[batch]).float(),
            )
            predictions.append(output["value"].cpu().numpy())
    return np.concatenate(predictions)


def _no_route_prediction(model, arrays, rows):
    adjacency = model.persistent_physiology.adjacency
    original = adjacency.detach().clone()
    try:
        adjacency.copy_(torch.eye(adjacency.shape[0], dtype=adjacency.dtype))
        return _predict_values(model, arrays, rows)[0]
    finally:
        adjacency.copy_(original)


def _owner_only_arrays(model, arrays, rows, target_index):
    """Remove every non-owner organ input from the complete causal history."""

    masked = list(arrays)
    input_matrix = arrays[1].copy()
    current = arrays[2].copy()
    mask = arrays[3].copy()
    ages = arrays[4].copy()
    history = arrays[7]
    presence = arrays[9].copy()
    selected = np.asarray(rows, dtype=np.int64)
    history_rows = np.unique(history[selected].reshape(-1))
    variable_count = current.shape[1]
    owner = int(model.variable_module_ids[target_index].item())
    membership = model.variable_module_membership[owner].detach().cpu().numpy() > 0.0
    removed_variables = np.flatnonzero(~membership)
    removed_organs = np.asarray(
        [index for index in range(presence.shape[1]) if index != owner],
        dtype=np.int64,
    )
    input_matrix[np.ix_(history_rows, removed_variables)] = 0.0
    input_matrix[np.ix_(history_rows, variable_count + removed_variables)] = 0.0
    input_matrix[np.ix_(history_rows, 2 * variable_count + removed_variables)] = 7.0
    presence_start = 3 * variable_count + 1
    if input_matrix.shape[1] >= presence_start + presence.shape[1]:
        input_matrix[np.ix_(history_rows, presence_start + removed_organs)] = 0.0
    current[np.ix_(selected, removed_variables)] = 0.0
    mask[np.ix_(selected, removed_variables)] = 0.0
    ages[np.ix_(selected, removed_variables)] = 7.0
    presence[np.ix_(selected, removed_organs)] = 0.0
    masked[1] = input_matrix
    masked[2] = current
    masked[3] = mask
    masked[4] = ages
    masked[9] = presence
    return tuple(masked)


def _trajectory_gate(model, arrays, rows, max_rows=256):
    batch = np.asarray(rows[:max_rows], dtype=np.int64)
    model.eval()
    with torch.no_grad():
        output = _model_call(model, arrays, batch)
        presence = torch.from_numpy(arrays[9][batch]).float()
        staged = model._predict_state_trajectory(
            output["context_latent"],
            output["organ_latents"],
            presence,
            torch.tensor([[1.0, 3.0]]).expand(len(batch), -1),
        )
        direct = model._predict_state_trajectory(
            output["context_latent"],
            output["organ_latents"],
            presence,
            torch.tensor([[3.0]]).expand(len(batch), -1),
        )
        body_difference = torch.max(
            torch.abs(staged["physiology_body"][:, -1] - direct["physiology_body"][:, 0])
        ).item()
        organ_difference = torch.max(
            torch.abs(staged["organs"][:, -1] - direct["organs"][:, 0])
        ).item()
        finite = all(
            bool(torch.isfinite(value).all())
            for value in (staged["physiology_body"], staged["organs"], staged["uncertainty"])
        )
    return {
        "rows": int(len(batch)),
        "staged_vs_direct_body_max_abs": float(body_difference),
        "staged_vs_direct_organ_max_abs": float(organ_difference),
        "all_states_finite": finite,
        "pass": bool(finite and body_difference <= 1e-6 and organ_difference <= 1e-6),
    }


def _valid_cell_rows(arrays, rows, target_index, horizon_hours):
    future_mask = arrays[6]
    horizon = arrays[8]
    selected = np.asarray(rows, dtype=np.int64)
    return selected[
        (future_mask[selected, target_index] > 0.0)
        & np.isclose(horizon[selected], float(horizon_hours))
    ]


def _active_regime(frame, rows, target):
    import pandas as pd

    if target not in PHYSIOLOGICAL_ACTIVE_RANGES:
        raise ValueError(f"no pre-specified active range for {target}")
    lower, upper = PHYSIOLOGICAL_ACTIVE_RANGES[target]
    value = pd.to_numeric(frame.iloc[rows][f"{target}_t"], errors="coerce").to_numpy()
    return np.isfinite(value) & ((value < lower) | (value > upper))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-examples", type=Path, required=True)
    parser.add_argument("--lockbox-examples", type=Path, required=True)
    parser.add_argument("--development-max-stays", type=int, default=200)
    parser.add_argument("--lockbox-max-stays", type=int, default=1000)
    parser.add_argument("--target", default="urine_output")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--nowcast-pretrain-epochs", type=int, default=0)
    parser.add_argument("--candidate-uncertainty-gate", action="store_true")
    parser.add_argument("--physiological-regime-gate", action="store_true")
    parser.add_argument("--owner-organ-target-stage", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    cell = (str(args.target), int(args.horizon))
    treatment_classes = _treatment_classes_for_target(cell[0])

    development, variables, module_names = load_joint_event_examples(
        args.development_examples,
        MODULES,
        horizons=(cell[1],),
        max_stays=args.development_max_stays,
        min_modules=2,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
    )
    lockbox, lockbox_variables, lockbox_modules = load_joint_event_examples(
        args.lockbox_examples,
        MODULES,
        horizons=(cell[1],),
        max_stays=args.lockbox_max_stays,
        min_modules=2,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
    )
    if tuple(variables) != tuple(lockbox_variables):
        raise ValueError("development and lockbox variable contracts differ")
    if tuple(module_names) != tuple(lockbox_modules):
        raise ValueError("development and lockbox organ contracts differ")
    development, development_delta_contract = _apply_target_delta_contract(
        development, cell[0], horizon_hours=cell[1]
    )
    lockbox, lockbox_delta_contract = _apply_target_delta_contract(
        lockbox, cell[0], horizon_hours=cell[1]
    )
    development, development_treatment_support = _configure_binary_treatment_context(
        development, treatment_classes
    )
    lockbox, lockbox_treatment_support = _configure_binary_treatment_context(
        lockbox, treatment_classes
    )
    frame, development_rows, lockbox_rows = _combine_development_lockbox(
        development, lockbox
    )
    train_rows, residual_gate_rows = _split_development_residual_gate(
        frame, development_rows
    )

    common = dict(
        seed=TRAINING_SEED,
        epochs=args.epochs,
        batch_size=1024,
        world_model=True,
        target_encoder_mode="fast",
        future_target_mode="window",
        regime_balanced_loss=True,
        state_normalization="robust",
        supervised_targets=(cell[0],),
        future_task_balanced_sampling=True,
        target_head_stage_epochs=1,
        owner_organ_target_stage=bool(args.owner_organ_target_stage),
        nowcast_pretrain_epochs=int(args.nowcast_pretrain_epochs),
        measurement_consistency_weight=(
            0.10 if int(args.nowcast_pretrain_epochs) > 0 else 0.0
        ),
        include_treatment_context=True,
    )
    base, base_arrays, base_losses = fit_joint(
        frame,
        variables,
        module_names,
        train_rows,
        persistent_physiology_state=False,
        **common,
    )
    candidate, arrays, candidate_losses = fit_joint(
        frame,
        variables,
        module_names,
        train_rows,
        persistent_physiology_state=True,
        initial_state_dict=base.state_dict(),
        coupling_preservation_weight=1.0,
        uncertainty_gate=bool(args.candidate_uncertainty_gate),
        **common,
    )
    target_index = variables.index(cell[0])
    development_residual_gate = _development_residual_gate(
        frame,
        candidate,
        base,
        arrays,
        base_arrays,
        residual_gate_rows,
        target_index,
        cell[1],
        cell[0],
        owner_only=bool(args.owner_organ_target_stage),
        physiological_regime=bool(args.physiological_regime_gate),
    )
    valid_rows = _valid_cell_rows(arrays, lockbox_rows, target_index, cell[1])
    if len(valid_rows) < 100:
        raise ValueError("lockbox has fewer than 100 valid urine_output@3h rows")

    base_train_prediction, train_cache = _base_feature_cache(
        base, base_arrays, train_rows, variables, (cell,)
    )
    del base_train_prediction
    base_prediction, test_cache = _base_feature_cache(
        base, base_arrays, valid_rows, variables, (cell,)
    )
    candidate_prediction, _ = _predict_values(candidate, arrays, valid_rows)
    expert_prediction, expert_metadata = _fit_cell_experts(
        train_cache, test_cache, base_arrays, variables, (cell,)
    )
    key = f"{cell[0]}@{cell[1]}h"
    expert_rows = test_cache[key]["rows"]
    if not np.array_equal(expert_rows, valid_rows):
        raise ValueError("expert cache rows do not match external cell rows")
    truth = arrays[5][valid_rows, target_index]
    current = arrays[2][valid_rows, target_index]
    unrestricted_candidate_prediction = candidate_prediction[:, target_index]
    unrestricted_base_prediction = base_prediction[:, target_index]
    owner_only_arrays = _owner_only_arrays(
        candidate, arrays, valid_rows, target_index
    )
    owner_only_prediction = _predict_values(
        candidate, owner_only_arrays, valid_rows
    )[0][:, target_index]
    owner_only_base_arrays = _owner_only_arrays(
        base, base_arrays, valid_rows, target_index
    )
    owner_only_base_prediction = _predict_values(
        base, owner_only_base_arrays, valid_rows
    )[0][:, target_index]
    active_regime = None
    if args.physiological_regime_gate:
        active_regime = _active_regime(frame, valid_rows, cell[0])
        unrestricted_candidate_prediction = np.where(
            active_regime, unrestricted_candidate_prediction, current
        )
        unrestricted_base_prediction = np.where(
            active_regime, unrestricted_base_prediction, current
        )
        owner_only_prediction = np.where(
            active_regime, owner_only_prediction, current
        )
        owner_only_base_prediction = np.where(
            active_regime, owner_only_base_prediction, current
        )
    raw_selected_candidate_prediction = (
        owner_only_prediction
        if args.owner_organ_target_stage
        else unrestricted_candidate_prediction
    )
    selected_base_prediction = (
        owner_only_base_prediction
        if args.owner_organ_target_stage
        else unrestricted_base_prediction
    )
    selected_candidate_prediction = (
        raw_selected_candidate_prediction
        if development_residual_gate["enabled"]
        else selected_base_prediction.copy()
    )
    point_predictions = {
        "candidate": selected_candidate_prediction,
        "base": selected_base_prediction,
        "expert": expert_prediction[key],
        "persistence": current,
    }
    point_errors = _patient_errors(frame, valid_rows, truth, point_predictions)
    point_comparisons = {
        reference: _comparison(point_errors, "candidate", reference)
        for reference in ("base", "expert", "persistence")
    }
    point_gate = {
        "rows": int(len(valid_rows)),
        "patients": int(frame.iloc[valid_rows]["subject_id"].nunique()),
        "patient_equalized_mae": {
            name: float(np.mean([row[name] for row in point_errors]))
            for name in point_predictions
        },
        "paired_patient_bootstrap": point_comparisons,
        "residual_policy": (
            "target residual enabled"
            if development_residual_gate["enabled"]
            else "exact owner-JEPA fallback"
        ),
        "raw_candidate_patient_equalized_mae": float(
            np.mean(
                [
                    row["raw_candidate"]
                    for row in _patient_errors(
                        frame,
                        valid_rows,
                        truth,
                        {"raw_candidate": raw_selected_candidate_prediction},
                    )
                ]
            )
        ),
        "physiological_regime_gate": (
            {
                "enabled": True,
                "normal_range": list(PHYSIOLOGICAL_ACTIVE_RANGES[cell[0]]),
                "active_rows": int(active_regime.sum()),
                "active_fraction": float(active_regime.mean()),
            }
            if active_regime is not None
            else {"enabled": False}
        ),
        "pass": all(value["pass"] for value in point_comparisons.values()),
    }

    conformal = _conformal_gate(
        frame, valid_rows, truth, point_predictions["candidate"]
    )

    evaluation_arrays = (
        owner_only_arrays if args.owner_organ_target_stage else arrays
    )
    base_evaluation_arrays = (
        owner_only_base_arrays if args.owner_organ_target_stage else base_arrays
    )
    no_route_full = _no_route_prediction(
        candidate, evaluation_arrays, valid_rows
    )
    no_route_predictions = {
        "candidate": point_predictions["candidate"],
        "no_route": no_route_full[:, target_index],
    }
    if active_regime is not None:
        no_route_predictions["no_route"] = np.where(
            active_regime, no_route_predictions["no_route"], current
        )
    no_route_errors = _patient_errors(
        frame, valid_rows, truth, no_route_predictions
    )
    cross_organ = {
        "candidate_patient_equalized_mae": float(
            np.mean([row["candidate"] for row in no_route_errors])
        ),
        "no_route_patient_equalized_mae": float(
            np.mean([row["no_route"] for row in no_route_errors])
        ),
        "paired_patient_bootstrap": _comparison(
            no_route_errors, "candidate", "no_route"
        ),
    }
    cross_organ["relative_mae_gain"] = float(
        (
            cross_organ["no_route_patient_equalized_mae"]
            - cross_organ["candidate_patient_equalized_mae"]
        )
        / max(cross_organ["no_route_patient_equalized_mae"], 1e-12)
    )
    cross_organ["minimum_relative_mae_gain"] = 0.001
    adjacency = candidate.persistent_physiology.adjacency.detach().cpu().numpy()
    graph_has_cross_organ_edges = bool(
        np.any(np.abs(adjacency - np.diag(np.diag(adjacency))) > 1e-8)
    )
    cross_organ["applicable"] = graph_has_cross_organ_edges
    cross_organ["reason"] = (
        "candidate contains cross-organ recurrent edges"
        if graph_has_cross_organ_edges
        else "cross-organ recurrent edges are fail-closed; no edge is under test"
    )
    cross_organ["pass"] = bool(
        not graph_has_cross_organ_edges
        or (
            cross_organ["paired_patient_bootstrap"]["pass"]
            and cross_organ["relative_mae_gain"] >= 0.001
        )
    )

    cross_input_predictions = {
        "candidate": unrestricted_candidate_prediction,
        "owner_only": owner_only_prediction,
    }
    cross_input_errors = _patient_errors(
        frame, valid_rows, truth, cross_input_predictions
    )
    cross_input_comparison = _comparison(
        cross_input_errors, "candidate", "owner_only"
    )
    candidate_cross_mae = float(
        np.mean([row["candidate"] for row in cross_input_errors])
    )
    owner_only_mae = float(
        np.mean([row["owner_only"] for row in cross_input_errors])
    )
    cross_organ_input = {
        "candidate_patient_equalized_mae": candidate_cross_mae,
        "owner_only_patient_equalized_mae": owner_only_mae,
        "paired_patient_bootstrap": cross_input_comparison,
        "relative_mae_gain": float(
            (owner_only_mae - candidate_cross_mae) / max(owner_only_mae, 1e-12)
        ),
        "minimum_relative_mae_gain": 0.001,
    }
    cross_organ_input["pass"] = bool(
        cross_input_comparison["pass"]
        and cross_organ_input["relative_mae_gain"] >= 0.001
    )

    owner_point_predictions = {
        "owner_only": owner_only_prediction,
        "candidate": point_predictions["candidate"],
        "base_owner_only": owner_only_base_prediction,
        "expert": point_predictions["expert"],
        "persistence": point_predictions["persistence"],
    }
    owner_point_errors = _patient_errors(
        frame, valid_rows, truth, owner_point_predictions
    )
    owner_point_comparisons = {
        reference: _comparison(owner_point_errors, "owner_only", reference)
        for reference in (
            "candidate",
            "base_owner_only",
            "expert",
            "persistence",
        )
    }
    owner_organ_point_gate = {
        "input_policy": (
            "complete target history plus variables assigned to the target's "
            "anatomical owner organ; every other organ is masked"
        ),
        "patient_equalized_mae": {
            name: float(np.mean([row[name] for row in owner_point_errors]))
            for name in owner_point_predictions
        },
        "paired_patient_bootstrap": owner_point_comparisons,
        "pass_vs_existing_references": all(
            owner_point_comparisons[reference]["pass"]
            for reference in ("base_owner_only", "expert", "persistence")
        ),
        "pass_vs_unrestricted_whole_body": owner_point_comparisons["candidate"][
            "pass"
        ],
    }
    owner_organ_point_gate["pass"] = bool(
        owner_organ_point_gate["pass_vs_existing_references"]
        and owner_organ_point_gate["pass_vs_unrestricted_whole_body"]
    )

    anchor_masked_candidate = _predict_anchor_masked(
        candidate, evaluation_arrays, valid_rows, target_index
    )[:, target_index]
    anchor_masked_base = _predict_anchor_masked(
        base, base_evaluation_arrays, valid_rows, target_index
    )[:, target_index]
    anchor_predictions = {
        "candidate": anchor_masked_candidate,
        "base": anchor_masked_base,
        "population_center": np.zeros_like(anchor_masked_candidate),
    }
    anchor_errors = _patient_errors(frame, valid_rows, truth, anchor_predictions)
    anchor_comparisons = {
        reference: _comparison(anchor_errors, "candidate", reference)
        for reference in ("base", "population_center")
    }
    anchor_masked_recovery = {
        "history_policy": "anchor target hidden; strictly earlier target history retained",
        "patient_equalized_mae": {
            name: float(np.mean([row[name] for row in anchor_errors]))
            for name in anchor_predictions
        },
        "paired_patient_bootstrap": anchor_comparisons,
        "pass": all(value["pass"] for value in anchor_comparisons.values()),
    }

    masked_candidate_arrays = _masked_arrays(
        evaluation_arrays, valid_rows, target_index
    )
    masked_base_arrays = _masked_arrays(
        base_evaluation_arrays, valid_rows, target_index
    )
    masked_candidate = _predict_values(
        candidate, masked_candidate_arrays, valid_rows
    )[0][:, target_index]
    masked_base = _predict_values(
        base, masked_base_arrays, valid_rows
    )[0][:, target_index]
    masked_predictions = {
        "candidate": masked_candidate,
        "base": masked_base,
        "population_center": np.zeros_like(masked_candidate),
    }
    masked_errors = _patient_errors(
        frame, valid_rows, truth, masked_predictions
    )
    masked_comparisons = {
        reference: _comparison(masked_errors, "candidate", reference)
        for reference in ("base", "population_center")
    }
    masked_recovery = {
        "history_policy": "target hidden from anchor and complete input history",
        "patient_equalized_mae": {
            name: float(np.mean([row[name] for row in masked_errors]))
            for name in masked_predictions
        },
        "paired_patient_bootstrap": masked_comparisons,
        "pass": all(value["pass"] for value in masked_comparisons.values()),
    }

    trajectory = _trajectory_gate(candidate, evaluation_arrays, valid_rows)
    if cell[0] in {"urine_output", "urine_output_per_kg_hour"}:
        conservation = {
            "applicable": True,
            "explicit_urine_mass_balance_available": False,
            "reason": (
                "The lockbox has output events but no complete intake/store "
                "ledger for an exact urine mass-balance equality."
            ),
            "pass": False,
        }
    else:
        conservation = {
            "applicable": False,
            "reason": (
                "No exact hidden-store conservation equality is defined for "
                f"the {cell[0]} point-forecast gate."
            ),
            "pass": True,
        }
    independent_untouched_lockbox = {
        "pass": False,
        "reason": (
            "MIMIC-IV exposed the preceding candidate's masked-recovery and "
            "no-route failures, so this corrected architecture requires a new "
            "untouched database lockbox."
        ),
        "required_next": "HiRID, AmsterdamUMCdb, or another frozen independent cohort",
    }
    full_promotion = bool(
        development_residual_gate["enabled"]
        and point_gate["pass"]
        and conformal["pass"]
        and cross_organ["pass"]
        and anchor_masked_recovery["pass"]
        and masked_recovery["pass"]
        and trajectory["pass"]
        and conservation["pass"]
        and independent_untouched_lockbox["pass"]
    )
    report = {
        "schema": "persistent_physiology_external_gate.v2",
        "cell": key,
        "candidate_only": not full_promotion,
        "training": {
            "development_database": "eICU",
            "lockbox_database": "MIMIC-IV",
            "development_rows": int(len(development)),
            "development_patients": int(development["subject_id"].nunique()),
            "fit_rows": int(len(train_rows)),
            "fit_patients": int(frame.iloc[train_rows]["subject_id"].nunique()),
            "residual_gate_rows": int(len(residual_gate_rows)),
            "residual_gate_patients": int(
                frame.iloc[residual_gate_rows]["subject_id"].nunique()
            ),
            "lockbox_rows": int(len(lockbox)),
            "lockbox_patients": int(lockbox["subject_id"].nunique()),
            "fixed_training_seed": TRAINING_SEED,
            "epochs": int(args.epochs),
            "nowcast_pretrain_epochs": int(args.nowcast_pretrain_epochs),
            "candidate_uncertainty_gate": bool(args.candidate_uncertainty_gate),
            "physiological_regime_gate": bool(args.physiological_regime_gate),
            "owner_organ_target_stage": bool(args.owner_organ_target_stage),
            "target_focused_coupling_weight": 1.0,
            "trained_nominal_horizons": [cell[1]],
            "development_label_contract": development_delta_contract,
            "validation_label_contract": lockbox_delta_contract,
            "treatment_context": {
                "fields": [
                    f"hist_{value}_present" for value in treatment_classes
                ],
                "representation": "anchor_pre_binary_presence",
                "development_support": development_treatment_support,
                "validation_support": lockbox_treatment_support,
            },
            "base_losses": base_losses,
            "candidate_losses": candidate_losses,
            "expert_metadata": expert_metadata,
        },
        "gates": {
            "development_residual_activation": development_residual_gate,
            "external_point_accuracy": point_gate,
            "candidate_specific_conformal": conformal,
            "anchor_masked_recovery": anchor_masked_recovery,
            "all_history_masked_recovery": masked_recovery,
            "cross_organ_no_route_ablation": cross_organ,
            "cross_organ_input_ablation": cross_organ_input,
            "owner_organ_point_accuracy": owner_organ_point_gate,
            "multi_step_consistency": trajectory,
            "explicit_conservation": conservation,
            "independent_untouched_lockbox": independent_untouched_lockbox,
        },
        "full_release_promotion_pass": full_promotion,
        "promotion_status": (
            "ready_for_runtime_review" if full_promotion else "candidate_only"
        ),
        "causal_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "persistent_whole_body_jepa_candidate.v2",
                "candidate_only": not full_promotion,
                "target_residual_enabled": bool(
                    development_residual_gate["enabled"]
                ),
                "cell": key,
                "variables": list(variables),
                "module_names": list(module_names),
                "state_dict": candidate.state_dict(),
                "scaler_medians": arrays[0].medians,
                "scaler_scales": arrays[0].scales,
                "fixed_training_seed": TRAINING_SEED,
                "external_gate_report": str(args.output),
                "causal_claim_allowed": False,
            },
            args.checkpoint,
        )
    print(json.dumps({
        "cell": key,
        "development_residual_enabled": development_residual_gate["enabled"],
        "external_point_pass": point_gate["pass"],
        "conformal_pass": conformal["pass"],
        "anchor_masked_recovery_pass": anchor_masked_recovery["pass"],
        "all_history_masked_recovery_pass": masked_recovery["pass"],
        "cross_organ_pass": cross_organ["pass"],
        "cross_organ_input_pass": cross_organ_input["pass"],
        "owner_organ_point_pass": owner_organ_point_gate["pass"],
        "multi_step_pass": trajectory["pass"],
        "conservation_pass": conservation["pass"],
        "full_release_promotion_pass": full_promotion,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
