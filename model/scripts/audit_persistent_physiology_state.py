"""Bounded patient-heldout audit for the persistent physiology candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from whole_body_joint_jepa import fit_joint, load_joint_event_examples


def _patient_equalized_mae(error: np.ndarray, patients: np.ndarray) -> float:
    values = []
    for patient in np.unique(patients):
        selected = patients == patient
        if selected.any():
            values.append(float(np.mean(error[selected])))
    return float(np.mean(values)) if values else float("nan")


def _predict(model, arrays, rows, batch_size=1024):
    (
        _, input_matrix, current, mask, ages, _, _, history, horizon, presence,
        *_,
    ) = arrays
    predictions = []
    body_states = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = np.asarray(rows[start : start + batch_size], dtype=np.int64)
            output = model(
                torch.from_numpy(input_matrix[history[batch]]).float(),
                torch.from_numpy(current[batch]).float(),
                torch.from_numpy(mask[batch]).float(),
                torch.from_numpy(ages[batch]).float(),
                torch.from_numpy(presence[batch]).float(),
                torch.from_numpy(horizon[batch]).float(),
            )
            predictions.append(output["value"].cpu().numpy())
            body_states.append(output["predicted_latent"].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(body_states)


def _cell_report(frame, variables, arrays, rows, base, persistent):
    _, _, current, _, _, future, future_mask, _, horizon, *_ = arrays
    patients = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    output = []
    for index, variable in enumerate(variables):
        for cell_horizon in sorted(np.unique(horizon[rows])):
            valid = (
                (future_mask[rows, index] > 0.0)
                & np.isclose(horizon[rows], cell_horizon)
            )
            if int(valid.sum()) < 20:
                continue
            truth = future[rows[valid], index]
            current_value = current[rows[valid], index]
            base_error = np.abs(base[valid, index] - truth)
            persistent_error = np.abs(persistent[valid, index] - truth)
            persistence_error = np.abs(current_value - truth)
            cell_patients = patients[valid]
            base_mae = _patient_equalized_mae(base_error, cell_patients)
            candidate_mae = _patient_equalized_mae(
                persistent_error, cell_patients
            )
            persistence_mae = _patient_equalized_mae(
                persistence_error, cell_patients
            )
            output.append(
                {
                    "cell": f"{variable}@{float(cell_horizon):g}h",
                    "rows": int(valid.sum()),
                    "patients": int(np.unique(cell_patients).size),
                    "base_jepa_patient_equalized_mae": base_mae,
                    "persistent_state_patient_equalized_mae": candidate_mae,
                    "persistence_patient_equalized_mae": persistence_mae,
                    "delta_persistent_vs_base": candidate_mae - base_mae,
                    "delta_persistent_vs_persistence": candidate_mae
                    - persistence_mae,
                    "persistent_beats_base": candidate_mae < base_mae,
                    "persistent_beats_persistence": candidate_mae
                    < persistence_mae,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-examples", type=Path, required=True)
    parser.add_argument("--max-stays", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--targets", nargs="*", default=())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modules = (
        "sepsis",
        "aki",
        "respiratory",
        "electrolyte_acid_base",
        "endocrine_stress",
        "gi_pancreatic_nutrition",
        "cardiac_injury",
        "musculoskeletal_rhabdo",
        "immune_inflammatory",
        "cardiovascular_instability",
        "hepatic_failure",
        "coagulopathy_heme",
    )
    frame, variables, module_names = load_joint_event_examples(
        args.event_examples,
        modules,
        max_stays=args.max_stays,
        min_modules=2,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
    )
    patients = np.asarray(sorted(frame["subject_id"].astype(str).unique()))
    rng = np.random.default_rng(20260828)
    rng.shuffle(patients)
    split = max(1, int(round(0.70 * len(patients))))
    train_patients = set(patients[:split])
    test_patients = set(patients[split:])
    train_rows = np.flatnonzero(
        frame["subject_id"].astype(str).isin(train_patients).to_numpy()
    )
    test_rows = np.flatnonzero(
        frame["subject_id"].astype(str).isin(test_patients).to_numpy()
    )
    common = dict(
        seed=17,
        epochs=args.epochs,
        batch_size=1024,
        world_model=True,
        target_encoder_mode="fast",
        future_target_mode="window",
        regime_balanced_loss=True,
        state_normalization="robust",
        supervised_targets=tuple(args.targets) or None,
        future_task_balanced_sampling=bool(args.targets),
        target_head_stage_epochs=1 if args.targets else 0,
    )
    base_model, base_arrays, base_losses = fit_joint(
        frame,
        variables,
        module_names,
        train_rows,
        persistent_physiology_state=False,
        **common,
    )
    persistent_model, persistent_arrays, persistent_losses = fit_joint(
        frame,
        variables,
        module_names,
        train_rows,
        persistent_physiology_state=True,
        initial_state_dict=base_model.state_dict(),
        **common,
    )
    base_prediction, base_latent = _predict(base_model, base_arrays, test_rows)
    persistent_prediction, persistent_latent = _predict(
        persistent_model, persistent_arrays, test_rows
    )
    cells = _cell_report(
        frame,
        variables,
        persistent_arrays,
        test_rows,
        base_prediction,
        persistent_prediction,
    )
    if args.targets:
        cells = [
            cell
            for cell in cells
            if cell["cell"].split("@", 1)[0] in set(args.targets)
        ]
    improved = [cell for cell in cells if cell["persistent_beats_base"]]
    beat_both = [
        cell
        for cell in cells
        if cell["persistent_beats_base"]
        and cell["persistent_beats_persistence"]
    ]
    report = {
        "schema": "persistent_physiology_state_pilot.v1",
        "candidate_only": True,
        "data": {
            "rows": int(len(frame)),
            "patients": int(len(patients)),
            "train_patients": int(len(train_patients)),
            "test_patients": int(len(test_patients)),
        },
        "architecture": {
            "persistent_body_state": True,
            "persistent_organ_states": True,
            "sparse_organ_graph": True,
            "same_body_trajectory_for_all_target_heads": True,
            "derived_readout_consistency_audit": True,
            "derived_readout_overwrite": False,
        },
        "training": {
            "epochs": int(args.epochs),
            "supervised_targets": list(args.targets),
            "base_loss": base_losses,
            "persistent_loss": persistent_losses,
        },
        "summary": {
            "evaluated_cells": len(cells),
            "persistent_beats_base_cells": len(improved),
            "persistent_beats_base_and_persistence_cells": len(beat_both),
            "latent_mean_absolute_difference": float(
                np.mean(np.abs(persistent_latent - base_latent))
            ),
            "persistent_future_readout": "direct_state_change_projection",
        },
        "cells": cells,
        "promotion_status": "candidate_only_bounded_patient_heldout_pilot",
        "required_next_gate": [
            "7_seed_patient_heldout",
            "hospital_heldout",
            "paired_patient_bootstrap",
            "90_percent_conformal",
            "masked_recovery_per_target",
            "cross_organ_ablation",
        ],
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
