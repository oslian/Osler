"""Formal point-accuracy gate for the persistent physiology candidate.

The candidate is compared against three frozen-at-test-time references:

* the original joint JEPA value decoder;
* a per-cell ridge expert fitted on the original JEPA's 311-value feature
  contract; and
* patient persistence.

Every comparison is patient equalized.  A cell passes only when it wins all
seven patient splits, paired patient bootstrap excludes zero, and the same
direction holds in a held-out hospital.  This script evaluates point accuracy;
interval, recovery, conservation, and external-database gates remain separate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
import torch

from whole_body_joint_jepa import fit_joint, load_joint_event_examples


MODULES = (
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
PATIENT_SPLIT_SEEDS = (11, 23, 37, 53, 71, 89, 107)


def _continuous_release_cells(registry_path: Path) -> tuple[tuple[str, int], ...]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    cells = []
    for contract in payload["cells"]:
        target = str(contract["target"])
        if target.endswith("_event"):
            continue
        cells.append((target, int(contract["horizon_hours"])))
    return tuple(cells)


def _model_call(model, arrays, batch):
    (
        _, input_matrix, current, mask, ages, _, _, history, horizon, presence,
        *_,
    ) = arrays
    return model(
        torch.from_numpy(input_matrix[history[batch]]).float(),
        torch.from_numpy(current[batch]).float(),
        torch.from_numpy(mask[batch]).float(),
        torch.from_numpy(ages[batch]).float(),
        torch.from_numpy(presence[batch]).float(),
        torch.from_numpy(horizon[batch]).float(),
    )


def _predict_values(model, arrays, rows, batch_size=1024):
    predictions = []
    latents = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = np.asarray(rows[start : start + batch_size], dtype=np.int64)
            output = _model_call(model, arrays, batch)
            predictions.append(output["value"].cpu().numpy())
            latents.append(output["predicted_latent"].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(latents)


def _base_feature_cache(
    model,
    arrays,
    rows,
    variables,
    cells,
    batch_size=512,
):
    """Build only release-cell features instead of materializing N x 53 x 64."""

    _, _, current, _, _, _, _, _, horizon, *_ = arrays
    variable_index = {variable: index for index, variable in enumerate(variables)}
    cache = {
        f"{target}@{horizon_hours}h": {"rows": [], "features": []}
        for target, horizon_hours in cells
        if target in variable_index
    }
    predictions = np.empty((len(rows), len(variables)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            batch = np.asarray(rows[start:stop], dtype=np.int64)
            output = _model_call(model, arrays, batch)
            predictions[start:stop] = output["value"].cpu().numpy()
            context = output["context_latent"].cpu().numpy()
            belief = output["belief_state"].cpu().numpy()
            target_future = output["predicted_target_latent"].cpu().numpy()
            organs = output["organ_latents"].cpu().numpy()
            base = output["value"].cpu().numpy()
            for target, horizon_hours in cells:
                if target not in variable_index:
                    continue
                selected = np.flatnonzero(
                    np.isclose(horizon[batch], float(horizon_hours))
                )
                if not len(selected):
                    continue
                index = variable_index[target]
                organ_index = int(model.variable_module_ids[index].item())
                features = np.concatenate(
                    [
                        context[selected],
                        belief[selected],
                        target_future[selected, index],
                        organs[selected, organ_index],
                        base[selected, index, None],
                        current[batch[selected]],
                        current[batch[selected], index, None],
                    ],
                    axis=1,
                ).astype(np.float32)
                key = f"{target}@{horizon_hours}h"
                cache[key]["rows"].append(batch[selected])
                cache[key]["features"].append(features)
    for payload in cache.values():
        if payload["rows"]:
            payload["rows"] = np.concatenate(payload["rows"])
            payload["features"] = np.concatenate(payload["features"])
        else:
            payload["rows"] = np.empty(0, dtype=np.int64)
            payload["features"] = np.empty((0, 311), dtype=np.float32)
    return predictions, cache


def _fit_cell_experts(train_cache, test_cache, arrays, variables, cells):
    _, _, current, _, _, future, future_mask, *_ = arrays
    variable_index = {variable: index for index, variable in enumerate(variables)}
    test_prediction = {}
    metadata = {}
    for target, horizon_hours in cells:
        key = f"{target}@{horizon_hours}h"
        if key not in train_cache or target not in variable_index:
            continue
        index = variable_index[target]
        train_rows = train_cache[key]["rows"]
        test_rows = test_cache[key]["rows"]
        train_valid = future_mask[train_rows, index] > 0.0
        if int(train_valid.sum()) < 100 or not len(test_rows):
            continue
        x_train = train_cache[key]["features"][train_valid].astype(np.float64)
        residual = (
            future[train_rows[train_valid], index]
            - current[train_rows[train_valid], index]
        ).astype(np.float64)
        center = np.median(x_train, axis=0)
        scale = 1.4826 * np.median(np.abs(x_train - center), axis=0)
        standard = np.std(x_train, axis=0)
        scale = np.where(np.isfinite(scale) & (scale >= 1e-5), scale, standard)
        scale = np.where(np.isfinite(scale) & (scale >= 1e-5), scale, 1.0)
        model = Ridge(alpha=10.0)
        model.fit((x_train - center) / scale, residual)
        predicted_residual = model.predict(
            (test_cache[key]["features"].astype(np.float64) - center) / scale
        )
        limit = max(float(np.quantile(np.abs(residual), 0.995)), 1e-4)
        predicted_residual = np.clip(predicted_residual, -limit, limit)
        test_prediction[key] = (
            current[test_rows, index] + predicted_residual
        ).astype(np.float32)
        metadata[key] = {
            "train_rows": int(train_valid.sum()),
            "ridge_alpha": 10.0,
            "normalized_residual_limit": limit,
        }
    return test_prediction, metadata


def _patient_cell_errors(
    frame,
    arrays,
    rows,
    variables,
    cells,
    base_prediction,
    candidate_prediction,
    expert_cache,
    expert_prediction,
    min_rows=20,
    min_patients=8,
):
    _, _, current, _, _, future, future_mask, _, horizon, *_ = arrays
    variable_index = {variable: index for index, variable in enumerate(variables)}
    row_offset = {int(row): offset for offset, row in enumerate(rows)}
    output = {}
    for target, horizon_hours in cells:
        if target not in variable_index:
            continue
        key = f"{target}@{horizon_hours}h"
        if key not in expert_prediction:
            continue
        index = variable_index[target]
        expert_rows = expert_cache[key]["rows"]
        valid = (
            (future_mask[expert_rows, index] > 0.0)
            & np.isclose(horizon[expert_rows], float(horizon_hours))
        )
        selected_rows = expert_rows[valid]
        if len(selected_rows) < int(min_rows):
            continue
        offsets = np.asarray([row_offset[int(row)] for row in selected_rows])
        patients = frame.iloc[selected_rows]["subject_id"].astype(str).to_numpy()
        if np.unique(patients).size < int(min_patients):
            continue
        truth = future[selected_rows, index]
        errors = {
            "persistent": np.abs(candidate_prediction[offsets, index] - truth),
            "base": np.abs(base_prediction[offsets, index] - truth),
            "expert": np.abs(expert_prediction[key][valid] - truth),
            "persistence": np.abs(current[selected_rows, index] - truth),
        }
        patient_rows = []
        for patient in np.unique(patients):
            patient_mask = patients == patient
            patient_rows.append(
                {
                    "patient": patient,
                    **{
                        name: float(np.mean(values[patient_mask]))
                        for name, values in errors.items()
                    },
                }
            )
        output[key] = {
            "rows": int(len(selected_rows)),
            "patients": int(len(patient_rows)),
            "patient_errors": patient_rows,
            **{
                f"{name}_patient_equalized_mae": float(
                    np.mean([row[name] for row in patient_rows])
                )
                for name in errors
            },
        }
    return output


def _bootstrap_delta(patient_errors, comparator, seed=20260828, draws=2000):
    candidate = np.asarray(
        [payload["persistent"] for payload in patient_errors], dtype=np.float64
    )
    reference = np.asarray(
        [payload[comparator] for payload in patient_errors], dtype=np.float64
    )
    delta = candidate - reference
    if not len(delta):
        return {"delta": None, "ci95": [None, None], "patients": 0, "pass": False}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(int(draws), len(delta)))
    samples = delta[indices].mean(axis=1)
    ci = np.quantile(samples, [0.025, 0.975])
    return {
        "delta": float(delta.mean()),
        "ci95": [float(ci[0]), float(ci[1])],
        "patients": int(len(delta)),
        "pass": bool(ci[1] < 0.0),
    }


def _aggregate_patient_splits(split_results, hospital_result, expected_split_count):
    all_cells = sorted(
        set().union(*(result["cells"].keys() for result in split_results))
    )
    output = []
    for cell in all_cells:
        supported = [result["cells"][cell] for result in split_results if cell in result["cells"]]
        patient_accumulator = defaultdict(lambda: defaultdict(list))
        for result in split_results:
            payload = result["cells"].get(cell)
            if payload is None:
                continue
            for row in payload["patient_errors"]:
                for model in ("persistent", "base", "expert", "persistence"):
                    patient_accumulator[row["patient"]][model].append(row[model])
        patient_errors = []
        for patient, values in patient_accumulator.items():
            patient_errors.append(
                {
                    "patient": patient,
                    **{model: float(np.mean(error)) for model, error in values.items()},
                }
            )
        comparisons = {
            comparator: _bootstrap_delta(patient_errors, comparator)
            for comparator in ("base", "expert", "persistence")
        }
        wins = {
            comparator: sum(
                payload["persistent_patient_equalized_mae"]
                < payload[f"{comparator}_patient_equalized_mae"]
                for payload in supported
            )
            for comparator in ("base", "expert", "persistence")
        }
        hospital = hospital_result["cells"].get(cell)
        hospital_wins = {
            comparator: bool(
                hospital is not None
                and hospital["persistent_patient_equalized_mae"]
                < hospital[f"{comparator}_patient_equalized_mae"]
            )
            for comparator in ("base", "expert", "persistence")
        }
        full_pass = bool(
            len(supported) == int(expected_split_count)
            and all(wins[name] == int(expected_split_count) for name in wins)
            and all(comparisons[name]["pass"] for name in comparisons)
            and all(hospital_wins.values())
        )
        output.append(
            {
                "cell": cell,
                "supported_patient_splits": len(supported),
                "patient_split_wins": wins,
                "paired_patient_bootstrap": comparisons,
                "hospital_heldout_wins": hospital_wins,
                "hospital_heldout_metrics": hospital,
                "point_accuracy_gate_pass": full_pass,
            }
        )
    return output


def _split_patients(frame, seed):
    patients = np.asarray(sorted(frame["subject_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    split = max(1, int(round(0.70 * len(patients))))
    train = set(patients[:split])
    test = set(patients[split:])
    patient_values = frame["subject_id"].astype(str)
    return (
        np.flatnonzero(patient_values.isin(train).to_numpy()),
        np.flatnonzero(patient_values.isin(test).to_numpy()),
    )


def _split_hospital(frame):
    patient_hospital = (
        frame[["subject_id", "hospitalid"]]
        .drop_duplicates("subject_id")
        .assign(hospitalid=lambda value: value["hospitalid"].astype(str))
    )
    counts = patient_hospital.groupby("hospitalid")["subject_id"].nunique()
    test_hospital = str(counts.sort_values(ascending=False).index[0])
    test_patient = set(
        patient_hospital.loc[
            patient_hospital["hospitalid"] == test_hospital, "subject_id"
        ].astype(str)
    )
    patient_values = frame["subject_id"].astype(str)
    test_rows = np.flatnonzero(patient_values.isin(test_patient).to_numpy())
    train_rows = np.flatnonzero(~patient_values.isin(test_patient).to_numpy())
    return train_rows, test_rows, test_hospital


def _run_split(
    name,
    frame,
    variables,
    module_names,
    cells,
    train_rows,
    test_rows,
    training_seed,
    epochs,
    supervised_targets=(),
):
    common = dict(
        seed=int(training_seed),
        epochs=int(epochs),
        batch_size=1024,
        world_model=True,
        target_encoder_mode="fast",
        future_target_mode="window",
        regime_balanced_loss=True,
        state_normalization="robust",
        supervised_targets=tuple(supervised_targets) or None,
        future_task_balanced_sampling=bool(supervised_targets),
        target_head_stage_epochs=1 if supervised_targets else 0,
    )
    base_model, base_arrays, base_losses = fit_joint(
        frame,
        variables,
        module_names,
        train_rows,
        persistent_physiology_state=False,
        **common,
    )
    candidate_model, candidate_arrays, candidate_losses = fit_joint(
        frame,
        variables,
        module_names,
        train_rows,
        persistent_physiology_state=True,
        initial_state_dict=base_model.state_dict(),
        **common,
    )
    base_train_prediction, train_cache = _base_feature_cache(
        base_model, base_arrays, train_rows, variables, cells
    )
    del base_train_prediction
    base_test_prediction, test_cache = _base_feature_cache(
        base_model, base_arrays, test_rows, variables, cells
    )
    candidate_prediction, candidate_latent = _predict_values(
        candidate_model, candidate_arrays, test_rows
    )
    expert_prediction, expert_metadata = _fit_cell_experts(
        train_cache, test_cache, base_arrays, variables, cells
    )
    cell_errors = _patient_cell_errors(
        frame,
        candidate_arrays,
        test_rows,
        variables,
        cells,
        base_test_prediction,
        candidate_prediction,
        test_cache,
        expert_prediction,
    )
    summary = {
        "name": name,
        "train_patients": int(frame.iloc[train_rows]["subject_id"].nunique()),
        "test_patients": int(frame.iloc[test_rows]["subject_id"].nunique()),
        "evaluated_cells": len(cell_errors),
        "candidate_beats_expert_cells": sum(
            payload["persistent_patient_equalized_mae"]
            < payload["expert_patient_equalized_mae"]
            for payload in cell_errors.values()
        ),
        "candidate_beats_expert_and_persistence_cells": sum(
            payload["persistent_patient_equalized_mae"]
            < payload["expert_patient_equalized_mae"]
            and payload["persistent_patient_equalized_mae"]
            < payload["persistence_patient_equalized_mae"]
            for payload in cell_errors.values()
        ),
        "latent_mean_absolute_value": float(np.mean(np.abs(candidate_latent))),
        "persistent_future_readout": "direct_state_change_projection",
    }
    print(json.dumps(summary), flush=True)
    return {
        "summary": summary,
        "base_losses": base_losses,
        "candidate_losses": candidate_losses,
        "expert_metadata": expert_metadata,
        "cells": cell_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-examples", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--max-stays", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--training-seed", type=int, default=20260829)
    parser.add_argument(
        "--patient-split-seeds",
        nargs="+",
        type=int,
        default=PATIENT_SPLIT_SEEDS,
    )
    parser.add_argument("--targets", nargs="*", default=())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame, variables, module_names = load_joint_event_examples(
        args.event_examples,
        MODULES,
        max_stays=args.max_stays,
        min_modules=2,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
    )
    cells = _continuous_release_cells(args.runtime_registry)
    if args.targets:
        requested = set(args.targets)
        unknown = sorted(requested - set(variables))
        if unknown:
            raise ValueError(f"targets not in canonical variables: {unknown}")
        cells = tuple(cell for cell in cells if cell[0] in requested)
    patient_results = []
    for seed in args.patient_split_seeds:
        train_rows, test_rows = _split_patients(frame, seed)
        patient_results.append(
            _run_split(
                f"patient_seed_{seed}",
                frame,
                variables,
                module_names,
                cells,
                train_rows,
                test_rows,
                args.training_seed,
                args.epochs,
                args.targets,
            )
        )
    hospital_train, hospital_test, hospital_id = _split_hospital(frame)
    hospital_result = _run_split(
        f"hospital_{hospital_id}_heldout",
        frame,
        variables,
        module_names,
        cells,
        hospital_train,
        hospital_test,
        args.training_seed,
        args.epochs,
        args.targets,
    )
    cells_report = _aggregate_patient_splits(
        patient_results,
        hospital_result,
        len(args.patient_split_seeds),
    )
    passed = [cell for cell in cells_report if cell["point_accuracy_gate_pass"]]
    report = {
        "schema": "persistent_physiology_state_point_gate.v1",
        "candidate_only": True,
        "comparison": {
            "candidate": "persistent_whole_body_jepa",
            "base": "joint_whole_body_jepa",
            "incumbent_style": "per_cell_ridge_on_311_value_base_jepa_contract",
            "safety_anchor": "patient_persistence",
        },
        "data": {
            "rows": int(len(frame)),
            "patients": int(frame["subject_id"].nunique()),
            "hospitals": int(frame["hospitalid"].astype(str).nunique()),
            "max_stays": int(args.max_stays),
        },
        "training": {
            "epochs": int(args.epochs),
            "patient_split_seeds": list(args.patient_split_seeds),
            "fixed_training_seed": int(args.training_seed),
            "hospital_heldout": hospital_id,
            "supervised_targets": list(args.targets),
        },
        "summary": {
            "evaluated_continuous_release_cells": len(cells_report),
            "point_accuracy_gate_pass_cells": len(passed),
            "passed_cells": [cell["cell"] for cell in passed],
            "any_cell_point_accuracy_proven": bool(passed),
            "whole_model_point_accuracy_proven": bool(cells_report)
            and len(passed) == len(cells_report),
        },
        "cells": cells_report,
        "patient_split_runs": [result["summary"] for result in patient_results],
        "hospital_split_run": hospital_result["summary"],
        "promotion_status": (
            "point_accuracy_cells_passed_other_gates_pending"
            if passed
            else "candidate_only_point_accuracy_not_proven"
        ),
        "not_evaluated_here": [
            "independent_database_lockbox",
            "90_percent_conformal_coverage",
            "masked_recovery",
            "cross_organ_ablation",
            "multi_step_conservation",
        ],
        "causal_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
