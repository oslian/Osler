"""Externally validate one shared body state by current-state reconstruction.

The target and its near-deterministic siblings are removed from the anchor and
complete causal history. One joint encoder must reconstruct every supported
target on MIMIC after fitting only on eICU. A capacity-matched placebo uses the
same architecture and missingness process but independently permuted eICU
physiology, preserving marginals while destroying cross-variable relations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.validate_persistent_physiology_external import (
    _combine_development_lockbox,
    _prefix_provenance,
)
from scripts.validate_persistent_physiology_state import MODULES, _model_call
from scripts.train_cross_target_nowcast import _care_unit_gate
from whole_body_jepa.nowcast import related_siblings
from whole_body_joint_jepa import fit_joint, load_joint_event_examples


SEEDS = (7, 11, 19, 23, 37, 53, 71)
TRAINING_SEED = 20260830


def _combine_development_sources(
    sources: list[pd.DataFrame],
) -> pd.DataFrame:
    """Combine source databases without crossing precomputed history indices."""

    if not sources:
        raise ValueError("at least one development source is required")
    history_columns = sorted(
        column for column in sources[0] if column.startswith("_history_")
    )
    frames = []
    offset = 0
    for index, source in enumerate(sources):
        if history_columns != sorted(
            column for column in source if column.startswith("_history_")
        ):
            raise ValueError("development history contracts differ")
        frame = _prefix_provenance(source, f"source_{index}")
        for column in history_columns:
            frame[column] = frame[column].astype(np.int64) + offset
        frames.append(frame)
        offset += len(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.attrs.update(copy.deepcopy(sources[0].attrs))
    return combined


def _source_balanced_rows(
    frame: pd.DataFrame,
    rows: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Give every development database equal sampling weight."""

    selected = np.asarray(rows, dtype=np.int64)
    database = frame.iloc[selected]["database"].fillna("__missing__").astype(str)
    groups = {
        name: selected[database.to_numpy() == name]
        for name in sorted(database.unique())
    }
    if len(groups) < 2:
        return selected
    target_count = max(len(values) for values in groups.values())
    rng = np.random.default_rng(seed)
    balanced = []
    for values in groups.values():
        balanced.append(
            rng.choice(values, size=target_count, replace=len(values) < target_count)
        )
    output = np.concatenate(balanced).astype(np.int64)
    rng.shuffle(output)
    return output


def _patient_partition(values: pd.Series, seed: int) -> np.ndarray:
    def bucket(value: object) -> int:
        digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % 2

    return values.map(bucket).to_numpy(dtype=np.int8)


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _patient_delta(
    frame: pd.DataFrame,
    rows: np.ndarray,
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    patient = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    delta = np.abs(candidate - truth) - np.abs(reference - truth)
    return (
        pd.DataFrame({"patient": patient, "delta": delta})
        .groupby("patient", sort=False)["delta"]
        .mean()
        .to_numpy(dtype=np.float64)
    )


def _permuted_physiology(
    frame: pd.DataFrame,
    rows: np.ndarray,
    variables: tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    placebo = frame.copy()
    rng = np.random.default_rng(seed)
    selected = np.asarray(rows, dtype=np.int64)
    for variable in variables:
        column = f"{variable}_t"
        values = pd.to_numeric(placebo.iloc[selected][column], errors="coerce").to_numpy()
        finite = np.flatnonzero(np.isfinite(values))
        shuffled = values.copy()
        shuffled[finite] = rng.permutation(values[finite])
        placebo.loc[placebo.index[selected], column] = shuffled
    placebo.attrs.update(frame.attrs)
    return placebo


def _masked_reconstruction_arrays(
    model,
    arrays,
    rows: np.ndarray,
    hidden_indices: tuple[int, ...],
):
    masked = list(arrays)
    input_matrix = arrays[1].copy()
    current = arrays[2].copy()
    mask = arrays[3].copy()
    ages = arrays[4].copy()
    history = arrays[7]
    presence = arrays[9].copy()
    selected = np.asarray(rows, dtype=np.int64)
    hidden = np.asarray(hidden_indices, dtype=np.int64)
    history_rows = np.unique(history[selected].reshape(-1))
    variable_count = current.shape[1]

    input_matrix[np.ix_(history_rows, hidden)] = 0.0
    input_matrix[np.ix_(history_rows, variable_count + hidden)] = 0.0
    input_matrix[np.ix_(history_rows, 2 * variable_count + hidden)] = 7.0
    current[np.ix_(selected, hidden)] = 0.0
    mask[np.ix_(selected, hidden)] = 0.0
    ages[np.ix_(selected, hidden)] = 7.0

    membership = model.variable_module_membership.detach().cpu().numpy() > 0.5
    presence_start = 3 * variable_count + 1
    for organ_index, organ_variables in enumerate(membership):
        remaining = np.flatnonzero(organ_variables & ~np.isin(np.arange(variable_count), hidden))
        anchor_present = (
            mask[np.ix_(selected, remaining)].sum(axis=1) > 0.0
            if len(remaining)
            else np.zeros(len(selected), dtype=bool)
        )
        presence[selected, organ_index] *= anchor_present.astype(np.float32)
        if input_matrix.shape[1] > presence_start + organ_index:
            history_present = (
                input_matrix[np.ix_(history_rows, variable_count + remaining)].sum(axis=1)
                > 0.0
                if len(remaining)
                else np.zeros(len(history_rows), dtype=bool)
            )
            input_matrix[np.ix_(history_rows, [presence_start + organ_index])] = (
                history_present.astype(np.float32)[:, None]
            )

    masked[1] = input_matrix
    masked[2] = current
    masked[3] = mask
    masked[4] = ages
    masked[9] = presence
    return tuple(masked)


def _predict_reconstruction(model, arrays, rows: np.ndarray, batch_size: int = 1024):
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = np.asarray(rows[start : start + batch_size], dtype=np.int64)
            output.append(_model_call(model, arrays, batch)["reconstruction"].cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-examples", type=Path, nargs="+", required=True
    )
    parser.add_argument("--lockbox-examples", type=Path, required=True)
    parser.add_argument("--development-max-stays", type=int, default=1000)
    parser.add_argument("--lockbox-max-stays", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-truth-age-hours", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--source-balanced", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    development_sources = []
    variables = None
    module_names = None
    for path in args.development_examples:
        source, source_variables, source_modules = load_joint_event_examples(
            path,
            MODULES,
            horizons=(3,),
            max_stays=args.development_max_stays,
            min_modules=2,
            organ_contract_mode="canonical_organs",
            history_mode="multiscale",
        )
        if variables is None:
            variables = source_variables
            module_names = source_modules
        elif tuple(variables) != tuple(source_variables) or tuple(module_names) != tuple(
            source_modules
        ):
            raise ValueError("development source contracts differ")
        development_sources.append(source)
    development = _combine_development_sources(development_sources)
    lockbox, lockbox_variables, lockbox_modules = load_joint_event_examples(
        args.lockbox_examples,
        MODULES,
        horizons=(3,),
        max_stays=args.lockbox_max_stays,
        min_modules=2,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
    )
    if tuple(variables) != tuple(lockbox_variables) or tuple(module_names) != tuple(lockbox_modules):
        raise ValueError("development and lockbox contracts differ")
    frame, train_rows, lockbox_rows = _combine_development_lockbox(development, lockbox)
    placebo_frame = _permuted_physiology(
        frame, train_rows, tuple(variables), TRAINING_SEED
    )
    fit_rows = (
        _source_balanced_rows(frame, train_rows, TRAINING_SEED)
        if args.source_balanced
        else train_rows
    )
    common = dict(
        seed=TRAINING_SEED,
        epochs=0,
        batch_size=1024,
        nowcast_pretrain_epochs=int(args.epochs),
        masked_forecast_probability=0.15,
        module_mask_probability=0.25,
        structured_mask_probability=1.0,
        reconstruction_truth_max_age_hours=float(args.max_truth_age_hours),
        measurement_consistency_weight=0.10,
        state_normalization="robust",
    )
    model, arrays, losses = fit_joint(
        frame, variables, module_names, fit_rows, **common
    )
    placebo, placebo_arrays, placebo_losses = fit_joint(
        placebo_frame, variables, module_names, fit_rows, **common
    )

    reports: dict[str, object] = {}
    variable_index = {name: index for index, name in enumerate(variables)}
    promoted = []
    selected = np.asarray(lockbox_rows, dtype=np.int64)
    for target, target_index in variable_index.items():
        valid = (arrays[3][selected, target_index] > 0.0) & (
            arrays[4][selected, target_index] <= float(args.max_truth_age_hours)
        )
        rows = selected[valid]
        patients = int(frame.iloc[rows]["subject_id"].nunique())
        if len(rows) < 500 or patients < 100:
            reports[target] = {
                "status": "insufficient_support",
                "rows": int(len(rows)),
                "patients": patients,
            }
            continue
        hidden_names = (target,) + tuple(
            value
            for value in sorted(related_siblings(target))
            if value in variable_index
        )
        hidden_indices = tuple(variable_index[value] for value in hidden_names)
        candidate_view = _masked_reconstruction_arrays(
            model, arrays, rows, hidden_indices
        )
        placebo_view = _masked_reconstruction_arrays(
            placebo, placebo_arrays, rows, hidden_indices
        )
        candidate = _predict_reconstruction(model, candidate_view, rows)[
            :, target_index
        ]
        placebo_prediction = _predict_reconstruction(
            placebo, placebo_view, rows
        )[:, target_index]
        truth = arrays[2][rows, target_index]
        population = np.zeros_like(truth)
        seed_reports = []
        row_patients = frame.iloc[rows]["subject_id"].astype(str)
        for seed in SEEDS:
            test = _patient_partition(row_patients, seed) == 1
            subset_rows = rows[test]
            candidate_delta = _patient_delta(
                frame, subset_rows, truth[test], candidate[test], population[test]
            )
            placebo_delta = _patient_delta(
                frame,
                subset_rows,
                truth[test],
                candidate[test],
                placebo_prediction[test],
            )
            population_ci = _bootstrap_ci(
                candidate_delta, seed, args.bootstrap_samples
            )
            placebo_ci = _bootstrap_ci(
                placebo_delta, seed + 1000, args.bootstrap_samples
            )
            care_unit_rows = frame.iloc[subset_rows].copy().reset_index(drop=True)
            care_unit_rows["truth"] = truth[test]
            care_unit = _care_unit_gate(
                care_unit_rows,
                candidate[test],
                population[test],
                placebo_prediction[test],
                minimum_subjects=50,
            )
            seed_reports.append(
                {
                    "seed": seed,
                    "patients": int(frame.iloc[subset_rows]["subject_id"].nunique()),
                    "delta_vs_population": float(candidate_delta.mean()),
                    "delta_vs_population_ci95": population_ci,
                    "delta_vs_placebo": float(placebo_delta.mean()),
                    "delta_vs_placebo_ci95": placebo_ci,
                    "care_unit": care_unit,
                    "pass": bool(
                        population_ci[1] < 0.0
                        and placebo_ci[1] < 0.0
                        and care_unit["passed"]
                    ),
                }
            )
        passed = all(item["pass"] for item in seed_reports)
        reports[target] = {
            "status": "evaluated",
            "rows": int(len(rows)),
            "patients": patients,
            "hidden_variables": list(hidden_names),
            "normalized_mae": {
                "shared_state": float(np.mean(np.abs(candidate - truth))),
                "population_center": float(np.mean(np.abs(population - truth))),
                "capacity_placebo": float(np.mean(np.abs(placebo_prediction - truth))),
            },
            "seed_reports": seed_reports,
            "promoted": passed,
        }
        if passed:
            promoted.append(target)

    training_databases = sorted(
        development["database"].dropna().astype(str).unique()
    )
    external_databases = sorted(lockbox["database"].dropna().astype(str).unique())
    payload = {
        "schema": "whole_body_jepa.shared_state_reconstruction.v3",
        "training_database": training_databases,
        "external_validation_database": external_databases,
        "task": "same-anchor target-and-sibling-masked reconstruction",
        "training": {
            "development_rows": int(len(development)),
            "development_patients": int(development["subject_id"].nunique()),
            "development_databases": training_databases,
            "unique_development_rows": int(len(train_rows)),
            "effective_source_balanced_rows": int(len(fit_rows)),
            "source_balanced": bool(args.source_balanced),
            "lockbox_rows": int(len(lockbox)),
            "lockbox_patients": int(lockbox["subject_id"].nunique()),
            "epochs": int(args.epochs),
            "random_mask_probability": 0.15,
            "module_mask_probability": 0.25,
            "structured_mask_probability": 1.0,
            "reconstruction_truth_max_age_hours": float(
                args.max_truth_age_hours
            ),
            "model_losses": losses,
            "placebo_losses": placebo_losses,
        },
        "promoted_targets": promoted,
        "reports": reports,
        "causal_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"evaluated": len(reports), "promoted": promoted}, indent=2))


if __name__ == "__main__":
    main()
