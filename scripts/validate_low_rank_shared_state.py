"""Test a low-rank whole-body state with the strict reconstruction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from scripts.train_cross_target_nowcast import _care_unit_gate
from scripts.validate_persistent_physiology_external import (
    _combine_development_lockbox,
)
from scripts.validate_persistent_physiology_state import MODULES
from scripts.validate_shared_state_reconstruction import (
    SEEDS,
    TRAINING_SEED,
    _bootstrap_ci,
    _masked_reconstruction_arrays,
    _patient_delta,
    _patient_partition,
    _permuted_physiology,
)
from whole_body_jepa.nowcast import related_siblings
from whole_body_joint_jepa import fit_joint, load_joint_event_examples


def _state_matrix(arrays) -> np.ndarray:
    return np.concatenate(
        [arrays[2], arrays[3], arrays[4] / 7.0, arrays[9]], axis=1
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-examples", type=Path, required=True)
    parser.add_argument("--lockbox-examples", type=Path, required=True)
    parser.add_argument("--development-max-stays", type=int, default=1000)
    parser.add_argument("--lockbox-max-stays", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-truth-age-hours", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    development, variables, module_names = load_joint_event_examples(
        args.development_examples,
        MODULES,
        horizons=(3,),
        max_stays=args.development_max_stays,
        min_modules=2,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
    )
    lockbox, lockbox_variables, lockbox_modules = load_joint_event_examples(
        args.lockbox_examples,
        MODULES,
        horizons=(3,),
        max_stays=args.lockbox_max_stays,
        min_modules=2,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
    )
    if tuple(variables) != tuple(lockbox_variables) or tuple(module_names) != tuple(
        lockbox_modules
    ):
        raise ValueError("development and lockbox contracts differ")
    frame, train_rows, lockbox_rows = _combine_development_lockbox(
        development, lockbox
    )
    placebo_frame = _permuted_physiology(
        frame, train_rows, tuple(variables), TRAINING_SEED
    )
    common = dict(
        seed=TRAINING_SEED,
        epochs=0,
        nowcast_pretrain_epochs=0,
        state_normalization="robust",
    )
    model, arrays, _ = fit_joint(
        frame, variables, module_names, train_rows, **common
    )
    placebo_model, placebo_arrays, _ = fit_joint(
        placebo_frame, variables, module_names, train_rows, **common
    )
    rank = min(int(args.rank), _state_matrix(arrays).shape[1] - 1)
    projector = PCA(
        n_components=rank, svd_solver="randomized", random_state=TRAINING_SEED
    )
    placebo_projector = PCA(
        n_components=rank, svd_solver="randomized", random_state=TRAINING_SEED
    )
    projector.fit(_state_matrix(arrays)[train_rows])
    placebo_projector.fit(_state_matrix(placebo_arrays)[train_rows])

    variable_index = {name: index for index, name in enumerate(variables)}
    reports: dict[str, object] = {}
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
            placebo_model, placebo_arrays, rows, hidden_indices
        )
        candidate_state = _state_matrix(candidate_view)[rows]
        placebo_state = _state_matrix(placebo_view)[rows]
        candidate = projector.inverse_transform(
            projector.transform(candidate_state)
        )[:, target_index]
        placebo = placebo_projector.inverse_transform(
            placebo_projector.transform(placebo_state)
        )[:, target_index]
        truth = arrays[2][rows, target_index]
        population = np.zeros_like(truth)
        seed_reports = []
        row_patients = frame.iloc[rows]["subject_id"].astype(str)
        for seed in SEEDS:
            test = _patient_partition(row_patients, seed) == 1
            subset_rows = rows[test]
            population_delta = _patient_delta(
                frame, subset_rows, truth[test], candidate[test], population[test]
            )
            placebo_delta = _patient_delta(
                frame, subset_rows, truth[test], candidate[test], placebo[test]
            )
            population_ci = _bootstrap_ci(
                population_delta, seed, args.bootstrap_samples
            )
            placebo_ci = _bootstrap_ci(
                placebo_delta, seed + 1000, args.bootstrap_samples
            )
            care_rows = frame.iloc[subset_rows].copy().reset_index(drop=True)
            care_rows["truth"] = truth[test]
            care_unit = _care_unit_gate(
                care_rows,
                candidate[test],
                population[test],
                placebo[test],
                minimum_subjects=50,
            )
            seed_reports.append(
                {
                    "seed": seed,
                    "delta_vs_population_ci95": population_ci,
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
                "low_rank_state": float(np.mean(np.abs(candidate - truth))),
                "population_center": float(np.mean(np.abs(population - truth))),
                "capacity_placebo": float(np.mean(np.abs(placebo - truth))),
            },
            "seed_reports": seed_reports,
            "promoted": passed,
        }
        if passed:
            promoted.append(target)

    external_databases = sorted(lockbox["database"].dropna().astype(str).unique())
    payload = {
        "schema": "whole_body_jepa.low_rank_shared_state.v1",
        "training_database": "eICU",
        "external_validation_database": external_databases,
        "rank": rank,
        "development_patients": int(development["subject_id"].nunique()),
        "lockbox_patients": int(lockbox["subject_id"].nunique()),
        "promoted_targets": promoted,
        "reports": reports,
        "causal_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"rank": rank, "promoted": promoted}, indent=2))


if __name__ == "__main__":
    main()
