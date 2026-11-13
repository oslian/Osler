"""Screen all continuous release cells for raw cross-variable source signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from osler_jepa.event_ledger import DEFAULT_MEASUREMENT_AGE_HOURS
from scripts.audit_change_behavior import label_mask
from scripts.audit_prospective_change import select_role_rows
from scripts.audit_release_masked_cells import input_frame, read_examples
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS
from whole_body_jepa.cross_variable_history import (
    cross_variable_history_features,
    cross_variable_history_views,
    matched_wrong_patient_rows,
)
from whole_body_jepa.multisource_intervals import (
    SEEDS,
    patient_split,
    representative_rows,
    sha256,
    stable_hash,
    subject_digest,
)
from whole_body_jepa.probability_calibration import patient_equal_weights
from whole_body_jepa.prospective_change import trend_features
from whole_body_jepa.runtime import HISTORY_LAGS_HOURS, _build_inputs, load_release


METHODS = (
    "own_history",
    "current_cross",
    "current_cross_placebo",
    "cross_history",
    "cross_history_placebo",
)


def target_role_masks(frame: pd.DataFrame, eligible: np.ndarray, target: str):
    """Choose a support-balanced hospital fold without inspecting outcomes."""

    hospitals = frame["hospitalid"].astype(str).to_numpy()
    subjects = frame["subject_id"].astype(str).to_numpy()
    known = ~pd.Series(hospitals).str.lower().isin({"unknown", "nan", "none", ""}).to_numpy()
    fold_ids = np.asarray(
        [stable_hash("catalog_cross_history_hospital", value) % 5 for value in hospitals]
    )
    total = max(1, len(np.unique(subjects[eligible])))
    candidates = []
    for fold in range(5):
        mask = known & (fold_ids == fold)
        support = len(np.unique(subjects[eligible & mask]))
        candidates.append((abs(support / total - 0.20), -support, fold, mask))
    _, _, selected_fold, hospital = min(candidates, key=lambda item: item[:3])
    excluded = np.isin(subjects, np.unique(subjects[hospital]))
    patient_bucket = np.asarray(
        [stable_hash("catalog_cross_history_patient", target, value) % 10 for value in subjects]
    )
    return {
        "fit": ~excluded & (patient_bucket < 7),
        "patient_test": ~excluded & (patient_bucket >= 7),
        "hospital_test": hospital,
    }, {
        "selected_fold": int(selected_fold),
        "supported_hospitals": int(len(np.unique(hospitals[hospital & eligible]))),
        "eligible_patients": int(total),
        "heldout_eligible_patients": int(len(np.unique(subjects[hospital & eligible]))),
    }


def continuous_cells(registry: dict) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (str(item["cell"]), str(item["target"]), int(item["horizon_hours"]))
        for item in registry["cells"]
        if not str(item["target"]).endswith("_event")
    )


def valid_rows(frame: pd.DataFrame, target: str, release_max_age: float) -> np.ndarray:
    age_limit = min(
        float(release_max_age),
        float(DEFAULT_MEASUREMENT_AGE_HOURS.get(target, release_max_age)),
    )
    age = pd.to_numeric(frame[f"{target}_age_hr"], errors="coerce")
    return label_mask(frame, target) & age.between(0.0, age_limit).to_numpy()


def prepare_features(frame, rows, target, assets, arrays, observations):
    selected = frame.iloc[rows].reset_index(drop=True)
    own = trend_features(frame, rows, target)
    current, full = cross_variable_history_features(
        frame,
        rows,
        variables=assets.checkpoint["variables"],
        target=target,
        normalized=observations[0],
        observed=observations[1],
        ages=observations[2],
        history_indices=arrays["history"],
        history_lags_hours=HISTORY_LAGS_HOURS,
    )
    donors = matched_wrong_patient_rows(
        selected,
        seed=stable_hash("catalog_cross_history_donor", target),
    )
    return selected, cross_variable_history_views(own, current, full, donors)


def fit_regressor(features, delta, subjects):
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        learning_rate=0.05,
        early_stopping=False,
        random_state=20260908,
    )
    return model.fit(features, delta, sample_weight=patient_equal_weights(subjects))


def paired_patient_delta(frame, candidate, reference, truth, *, salt, repetitions):
    values = np.abs(candidate - truth) - np.abs(reference - truth)
    grouped = pd.DataFrame(
        {"subject_id": frame.subject_id.to_numpy(), "delta": values}
    ).groupby("subject_id", sort=False)["delta"].mean().to_numpy(float)
    if len(grouped) < 30 or not np.isfinite(grouped).all():
        return {
            "patients": len(grouped),
            "mean": None,
            "ci95": None,
            "pass": False,
            "reason": "insufficient_finite_patients",
        }
    rng = np.random.default_rng(stable_hash("catalog_cross_history_bootstrap", salt))
    draws = np.asarray(
        [rng.choice(grouped, len(grouped), replace=True).mean() for _ in range(repetitions)]
    )
    interval = np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "patients": len(grouped),
        "mean": float(grouped.mean()),
        "ci95": [float(interval[0]), float(interval[1])],
        "pass": bool(grouped.mean() < 0 and interval[1] < 0),
    }


def seven_resample_deltas(frame, candidate, reference, truth):
    values = np.abs(candidate - truth) - np.abs(reference - truth)
    results = []
    for seed in SEEDS:
        _, test = patient_split(frame.subject_id, seed)
        patient = pd.DataFrame(
            {
                "subject_id": frame.loc[test, "subject_id"].to_numpy(),
                "delta": values[test],
            }
        ).groupby("subject_id", sort=False)["delta"].mean()
        results.append(
            {
                "seed": seed,
                "patients": len(patient),
                "mean": None if patient.empty else float(patient.mean()),
                "pass": bool(not patient.empty and patient.mean() < 0),
            }
        )
    return results


def comparison(frame, predictions, truth, candidate, reference, *, salt, repetitions):
    paired = paired_patient_delta(
        frame,
        predictions[candidate],
        predictions[reference],
        truth,
        salt=salt,
        repetitions=repetitions,
    )
    resamples = seven_resample_deltas(
        frame, predictions[candidate], predictions[reference], truth
    )
    return {
        "paired_patient": paired,
        "seven_patient_resamples": resamples,
        "pass": bool(paired["pass"] and all(item["pass"] for item in resamples)),
    }


def semantic_gate(
    frame,
    predictions,
    truth,
    current,
    target,
    candidate,
    references,
    *,
    role,
    repetitions,
):
    comparisons = {
        reference: comparison(
            frame,
            predictions,
            truth,
            candidate,
            reference,
            salt=f"{target}:{role}:{candidate}:{reference}",
            repetitions=repetitions,
        )
        for reference in references
    }
    threshold = float(CHANGE_THRESHOLDS[target])
    stable = np.abs(truth - current) < threshold
    stable_result = paired_patient_delta(
        frame.loc[stable].reset_index(drop=True),
        predictions[candidate][stable],
        current[stable],
        truth[stable],
        salt=f"{target}:{role}:{candidate}:stable",
        repetitions=repetitions,
    )
    stable_pass = bool(
        stable_result["ci95"] is not None
        and stable_result["ci95"][1] <= 0.05 * threshold
    )
    return {
        "comparisons": comparisons,
        "stable_harm_vs_persistence": stable_result,
        "pass": bool(all(item["pass"] for item in comparisons.values()) and stable_pass),
    }


def evaluate(frame, features, models, target, role, repetitions):
    selected = representative_rows(
        frame, stable_hash("catalog_cross_history_primary", target, role)
    )
    work = frame.iloc[selected].reset_index(drop=True)
    current = work[f"{target}_t"].to_numpy(float)
    truth = work[f"future_{target}"].to_numpy(float)
    predictions = {
        method: current + models[method].predict(features[method][selected])
        for method in METHODS
    }
    predictions["persistence"] = current.copy()
    current_gate = semantic_gate(
        work,
        predictions,
        truth,
        current,
        target,
        "current_cross",
        ("own_history", "current_cross_placebo"),
        role=role,
        repetitions=repetitions,
    )
    history_gate = semantic_gate(
        work,
        predictions,
        truth,
        current,
        target,
        "cross_history",
        ("current_cross", "cross_history_placebo"),
        role=role,
        repetitions=repetitions,
    )
    return {
        "patients": int(work.subject_id.nunique()),
        "mae": {
            name: float(np.mean(np.abs(values - truth)))
            for name, values in predictions.items()
        },
        "current_semantics": current_gate,
        "temporal_semantics": history_gate,
        "candidate_vs_persistence": {
            "current_cross": comparison(
                work,
                predictions,
                truth,
                "current_cross",
                "persistence",
                salt=f"{target}:{role}:current:persistence",
                repetitions=repetitions,
            ),
            "cross_history": comparison(
                work,
                predictions,
                truth,
                "cross_history",
                "persistence",
                salt=f"{target}:{role}:history:persistence",
                repetitions=repetitions,
            ),
        },
    }


def write_report(report, output):
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(args):
    torch.set_num_threads(args.threads)
    assets = load_release(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = Path("CROSS_VARIABLE_HISTORY_SCREEN_PROTOCOL_20260908.md")
    bindings = {
        path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()
    }
    cells = continuous_cells(assets.registry)
    report = {
        "schema": "whole_body_jepa.cross_variable_history_catalog_screen.v1",
        "complete": False,
        "status": "source_attribution_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "release_binding": bindings,
        "protocol_sha256": sha256(protocol),
        "input_sha256": sha256(args.eicu),
        "catalog_continuous_cells": len(cells),
        "history_lags_hours": [24.0, 12.0, 6.0, 3.0, 1.0],
        "cells": {},
    }
    for horizon in sorted({item[2] for item in cells}):
        print(f"read eICU {horizon}h", flush=True)
        frame = read_examples(args.eicu, assets, "eicu", horizon, grid=True)
        filtered = input_frame(
            frame, assets.checkpoint["variables"], assets.max_age_hours
        )
        arrays = _build_inputs(filtered, assets, horizon)
        observations = assets.scaler.normalize_current(filtered)
        for cell, target, cell_horizon in cells:
            if cell_horizon != horizon:
                continue
            print(f"screen {cell}", flush=True)
            eligible = valid_rows(frame, target, assets.max_age_hours)
            roles, hospital_role = target_role_masks(frame, eligible, cell)
            role_subjects = {
                role: set(frame.loc[mask, "subject_id"]) for role, mask in roles.items()
            }
            if any(
                role_subjects[left] & role_subjects[right]
                for left in role_subjects
                for right in role_subjects
                if left < right
            ):
                raise ValueError("patient roles overlap")
            role_data = {}
            role_rows = {}
            for role in ("fit", "patient_test", "hospital_test"):
                rows = select_role_rows(
                    frame,
                    eligible,
                    roles[role],
                    patients=(args.max_fit_patients if role == "fit" else args.max_test_patients),
                    anchors=(args.fit_anchors if role == "fit" else 1),
                    salt=f"catalog-cross-history:{cell}:{role}",
                )
                role_rows[role] = rows
                if len(rows) < args.min_rows or frame.iloc[rows].subject_id.nunique() < args.min_patients:
                    role_data[role] = None
                    continue
                role_data[role] = prepare_features(
                    frame, rows, target, assets, arrays, observations
                )
            support = {
                role: {
                    "rows": int(len(role_rows[role])),
                    "patients": int(frame.iloc[role_rows[role]].subject_id.nunique()),
                    "subject_digest": None
                    if not len(role_rows[role])
                    else subject_digest(frame.iloc[role_rows[role]].subject_id),
                }
                for role, data in role_data.items()
            }
            if any(data is None for data in role_data.values()):
                report["cells"][cell] = {
                    "target": target,
                    "horizon_hours": horizon,
                    "status": "insufficient_support",
                    "support": support,
                    "hospital_role": hospital_role,
                    "observation_aware_encoder_eligible": False,
                }
                write_report(report, args.output)
                continue
            fit_frame, fit_features = role_data["fit"]
            delta = (
                fit_frame[f"future_{target}"] - fit_frame[f"{target}_t"]
            ).to_numpy(float)
            models = {
                method: fit_regressor(
                    fit_features[method], delta, fit_frame.subject_id.to_numpy()
                )
                for method in METHODS
            }
            evaluations = {
                role: evaluate(
                    role_data[role][0],
                    role_data[role][1],
                    models,
                    target,
                    role,
                    args.bootstrap,
                )
                for role in ("patient_test", "hospital_test")
            }
            current_pass = all(
                evaluations[role]["current_semantics"]["pass"]
                for role in evaluations
            )
            temporal_pass = all(
                evaluations[role]["temporal_semantics"]["pass"]
                for role in evaluations
            )
            report["cells"][cell] = {
                "target": target,
                "horizon_hours": horizon,
                "status": "evaluated",
                "support": support,
                "hospital_role": hospital_role,
                "feature_widths": {
                    method: int(fit_features[method].shape[1]) for method in METHODS
                },
                "evaluation": evaluations,
                "current_semantics_eligible": current_pass,
                "temporal_semantics_eligible": temporal_pass,
                "observation_aware_encoder_eligible": bool(current_pass or temporal_pass),
            }
            print(
                json.dumps(
                    {
                        "cell": cell,
                        "current": current_pass,
                        "temporal": temporal_pass,
                        "patient_mae": evaluations["patient_test"]["mae"],
                    }
                ),
                flush=True,
            )
            write_report(report, args.output)
        del frame, arrays, observations

    eligible = [
        cell
        for cell, item in report["cells"].items()
        if item.get("observation_aware_encoder_eligible")
    ]
    report["eligible_cells"] = eligible
    report["eligible_count"] = len(eligible)
    report["complete"] = True
    if bindings != {
        path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()
    }:
        raise ValueError("release changed during source screen")
    write_report(report, args.output)
    print(f"FINISHED: {len(eligible)}/{len(cells)} cells source-eligible", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--eicu",
        type=Path,
        default=Path(
            "/Users/chunyouchang/Desktop/llm_project/medical_jepa/"
            "eicu_canonical_event_examples_full_5000.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cross_variable_history_catalog_20260908_v2"),
    )
    parser.add_argument("--max-fit-patients", type=int, default=5000)
    parser.add_argument("--max-test-patients", type=int, default=1000)
    parser.add_argument("--fit-anchors", type=int, default=4)
    parser.add_argument("--min-patients", type=int, default=100)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
