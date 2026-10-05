"""Test whether heterogeneous near-miss forecasts contain complementary signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_change_behavior import frozen_outputs
from scripts.audit_prospective_change import select_role_rows
from scripts.audit_release_masked_cells import input_frame, read_examples
from scripts.screen_cross_variable_history_catalog import (
    METHODS,
    comparison,
    fit_regressor,
    prepare_features,
    semantic_gate,
    valid_rows,
)
from whole_body_jepa.multisource_intervals import SEEDS, sha256, stable_hash, subject_digest
from whole_body_jepa.near_miss_portfolio import (
    bootstrap_simplex_weights,
    patient_equal_mae,
    predict_portfolio,
    unique_prediction_columns,
)
from whole_body_jepa.runtime import _build_inputs, load_release


CELLS = ("bun@24h", "map@3h", "bicarbonate@12h", "creatinine@6h")
REAL_SOURCES = ("persistence", "release", "own_history", "current_cross", "cross_history")
PLACEBO_METHOD = {
    "persistence": "persistence",
    "release": "release",
    "own_history": "own_history",
    "current_cross": "current_cross_placebo",
    "cross_history": "cross_history_placebo",
}


def portfolio_role_masks(frame: pd.DataFrame, eligible: np.ndarray, cell: str):
    """Build disjoint fit/tune/patient/hospital roles without reading outcomes."""

    hospitals = frame["hospitalid"].astype(str).to_numpy()
    subjects = frame["subject_id"].astype(str).to_numpy()
    known = ~pd.Series(hospitals).str.lower().isin({"unknown", "nan", "none", ""}).to_numpy()
    folds = np.asarray([stable_hash("near_miss_hospital", value) % 5 for value in hospitals])
    total = max(1, len(np.unique(subjects[eligible])))
    options = []
    for fold in range(5):
        mask = known & (folds == fold)
        support = len(np.unique(subjects[eligible & mask]))
        options.append((abs(support / total - 0.20), -support, fold, mask))
    _, _, selected_fold, hospital = min(options, key=lambda item: item[:3])
    excluded = np.isin(subjects, np.unique(subjects[hospital]))
    bucket = np.asarray([stable_hash("near_miss_patient", cell, value) % 10 for value in subjects])
    roles = {
        "fit": ~excluded & (bucket < 5),
        "tune": ~excluded & (bucket >= 5) & (bucket < 7),
        "patient_test": ~excluded & (bucket >= 7),
        "hospital_test": hospital,
    }
    return roles, {
        "selected_fold": int(selected_fold),
        "eligible_patients": int(total),
        "heldout_eligible_patients": int(len(np.unique(subjects[hospital & eligible]))),
    }


def prediction_bundle(frame, whole_frame, features, models, assets, arrays, rows, cell):
    """Return aligned real and placebo point forecasts for one patient role."""

    target = cell.split("@", 1)[0]
    current = frame[f"{target}_t"].to_numpy(float)
    truth = frame[f"future_{target}"].to_numpy(float)
    methods = {
        method: current + models[method].predict(features[method]) for method in METHODS
    }
    release = frozen_outputs(
        assets, arrays, rows=rows, frame=whole_frame, cell=cell
    ).point.to_numpy(float)
    real = {
        "persistence": current,
        "release": release,
        "own_history": methods["own_history"],
        "current_cross": methods["current_cross"],
        "cross_history": methods["cross_history"],
    }
    placebo = {
        name: (
            real[name]
            if PLACEBO_METHOD[name] in {"persistence", "release", "own_history"}
            else methods[PLACEBO_METHOD[name]]
        )
        for name in REAL_SOURCES
    }
    return current, truth, real, placebo


def stack_matrix(predictions: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([predictions[name] for name in names])


def correlation(predictions: dict[str, np.ndarray], truth: np.ndarray, names: tuple[str, ...]):
    errors = np.column_stack([predictions[name] - truth for name in names])
    values = np.corrcoef(errors, rowvar=False)
    return {
        left: {right: float(values[i, j]) for j, right in enumerate(names)}
        for i, left in enumerate(names)
    }


def tune_portfolios(frame, real, placebo):
    truth = frame["_truth"].to_numpy(float)
    subjects = frame["subject_id"].to_numpy()
    raw = stack_matrix(real, REAL_SOURCES)
    unique, names, aliases = unique_prediction_columns(raw, REAL_SOURCES)
    placebo_matrix = stack_matrix(placebo, names)
    weights, bags = bootstrap_simplex_weights(unique, truth, subjects, SEEDS)
    placebo_weights, placebo_bags = bootstrap_simplex_weights(
        placebo_matrix, truth, subjects, SEEDS
    )
    maes = {
        name: patient_equal_mae(real[name], truth, subjects) for name in names
    }
    best_single = min(names, key=lambda name: (maes[name], name))
    return {
        "sources": names,
        "aliases": aliases,
        "weights": weights,
        "bag_weights": bags,
        "placebo_weights": placebo_weights,
        "placebo_bag_weights": placebo_bags,
        "best_single": best_single,
        "tune_patient_equal_mae": maes,
    }


def evaluate_role(frame, truth, current, real, placebo, tuning, target, role, repetitions):
    names = tuple(tuning["sources"])
    real_matrix = stack_matrix(real, names)
    placebo_matrix = stack_matrix(placebo, names)
    predictions = dict(real)
    predictions.update(
        {
            "portfolio": predict_portfolio(real_matrix, tuning["weights"]),
            "placebo_portfolio": predict_portfolio(
                placebo_matrix, tuning["placebo_weights"]
            ),
            "equal_portfolio": real_matrix.mean(axis=1),
            "best_single": real[tuning["best_single"]],
        }
    )
    stacked = real_matrix
    winner = np.argmin(np.abs(stacked - truth[:, None]), axis=1)
    predictions["outcome_oracle"] = stacked[np.arange(len(truth)), winner]
    references = (
        "persistence",
        "release",
        "best_single",
        "equal_portfolio",
        "placebo_portfolio",
    )
    gate = semantic_gate(
        frame,
        predictions,
        truth,
        current,
        target,
        "portfolio",
        references,
        role=role,
        repetitions=repetitions,
    )
    return {
        "patients": int(frame.subject_id.nunique()),
        "mae": {
            name: float(np.mean(np.abs(value - truth)))
            for name, value in predictions.items()
        },
        "patient_equal_mae": {
            name: patient_equal_mae(value, truth, frame.subject_id)
            for name, value in predictions.items()
        },
        "portfolio_gate": gate,
        "oracle_vs_best_single": comparison(
            frame,
            predictions,
            truth,
            "outcome_oracle",
            "best_single",
            salt=f"near-miss:{target}:{role}:oracle",
            repetitions=repetitions,
        ),
        "error_correlation": correlation(real, truth, names),
        "oracle_outcome_aware_not_a_model": True,
    }


def select_data(frame, eligible, roles, cell, args):
    selected = {}
    for role in ("fit", "tune", "patient_test", "hospital_test"):
        selected[role] = select_role_rows(
            frame,
            eligible,
            roles[role],
            patients=args.max_fit_patients if role == "fit" else args.max_test_patients,
            anchors=args.fit_anchors if role == "fit" else 1,
            salt=f"near-miss:{cell}:{role}",
        )
    return selected


def run_cell(frame, assets, arrays, cell, args):
    target, horizon_text = cell.split("@", 1)
    horizon = int(horizon_text[:-1])
    eligible = valid_rows(frame, target, assets.max_age_hours)
    roles, hospital = portfolio_role_masks(frame, eligible, cell)
    rows = select_data(frame, eligible, roles, cell, args)
    support = {
        role: {
            "rows": int(len(index)),
            "patients": int(frame.iloc[index].subject_id.nunique()),
            "subject_digest": None if not len(index) else subject_digest(frame.iloc[index].subject_id),
        }
        for role, index in rows.items()
    }
    if any(item["patients"] < args.min_patients for item in support.values()):
        return {
            "target": target,
            "horizon_hours": horizon,
            "status": "insufficient_support",
            "support": support,
            "hospital_role": hospital,
        }
    prepared = {
        role: prepare_features(frame, index, target, assets, arrays, assets.scaler.normalize_current(frame))
        for role, index in rows.items()
    }
    fit_frame, fit_features = prepared["fit"]
    delta = (fit_frame[f"future_{target}"] - fit_frame[f"{target}_t"]).to_numpy(float)
    models = {
        method: fit_regressor(fit_features[method], delta, fit_frame.subject_id.to_numpy())
        for method in METHODS
    }
    bundles = {}
    for role in ("tune", "patient_test", "hospital_test"):
        role_frame, features = prepared[role]
        current, truth, real, placebo = prediction_bundle(
            role_frame, frame, features, models, assets, arrays, rows[role], cell
        )
        role_frame["_truth"] = truth
        bundles[role] = (role_frame, current, truth, real, placebo)
    tune_frame, _, _, tune_real, tune_placebo = bundles["tune"]
    tuning = tune_portfolios(tune_frame, tune_real, tune_placebo)
    serialized_tuning = {
        "sources": list(tuning["sources"]),
        "aliases": tuning["aliases"],
        "weights": dict(zip(tuning["sources"], tuning["weights"].tolist(), strict=True)),
        "weight_bags": [
            dict(zip(tuning["sources"], row.tolist(), strict=True))
            for row in tuning["bag_weights"]
        ],
        "placebo_weights": dict(
            zip(tuning["sources"], tuning["placebo_weights"].tolist(), strict=True)
        ),
        "placebo_weight_bags": [
            dict(zip(tuning["sources"], row.tolist(), strict=True))
            for row in tuning["placebo_bag_weights"]
        ],
        "best_single": tuning["best_single"],
        "tune_patient_equal_mae": tuning["tune_patient_equal_mae"],
    }
    evaluations = {}
    for role in ("patient_test", "hospital_test"):
        role_frame, current, truth, real, placebo = bundles[role]
        evaluations[role] = evaluate_role(
            role_frame,
            truth,
            current,
            real,
            placebo,
            tuning,
            target,
            role,
            args.bootstrap,
        )
    return {
        "target": target,
        "horizon_hours": horizon,
        "status": "evaluated",
        "support": support,
        "hospital_role": hospital,
        "tuning": serialized_tuning,
        "evaluation": evaluations,
        "portfolio_source_eligible": bool(
            all(item["portfolio_gate"]["pass"] for item in evaluations.values())
        ),
        "oracle_complementarity_present": bool(
            all(item["oracle_vs_best_single"]["pass"] for item in evaluations.values())
        ),
    }


def write_report(report, output):
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(args):
    torch.set_num_threads(args.threads)
    assets = load_release(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = Path("NEAR_MISS_PORTFOLIO_PROTOCOL_20260908.md")
    bindings = {path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()}
    report = {
        "schema": "whole_body_jepa.near_miss_portfolio.v1",
        "complete": False,
        "status": "candidate_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "protocol_sha256": sha256(protocol),
        "release_binding": bindings,
        "input_sha256": sha256(args.eicu),
        "cells": {},
    }
    for horizon in sorted({int(cell.split("@", 1)[1][:-1]) for cell in CELLS}):
        print(f"read eICU {horizon}h", flush=True)
        frame = read_examples(args.eicu, assets, "eicu", horizon, grid=True)
        arrays = _build_inputs(
            input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours),
            assets,
            horizon,
        )
        for cell in [name for name in CELLS if name.endswith(f"@{horizon}h")]:
            print(f"portfolio {cell}", flush=True)
            report["cells"][cell] = run_cell(frame, assets, arrays, cell, args)
            summary = report["cells"][cell]
            print(
                json.dumps(
                    {
                        "cell": cell,
                        "eligible": summary.get("portfolio_source_eligible", False),
                        "oracle": summary.get("oracle_complementarity_present", False),
                    }
                ),
                flush=True,
            )
            write_report(report, args.output)
        del frame, arrays
    report["eligible_cells"] = [
        cell for cell, item in report["cells"].items() if item.get("portfolio_source_eligible")
    ]
    report["oracle_complementary_cells"] = [
        cell for cell, item in report["cells"].items() if item.get("oracle_complementarity_present")
    ]
    report["complete"] = True
    if bindings != {path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()}:
        raise ValueError("release changed during portfolio audit")
    write_report(report, args.output)
    print(
        f"FINISHED: {len(report['eligible_cells'])}/{len(CELLS)} frozen portfolios eligible",
        flush=True,
    )


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
        "--output", type=Path, default=Path("artifacts/near_miss_portfolio_20260908")
    )
    parser.add_argument("--max-fit-patients", type=int, default=5000)
    parser.add_argument("--max-test-patients", type=int, default=1500)
    parser.add_argument("--fit-anchors", type=int, default=4)
    parser.add_argument("--min-patients", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
