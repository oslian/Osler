"""Test state-dependent routing across heterogeneous near-miss forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_near_miss_portfolio import (
    CELLS,
    METHODS,
    REAL_SOURCES,
    PLACEBO_METHOD,
    portfolio_role_masks,
    select_data,
    stack_matrix,
    tune_portfolios,
)
from scripts.audit_change_behavior import frozen_outputs
from scripts.audit_release_masked_cells import input_frame, read_examples
from scripts.screen_cross_variable_history_catalog import (
    comparison,
    fit_regressor,
    prepare_features,
    semantic_gate,
    valid_rows,
)
from whole_body_jepa.multisource_intervals import sha256, subject_digest
from whole_body_jepa.near_miss_portfolio import (
    fit_expected_error_router,
    patient_equal_mae,
    predict_portfolio,
    route_expected_error,
    state_router_features,
)
from whole_body_jepa.runtime import _build_inputs, load_release


def role_predictions(frame, whole_frame, features, models, assets, arrays, rows, cell):
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


def choice_frequency(choice, names):
    return {
        name: float(np.mean(choice == column)) for column, name in enumerate(names)
    }


def evaluate(frame, features, current, truth, real, placebo, tuning, routers, target, role, scale, repetitions):
    names = tuple(tuning["sources"])
    real_matrix = stack_matrix(real, names)
    placebo_matrix = stack_matrix(placebo, names)
    real_features = state_router_features(
        features["own_history"], real_matrix, current, scale=scale
    )
    placebo_features = state_router_features(
        features["own_history"], placebo_matrix, current, scale=scale
    )
    routed, choice, expected = route_expected_error(
        routers["real"], real_features, real_matrix
    )
    placebo_routed, placebo_choice, placebo_expected = route_expected_error(
        routers["placebo"], placebo_features, placebo_matrix
    )
    fixed = predict_portfolio(real_matrix, tuning["weights"])
    predictions = dict(real)
    predictions.update(
        {
            "state_router": routed,
            "placebo_state_router": placebo_routed,
            "fixed_portfolio": fixed,
            "best_single": real[tuning["best_single"]],
        }
    )
    winner = np.argmin(np.abs(real_matrix - truth[:, None]), axis=1)
    predictions["outcome_oracle"] = real_matrix[np.arange(len(truth)), winner]
    gate = semantic_gate(
        frame,
        predictions,
        truth,
        current,
        target,
        "state_router",
        (
            "persistence",
            "release",
            "best_single",
            "fixed_portfolio",
            "placebo_state_router",
        ),
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
        "state_router_gate": gate,
        "source_frequency": choice_frequency(choice, names),
        "placebo_source_frequency": choice_frequency(placebo_choice, names),
        "mean_expected_error": {
            name: float(expected[:, column].mean())
            for column, name in enumerate(names)
        },
        "placebo_mean_expected_error": {
            name: float(placebo_expected[:, column].mean())
            for column, name in enumerate(names)
        },
        "oracle_agreement": float(np.mean(choice == winner)),
        "oracle_outcome_aware_not_a_model": True,
        "oracle_vs_router": comparison(
            frame,
            predictions,
            truth,
            "outcome_oracle",
            "state_router",
            salt=f"near-miss-router:{target}:{role}:oracle",
            repetitions=repetitions,
        ),
    }


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
        return {"status": "insufficient_support", "support": support}
    observations = assets.scaler.normalize_current(frame)
    prepared = {
        role: prepare_features(frame, index, target, assets, arrays, observations)
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
        values = role_predictions(
            role_frame, frame, features, models, assets, arrays, rows[role], cell
        )
        bundles[role] = (role_frame, features, *values)
    tune_frame, tune_features, tune_current, tune_truth, tune_real, tune_placebo = bundles["tune"]
    tune_frame["_truth"] = tune_truth
    tuning = tune_portfolios(tune_frame, tune_real, tune_placebo)
    names = tuple(tuning["sources"])
    scale = float(assets.scaler.scales[assets.checkpoint["variables"].index(target)])
    real_matrix = stack_matrix(tune_real, names)
    placebo_matrix = stack_matrix(tune_placebo, names)
    routers = {
        "real": fit_expected_error_router(
            state_router_features(
                tune_features["own_history"], real_matrix, tune_current, scale=scale
            ),
            real_matrix,
            tune_truth,
            tune_frame.subject_id,
        ),
        "placebo": fit_expected_error_router(
            state_router_features(
                tune_features["own_history"], placebo_matrix, tune_current, scale=scale
            ),
            placebo_matrix,
            tune_truth,
            tune_frame.subject_id,
        ),
    }
    evaluations = {}
    for role in ("patient_test", "hospital_test"):
        role_frame, features, current, truth, real, placebo = bundles[role]
        evaluations[role] = evaluate(
            role_frame,
            features,
            current,
            truth,
            real,
            placebo,
            tuning,
            routers,
            target,
            role,
            scale,
            args.bootstrap,
        )
    return {
        "target": target,
        "horizon_hours": horizon,
        "status": "evaluated",
        "support": support,
        "hospital_role": hospital,
        "sources": list(names),
        "fixed_weights": dict(zip(names, tuning["weights"].tolist(), strict=True)),
        "best_single": tuning["best_single"],
        "evaluation": evaluations,
        "state_router_source_eligible": bool(
            all(item["state_router_gate"]["pass"] for item in evaluations.values())
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
    protocol = Path("NEAR_MISS_STATE_ROUTING_PROTOCOL_20260908.md")
    bindings = {path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()}
    report = {
        "schema": "whole_body_jepa.near_miss_state_router.v1",
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
            print(f"state route {cell}", flush=True)
            report["cells"][cell] = run_cell(frame, assets, arrays, cell, args)
            print(
                json.dumps(
                    {
                        "cell": cell,
                        "eligible": report["cells"][cell].get(
                            "state_router_source_eligible", False
                        ),
                    }
                ),
                flush=True,
            )
            write_report(report, args.output)
        del frame, arrays
    report["eligible_cells"] = [
        cell
        for cell, item in report["cells"].items()
        if item.get("state_router_source_eligible")
    ]
    report["complete"] = True
    if bindings != {path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()}:
        raise ValueError("release changed during state-routing audit")
    write_report(report, args.output)
    print(
        f"FINISHED: {len(report['eligible_cells'])}/{len(CELLS)} state routers eligible",
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
        "--output", type=Path, default=Path("artifacts/near_miss_state_router_20260908")
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
