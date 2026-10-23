"""MIMIC-only selection of online point-forecast recalibration under simulated shift.

Same deployment scenarios and shared-clock replay as ``site_adaptive_interval_dev``
(S1 time, S2 cardiac care units; warm-up stays are not scored).  The frozen interval method
(``weight_site_unit_0.02``) runs on top of each corrected forecast.  The
recipe's comparator receives the same correction, so the comparator gate stays
a like-for-like test of the recipe.

Candidates: none; site offset; site offset + gain; site offset + gain +
care-unit offset (eta / 2); eta in {0.005, 0.01, 0.02}.

Predeclared selection rule (written before any candidate was scored): maximize
the number of streams with a full pass (patient-axis point gate against
persistence and the corrected comparator, plus patient- and care-unit-equal
coverage in [0.87, 0.93]); ties -> lower median MAE relative to no
correction.  The winner is adopted only if its full passes are not fewer than
no correction's and its median MAE ratio is below 1.  No eICU or NWICU data
is read.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import (
    cap_by_subject,
    fit_continuous_recipe,
    group_equal_coverage,
    normalize_cohort,
    paired_comparison,
    stable_seed,
)
from osler_jepa.interval_scale import fit_interval_scale
from osler_jepa.patient_split import selection_patients
from osler_jepa.site_adaptive_conformal import adaptive_stream_widths, covariate_shift_weights
from osler_jepa.site_adaptive_point import PointCorrectionConfig, online_point_correction
from site_adaptive_external_audit import FROZEN_CONFIG
from site_adaptive_interval_dev import BOUNDS, SCENARIOS, STREAM_ROWS, Replay, covariate_matrix, replay
from target_specific_domain_conformal_audit import _safe_float


def candidates() -> dict[str, PointCorrectionConfig]:
    output = {"none": PointCorrectionConfig()}
    for eta in (0.005, 0.01, 0.02):
        output[f"offset_{eta}"] = PointCorrectionConfig(eta_offset=eta)
        output[f"affine_{eta}"] = PointCorrectionConfig(eta_offset=eta, eta_gain=eta)
        output[f"affine_unit_{eta}"] = PointCorrectionConfig(eta_offset=eta, eta_gain=eta, level_etas={"care_unit": eta / 2})
    return output


def scenario_streams(full, years, *, cell, target, horizon, source):
    """Yield (scenario name, stream inputs) for the MIMIC deployment scenarios."""

    full = full.assign(anchor_year_group=full["subject_id"].map(years).fillna("unknown"))
    for name, (column, source_values, target_values) in SCENARIOS.items():
        in_target = full[column].isin(target_values).to_numpy()
        in_source = full[column].isin(source_values).to_numpy() if source_values else ~in_target
        source_frame = cap_by_subject(full[in_source], 15000, stable_seed(cell, name, "source"))
        target_frame = cap_by_subject(full[in_target], STREAM_ROWS, stable_seed(cell, name, "target"))
        source_frame = source_frame.assign(subject_id="SRC:" + source_frame["subject_id"].astype(str))
        target_frame = target_frame.assign(subject_id="TGT:" + target_frame["subject_id"].astype(str))
        frame = pd.concat([source_frame, target_frame], ignore_index=True, sort=False)
        frame.sort_values(["subject_id", "anchor_time"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        subject = frame["subject_id"].astype(str)
        train, test = np.flatnonzero(subject.str.startswith("SRC:")), np.flatnonzero(subject.str.startswith("TGT:"))
        seed = stable_seed(cell, "mimic_to_eicu")
        fitted = fit_continuous_recipe(frame, train, test, target=target, source=source, seed=seed)
        future, current = _safe_float(frame, f"future_{target}"), _safe_float(frame, f"{target}_t")
        held_out = fitted["calibration_rows"]
        residual = np.abs(fitted["calibration_prediction"] - future[held_out])
        choosing = selection_patients(frame.iloc[held_out]["subject_id"].to_numpy())
        sel, cal = held_out[choosing], held_out[~choosing]       # scales on selection patients, scores on conformal patients
        scale = fit_interval_scale("quintile", current[sel], np.full(len(sel), np.nan), residual[choosing],
                                   frame.iloc[sel]["subject_id"].to_numpy())
        stream = {
            "target": target,
            "calibration_frame": frame.iloc[cal],
            "cal_covariates": covariate_matrix(frame, cal, target, frame.iloc[cal]),
            "frame": frame,
            "seed": seed,
            "persistence_comparator": fitted["comparator_name"] == "persistence_matched_baseline",
            "candidate_of": pd.Series(fitted["test_prediction"], index=fitted["test_rows"]),
            "comparator_of": pd.Series(np.asarray(fitted["comparator"], dtype=float), index=fitted["test_rows"]),
            "future": future,
            "current": current,
            "point_scale": float(np.median(residual[choosing])),
            "cal_scores": residual[~choosing] / scale(current[cal], np.full(len(cal), np.nan)),
            "scale_of_value": lambda values, scale=scale: scale(values, np.full(len(values), np.nan)),
            "test_rows": fitted["test_rows"],
        }
        yield name, deployment_view(stream, replay(frame, fitted["test_rows"], target, stable_seed(cell, name, "clock")))


def deployment_view(stream: dict, deployment: Replay) -> dict:
    """Stream inputs for one deployment, in clock order; ``scored`` marks non-warm-up rows."""

    frame, rows = stream["frame"], deployment.rows
    now = stream["current"][rows]
    return {
        **stream,
        "deployment": deployment,
        "rows": rows,
        "scored": ~deployment.warm,
        "candidate": stream["candidate_of"][rows].to_numpy(),
        "comparator": stream["comparator_of"][rows].to_numpy(),
        "observed": stream["future"][rows],
        "now": now,
        "weights": covariate_shift_weights(stream["cal_covariates"],
                                           covariate_matrix(frame, deployment.day_one, stream["target"], stream["calibration_frame"])),
        "scales": stream["scale_of_value"](now),
        "common": deployment.common(frame, {"care_unit": "careunit"}),
    }


def score_stream(stream: dict, prediction: np.ndarray, comparator: np.ndarray | None, reference_mae: float | None) -> dict:
    """Point gate (patient axis vs persistence and, if given, the comparator) plus frozen adaptive interval."""

    frame, observed, now, scored = stream["frame"], stream["observed"], stream["now"], stream["scored"]
    widths = adaptive_stream_widths(calibration_scores=stream["cal_scores"], calibration_weights=stream["weights"],
                                    scales=stream["scales"], predictions=prediction, observed=observed,
                                    config=FROZEN_CONFIG, **stream["common"]).widths[scored]
    rows, loss = stream["rows"][scored], np.abs(prediction - observed)[scored]
    others = [np.abs(now - observed)[scored]] + ([np.abs(comparator - observed)[scored]] if comparator is not None else [])
    point = all(paired_comparison(frame, rows, loss, other, seed=stream["seed"])["patient"]["pass"] for other in others)
    covered = loss <= widths
    patient = group_equal_coverage(frame, rows, covered, "subject_id")
    unit = group_equal_coverage(frame, rows, covered, "careunit")
    mae = float(np.mean(loss))
    return {
        "point_pass": bool(point),
        "patient_equal_coverage": patient,
        "careunit_equal_coverage": unit,
        "full_pass": bool(point and BOUNDS[0] <= patient <= BOUNDS[1] and BOUNDS[0] <= unit <= BOUNDS[1]),
        "mae": mae,
        "mae_ratio_vs_none": mae / (reference_mae or mae),
    }


def corrected_experts(stream: dict, config: PointCorrectionConfig) -> tuple[np.ndarray, np.ndarray]:
    corrected = online_point_correction(current=stream["now"], prediction=stream["candidate"], observed=stream["observed"],
                                        scale=stream["point_scale"], config=config, **stream["common"])
    comparator = stream["now"] if stream["persistence_comparator"] else online_point_correction(
        current=stream["now"], prediction=stream["comparator"], observed=stream["observed"],
        scale=stream["point_scale"], config=config, **stream["common"])
    return corrected, comparator


def run_cell(full, years, *, cell, target, horizon, source) -> dict:
    results = {}
    for name, stream in scenario_streams(full, years, cell=cell, target=target, horizon=horizon, source=source):
        scenario, reference = {}, None
        for label, config in candidates().items():
            corrected, comparator = corrected_experts(stream, config)
            scenario[label] = score_stream(stream, corrected, comparator, reference)
            reference = reference or scenario[label]["mae"]
        results[name] = {"stream_rows": int(len(stream["rows"])), "scored_rows": int(stream["scored"].sum()), "candidates": scenario}
    return results


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = [row for row in development["cells"] if row["task_type"] != "binary_event_probability"]
    patients = pd.read_csv(args.patients, usecols=["subject_id", "anchor_year_group"])
    years = pd.Series(patients["anchor_year_group"].to_numpy(), index="MIMIC:" + patients["subject_id"].astype(str))
    by_horizon: dict[int, list[dict]] = {}
    for row in contracts:
        by_horizon.setdefault(int(row["horizon_hours"]), []).append(row)
    cells = []
    for horizon in sorted(by_horizon):
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        for contract in sorted(by_horizon[horizon], key=lambda row: row["target"]):
            cell, target = contract["cell"], contract["target"]
            print(f"point-correction dev {cell}", flush=True)
            full, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
            full = full.assign(first_careunit=full["careunit"])
            cells.append({"cell": cell, **run_cell(full, years, cell=cell, target=target, horizon=horizon, source=contract["source_model"])})
        del raw
    names = list(candidates())
    summary = {}
    for label in names:
        streams = [cell[s]["candidates"][label] for cell in cells for s in SCENARIOS]
        summary[label] = {
            "full_pass_streams": int(sum(item["full_pass"] for item in streams)),
            "point_pass_streams": int(sum(item["point_pass"] for item in streams)),
            "streams": len(streams),
            "median_mae_ratio_vs_none": float(np.median([item["mae_ratio_vs_none"] for item in streams])),
        }
    ranked = sorted(names, key=lambda label: (-summary[label]["full_pass_streams"], summary[label]["median_mae_ratio_vs_none"]))
    winner = ranked[0]
    adopted = (
        winner
        if winner != "none"
        and summary[winner]["full_pass_streams"] >= summary["none"]["full_pass_streams"]
        and summary[winner]["median_mae_ratio_vs_none"] < 1.0
        else "none"
    )
    output = {
        "schema": "site_adaptive_point_dev.v2",
        "database": "MIMIC-IV only",
        "selection_rule": "most full-pass streams; ties -> lower median MAE ratio; adopt only if not fewer full passes than none and MAE ratio < 1",
        "summary": summary,
        "ranking": ranked,
        "selected": adopted,
        "cells": cells,
        "external_data_read": False,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", type=Path, default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"))
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--patients", type=Path, default=Path.home() / "Downloads/mimic-iv-3.1/hosp/patients.csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"selected": result["selected"], "summary": result["summary"]}, indent=1))
