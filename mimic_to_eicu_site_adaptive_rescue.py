"""Rescue cross-database cells with leakage-safe site adaptation.

Point recipes are fit on MIMIC only.  For each of seven deterministic splits,
a small set of eICU hospitals may select a predeclared source and calibrate a
target-specific residual scale.  At least ten untouched eICU hospitals form
the evaluation set.  No evaluation label can affect source or interval choice.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from mimic_to_eicu_49_cell_external_audit import (
    COVERAGE_BOUNDS,
    choose_eicu_cohort,
    feature_subset,
    fit_ridge_source,
    group_equal_coverage,
    hospital_bootstrap,
    mimic_path,
    normalize_cohort,
    paired_comparison,
    stable_seed,
)
from osler_jepa.treatment_units import treatment_feature_actions
from target_specific_domain_conformal_audit import (
    TARGET_CHANGE_THRESHOLDS,
    _exclude_sibling_features,
    _feature_frame,
    _fit_hurdle,
    _fit_transform,
    _patient_bootstrap,
    _safe_float,
    _split_fit_calibration,
)
from teacher_anchored_joint_jepa_audit import (
    _patient_cluster_conformal_quantile,
)


SPLIT_SEEDS = (7, 11, 19, 23, 37, 53, 71)
MIN_TEST_HOSPITALS = 10
MIN_ADAPTATION_SUBJECTS = 30
ADAPTIVE_COVERAGE_GRID = tuple(np.round(np.arange(0.75, 0.961, 0.01), 2))
SCALE_CONTEXT_TARGETS = (
    "heart_rate",
    "map",
    "respiratory_rate",
    "o2sat",
    "glucose",
    "creatinine",
    "bun",
    "sodium",
    "potassium",
    "bicarbonate",
    "hemoglobin",
    "platelets",
)


def rescue_gate(*, source_rescue: bool, point_pass: bool, interval_pass: bool) -> bool:
    """Apply only the evidence gate that the rescue experiment changes.

    Interval-only cells already passed the frozen zero-shot point gate.  Source
    rescue cells change the point predictor, so they must re-pass both gates.
    """

    return bool(interval_pass and (point_pass or not source_rescue))


def split_adaptation_hospitals(frame: pd.DataFrame, seed: int) -> tuple[set[str], set[str]]:
    hospitals = np.asarray(sorted(frame["hospitalid"].astype(str).unique()), dtype=object)
    if len(hospitals) < MIN_TEST_HOSPITALS + 2:
        return set(), set()
    np.random.default_rng(seed).shuffle(hospitals)
    adaptation_count = max(2, int(np.ceil(0.20 * len(hospitals))))
    adaptation_count = min(adaptation_count, len(hospitals) - MIN_TEST_HOSPITALS)
    return set(map(str, hospitals[:adaptation_count])), set(map(str, hospitals[adaptation_count:]))


def split_adaptation_subjects(
    frame: pd.DataFrame,
    rows: np.ndarray,
    seed: int,
    *,
    source_selection: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = np.asarray(
        sorted(frame.iloc[rows]["subject_id"].astype(str).unique()), dtype=object
    )
    np.random.default_rng(seed).shuffle(subjects)
    if source_selection:
        parts = np.array_split(subjects, 3)
        selection_subjects, scale_subjects, conformal_subjects = map(set, parts)
    else:
        parts = np.array_split(subjects, 2)
        selection_subjects = set()
        scale_subjects, conformal_subjects = map(set, parts)
    selected_subject = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    selection = rows[np.isin(selected_subject, list(selection_subjects))]
    scale = rows[np.isin(selected_subject, list(scale_subjects))]
    conformal = rows[np.isin(selected_subject, list(conformal_subjects))]
    return selection, scale, conformal


def patient_equal_mae(
    frame: pd.DataFrame,
    rows: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    error = np.abs(prediction - truth)
    return float(np.mean([error[subjects == item].mean() for item in np.unique(subjects)]))


def common_treatment_columns(
    mimic: pd.DataFrame,
    eicu: pd.DataFrame,
    target: str,
) -> list[str]:
    permitted = treatment_feature_actions(target)
    columns = []
    for column in sorted(set(mimic.columns) & set(eicu.columns)):
        if not column.startswith("hist_"):
            continue
        if not any(
            column == f"hist_{action}" or column.startswith(f"hist_{action}_")
            for action in permitted
        ):
            continue
        mimic_values = pd.to_numeric(mimic[column], errors="coerce").fillna(0.0)
        eicu_values = pd.to_numeric(eicu[column], errors="coerce").fillna(0.0)
        if int((mimic_values != 0.0).sum()) >= 30 and int((eicu_values != 0.0).sum()) >= 30:
            columns.append(column)
    return columns


def fit_ridge_columns(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    target: str,
    columns: list[str],
    seed: int,
) -> dict:
    fit_rows, calibration_rows = _split_fit_calibration(frame, train_rows, seed)
    selected = _exclude_sibling_features(features[columns], target)
    current = _safe_float(frame, f"{target}_t")
    future = _safe_float(frame, f"future_{target}")
    valid = np.isfinite(current) & np.isfinite(future)
    fit_rows = fit_rows[valid[fit_rows]]
    calibration_rows = calibration_rows[valid[calibration_rows]]
    test_rows = test_rows[valid[test_rows]]
    transform = _fit_transform(selected, fit_rows)
    x_fit = transform.apply(selected.iloc[fit_rows])
    x_test = transform.apply(selected.iloc[test_rows])
    delta = future[fit_rows] - current[fit_rows]
    limit = max(float(np.quantile(np.abs(delta), 0.99)), 1e-6)
    model = Ridge(alpha=25.0).fit(x_fit, delta)
    prediction = current[test_rows] + np.clip(model.predict(x_test), -limit, limit)
    return {
        "test_rows": test_rows,
        "test_prediction": prediction,
        "feature_count": int(len(transform.columns)),
    }


def prediction_sources(
    mimic: pd.DataFrame,
    eicu: pd.DataFrame,
    *,
    target: str,
    current_source: str,
    allow_source_rescue: bool,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    frame = pd.concat([mimic, eicu], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    train_rows = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("MIMIC:"))
    test_rows = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("EICU:"))
    features = _feature_frame(frame)
    sources: dict[str, np.ndarray] = {}
    metadata: dict = {}

    measurement = fit_ridge_source(
        frame,
        features,
        train_rows,
        test_rows,
        target=target,
        source="measurement_pure_ridge",
        seed=seed,
    )
    rows = measurement["test_rows"]
    sources["measurement_pure_ridge"] = measurement["test_prediction"]
    metadata["measurement_pure_ridge"] = {"feature_count": measurement["feature_count"]}

    if current_source == "target_specific_change_hurdle" or allow_source_rescue:
        if target in TARGET_CHANGE_THRESHOLDS:
            hurdle_rows, prediction, _, _, _, policy = _fit_hurdle(
                frame,
                features,
                train_rows,
                test_rows,
                seed=seed,
                target=target,
                change_threshold=float(TARGET_CHANGE_THRESHOLDS[target]),
            )
            if np.array_equal(rows, hurdle_rows):
                sources["target_specific_change_hurdle"] = prediction
                metadata["target_specific_change_hurdle"] = {"mimic_policy": policy}

    shared_treatment = common_treatment_columns(mimic, eicu, target)
    if current_source == "observed_treatment_context_ridge" or allow_source_rescue:
        if shared_treatment:
            physiology = feature_subset(features, "measurement_pure_ridge", target)
            treatment_columns = [column for column in shared_treatment if column in features]
            selected_columns = list(dict.fromkeys([*physiology.columns, *treatment_columns]))
            treatment = fit_ridge_columns(
                frame,
                features,
                train_rows,
                test_rows,
                target=target,
                columns=selected_columns,
                seed=seed,
            )
            if np.array_equal(rows, treatment["test_rows"]):
                sources["common_schema_treatment_ridge"] = treatment["test_prediction"]
                metadata["common_schema_treatment_ridge"] = {
                    "feature_count": treatment["feature_count"],
                    "shared_treatment_columns": treatment_columns,
                }

    source_alias = {
        "observed_treatment_context_ridge": "common_schema_treatment_ridge",
    }.get(current_source, current_source)
    fixed_source = (
        source_alias if source_alias in sources else "measurement_pure_ridge"
    )
    sources["persistence"] = _safe_float(frame, f"{target}_t")[rows]
    return frame, sources, {
        "rows": rows,
        "source_metadata": metadata,
        "fixed_source": fixed_source,
    }


def scale_features(
    frame: pd.DataFrame,
    rows: np.ndarray,
    prediction: np.ndarray,
    target: str,
) -> np.ndarray:
    current = _safe_float(frame, f"{target}_t")[rows]
    age = np.log1p(np.clip(_safe_float(frame, f"{target}_age_hr")[rows], 0.0, 168.0))
    onset = np.log1p(np.clip(_safe_float(frame, "hours_since_onset")[rows], 0.0, 720.0))
    columns = [
        current,
        prediction,
        np.abs(prediction - current),
        np.nan_to_num(age, nan=np.log1p(168.0)),
        np.nan_to_num(onset, nan=0.0),
    ]
    for context_target in SCALE_CONTEXT_TARGETS:
        value_column = f"{context_target}_t"
        age_column = f"{context_target}_age_hr"
        if value_column in frame:
            values = _safe_float(frame, value_column)
            if np.isfinite(values).any():
                columns.append(values[rows])
        if age_column in frame:
            ages = _safe_float(frame, age_column)
            if np.isfinite(ages).any():
                columns.append(np.log1p(np.clip(ages[rows], 0.0, 168.0)))
    careunit = frame.iloc[rows]["careunit"].fillna("unknown").astype(str)
    for category in sorted(frame["careunit"].fillna("unknown").astype(str).unique()):
        columns.append((careunit == category).to_numpy(dtype=np.float64))
    return np.column_stack(columns)


def hospital_cv_conformal_quantile(
    frame: pd.DataFrame,
    rows: np.ndarray,
    residuals: np.ndarray,
) -> tuple[float, dict]:
    """Tune conformal coverage by leave-one-adaptation-hospital-out CV."""

    rows = np.asarray(rows, dtype=np.int64)
    scores = np.asarray(residuals, dtype=np.float64)
    finite = np.isfinite(scores)
    rows = rows[finite]
    scores = scores[finite]
    hospitals = frame.iloc[rows]["hospitalid"].astype(str).to_numpy()
    unique_hospitals = np.unique(hospitals)
    if len(rows) < 90 or len(unique_hospitals) < 4:
        quantile, _, metadata = _patient_cluster_conformal_quantile(
            frame, rows, scores, coverage=0.90
        )
        return quantile, {**metadata, "method": "hospital_cv_fallback"}

    diagnostics = {}
    for nominal in ADAPTIVE_COVERAGE_GRID:
        crossfit_covered = np.zeros(len(rows), dtype=bool)
        available = np.zeros(len(rows), dtype=bool)
        for hospital in unique_hospitals:
            heldout = hospitals == hospital
            fit = ~heldout
            if int(fit.sum()) < 30 or int(heldout.sum()) < 10:
                continue
            quantile, _, _ = _patient_cluster_conformal_quantile(
                frame, rows[fit], scores[fit], coverage=float(nominal)
            )
            crossfit_covered[heldout] = scores[heldout] <= quantile
            available[heldout] = True
        if int(available.sum()) < 30:
            continue
        selected_rows = rows[available]
        selected_covered = crossfit_covered[available]
        coverage = {
            "patient": group_equal_coverage(
                frame, selected_rows, selected_covered, "subject_id"
            ),
            "hospital": group_equal_coverage(
                frame, selected_rows, selected_covered, "hospitalid"
            ),
            "careunit": group_equal_coverage(
                frame, selected_rows, selected_covered, "careunit"
            ),
            **time_coverage(frame, selected_rows, selected_covered),
        }
        deviations = [abs(value - 0.90) for value in coverage.values()]
        diagnostics[float(nominal)] = {
            "coverage": coverage,
            "max_deviation": float(max(deviations)),
            "mean_deviation": float(np.mean(deviations)),
        }
    if not diagnostics:
        quantile, _, metadata = _patient_cluster_conformal_quantile(
            frame, rows, scores, coverage=0.90
        )
        return quantile, {**metadata, "method": "hospital_cv_fallback"}
    selected = min(
        diagnostics,
        key=lambda nominal: (
            diagnostics[nominal]["max_deviation"],
            diagnostics[nominal]["mean_deviation"],
            abs(nominal - 0.90),
        ),
    )
    quantile, rank, metadata = _patient_cluster_conformal_quantile(
        frame, rows, scores, coverage=float(selected)
    )
    return quantile, {
        **metadata,
        "method": "leave_one_hospital_out_adaptive_patient_cluster_conformal",
        "selected_nominal_coverage": float(selected),
        "q_rank": int(rank),
        "hospital_cv_diagnostic": diagnostics[selected],
        "adaptation_hospitals": int(len(unique_hospitals)),
        "test_labels_used": False,
    }


def fit_scale_head(features: np.ndarray, residual: np.ndarray, seed: int) -> dict:
    matrix = np.asarray(features, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    medians = np.nanmedian(matrix, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    lower = np.quantile(matrix, 0.005, axis=0)
    upper = np.quantile(matrix, 0.995, axis=0)
    matrix = np.clip(matrix, lower, upper)
    positive = residual[np.isfinite(residual) & (residual > 0.0)]
    floor = max(float(np.quantile(positive, 0.02)), 1e-6)
    ceiling = max(float(np.quantile(positive, 0.98)), floor)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=100,
        max_leaf_nodes=7,
        min_samples_leaf=50,
        l2_regularization=20.0,
        random_state=seed,
    ).fit(matrix, np.log(np.maximum(residual, floor)))
    return {
        "model": model,
        "medians": medians,
        "lower": lower,
        "upper": upper,
        "floor": floor,
        "ceiling": ceiling,
    }


def predict_scale(model: dict, features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, model["medians"])
    matrix = np.clip(matrix, model["lower"], model["upper"])
    prediction = np.exp(model["model"].predict(matrix))
    return np.clip(prediction, model["floor"], model["ceiling"])


def time_coverage(frame: pd.DataFrame, rows: np.ndarray, covered: np.ndarray) -> dict:
    times = pd.to_datetime(frame.iloc[rows]["anchor_time"], errors="coerce")
    threshold = times.quantile(0.5)
    early = (times <= threshold).to_numpy()
    late = ~early
    return {
        "early": group_equal_coverage(frame, rows[early], covered[early], "subject_id"),
        "late": group_equal_coverage(frame, rows[late], covered[late], "subject_id"),
    }


def evaluate_split(
    frame: pd.DataFrame,
    sources: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    target: str,
    source_rescue: bool,
    fixed_source: str,
    seed: int,
) -> dict:
    hospitals = frame.iloc[rows]["hospitalid"].astype(str).to_numpy()
    adaptation_hospitals, evaluation_hospitals = split_adaptation_hospitals(
        frame.iloc[rows], seed
    )
    if not adaptation_hospitals:
        return {"available": False, "reason": "insufficient_hospitals"}
    adaptation_rows = rows[np.isin(hospitals, list(adaptation_hospitals))]
    evaluation_rows = rows[np.isin(hospitals, list(evaluation_hospitals))]
    selection_rows, scale_rows, conformal_rows = split_adaptation_subjects(
        frame,
        adaptation_rows,
        seed + 1,
        source_selection=source_rescue,
    )
    if min(len(scale_rows), len(conformal_rows), len(evaluation_rows)) < 30:
        return {"available": False, "reason": "insufficient_adaptation_rows"}
    truth_all = _safe_float(frame, f"future_{target}")[rows]
    row_position = {int(row): index for index, row in enumerate(rows)}

    def values(name: str, selected_rows: np.ndarray) -> np.ndarray:
        positions = np.asarray([row_position[int(row)] for row in selected_rows], dtype=np.int64)
        return sources[name][positions]

    selectable = list(sources)
    if source_rescue:
        truth_selection = _safe_float(frame, f"future_{target}")[selection_rows]
        selected_source = min(
            selectable,
            key=lambda name: patient_equal_mae(
                frame,
                selection_rows,
                values(name, selection_rows),
                truth_selection,
            ),
        )
    else:
        selected_source = fixed_source

    scale_prediction = values(selected_source, scale_rows)
    scale_truth = _safe_float(frame, f"future_{target}")[scale_rows]
    scale_model = fit_scale_head(
        scale_features(frame, scale_rows, scale_prediction, target),
        np.abs(scale_prediction - scale_truth),
        seed + 2,
    )
    conformal_prediction = values(selected_source, conformal_rows)
    conformal_truth = _safe_float(frame, f"future_{target}")[conformal_rows]
    conformal_scale = predict_scale(
        scale_model,
        scale_features(frame, conformal_rows, conformal_prediction, target),
    )
    normalized_residual = np.abs(conformal_prediction - conformal_truth) / np.maximum(
        conformal_scale, 1e-8
    )
    q90, metadata = hospital_cv_conformal_quantile(
        frame, conformal_rows, normalized_residual
    )
    candidate = values(selected_source, evaluation_rows)
    truth = _safe_float(frame, f"future_{target}")[evaluation_rows]
    persistence = values("persistence", evaluation_rows)
    candidate_loss = np.abs(candidate - truth)
    persistence_loss = np.abs(persistence - truth)
    versus_persistence = paired_comparison(
        frame,
        evaluation_rows,
        candidate_loss,
        persistence_loss,
        seed=seed + 3,
    )
    point_pass = bool(
        selected_source != "persistence"
        and versus_persistence["patient"]["pass"]
        and versus_persistence["hospital"]["pass"]
    )
    if selected_source not in {"persistence", "measurement_pure_ridge"}:
        measurement = values("measurement_pure_ridge", evaluation_rows)
        versus_measurement = paired_comparison(
            frame,
            evaluation_rows,
            candidate_loss,
            np.abs(measurement - truth),
            seed=seed + 5,
        )
        point_pass = bool(
            point_pass
            and versus_measurement["patient"]["pass"]
            and versus_measurement["hospital"]["pass"]
        )
    else:
        versus_measurement = None
    widths = q90 * predict_scale(
        scale_model,
        scale_features(frame, evaluation_rows, candidate, target),
    )
    covered = candidate_loss <= widths
    patient_coverage = group_equal_coverage(
        frame, evaluation_rows, covered, "subject_id"
    )
    hospital_coverage = group_equal_coverage(
        frame, evaluation_rows, covered, "hospitalid"
    )
    careunit_coverage = group_equal_coverage(
        frame, evaluation_rows, covered, "careunit"
    )
    time = time_coverage(frame, evaluation_rows, covered)
    coverage_values = [patient_coverage, hospital_coverage, careunit_coverage, *time.values()]
    interval_pass = all(COVERAGE_BOUNDS[0] <= value <= COVERAGE_BOUNDS[1] for value in coverage_values)
    return {
        "available": True,
        "adaptation_hospitals": len(adaptation_hospitals),
        "evaluation_hospitals": len(evaluation_hospitals),
        "adaptation_rows": int(len(adaptation_rows)),
        "evaluation_rows": int(len(evaluation_rows)),
        "selected_source": selected_source,
        "candidate_mae": float(candidate_loss.mean()),
        "persistence_mae": float(persistence_loss.mean()),
        "versus_persistence": versus_persistence,
        "versus_measurement_pure": versus_measurement,
        "interval": {
            "median_half_width": float(np.median(widths)),
            "patient_equal_coverage": patient_coverage,
            "hospital_equal_coverage": hospital_coverage,
            "careunit_equal_coverage": careunit_coverage,
            "time_coverage": time,
            "normalized_q90": float(q90),
            "calibration": metadata,
        },
        "point_pass": point_pass,
        "interval_pass": interval_pass,
        "full_pass": rescue_gate(
            source_rescue=source_rescue,
            point_pass=point_pass,
            interval_pass=interval_pass,
        ),
    }


def audit_cell(
    contract: dict,
    original: dict,
    mimic: pd.DataFrame,
    eicu: pd.DataFrame,
) -> dict:
    cell = contract["cell"]
    target = contract["target"]
    seed = stable_seed(cell, "site_adaptive")
    source_rescue = not bool(original["point_pass"])
    frame, sources, metadata = prediction_sources(
        mimic,
        eicu,
        target=target,
        current_source=contract["source_model"],
        allow_source_rescue=source_rescue,
        seed=seed,
    )
    reports = [
        evaluate_split(
            frame,
            sources,
            metadata["rows"],
            target=target,
            source_rescue=source_rescue,
            fixed_source=metadata["fixed_source"],
            seed=split_seed,
        )
        for split_seed in SPLIT_SEEDS
    ]
    available = [report for report in reports if report.get("available")]
    source_counts = Counter(report["selected_source"] for report in available)
    return {
        "cell": cell,
        "target": target,
        "horizon_hours": contract["horizon_hours"],
        "original_source": contract["source_model"],
        "original_point_pass": bool(original["point_pass"]),
        "original_full_external_pass": bool(original["full_external_pass"]),
        "rescue_type": "source_and_interval" if source_rescue else "interval_only",
        "source_candidates": sorted(sources),
        "source_metadata": metadata["source_metadata"],
        "gate_policy": (
            "new_source_must_pass_point_and_interval"
            if source_rescue
            else "frozen_zero_shot_point_evidence_plus_new_interval"
        ),
        "split_reports": reports,
        "evaluated_splits": len(available),
        "point_pass_splits": sum(report["point_pass"] for report in available),
        "interval_pass_splits": sum(report["interval_pass"] for report in available),
        "full_pass_splits": sum(report["full_pass"] for report in available),
        "selected_source_counts": dict(source_counts),
        "site_adaptive_validated": bool(
            len(available) == len(SPLIT_SEEDS)
            and all(report["full_pass"] for report in available)
        ),
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# MIMIC-to-eICU Site-Adaptive Rescue",
        "",
        f"- Frozen independent-database full passes retained: `{report['original_full_pass_cells']}`",
        f"- Rescue cells attempted: `{report['attempted_cells']}`",
        f"- New 7/7 site-adaptive passes: `{report['new_site_adaptive_pass_cells']}`",
        f"- Combined usable cells: `{report['combined_usable_cells']}`",
        "",
        "| Cell | Rescue | 7-split point | 7-split interval | 7-split full | Sources | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in report["cells"]:
        lines.append(
            f"| {cell['cell']} | {cell['rescue_type']} | {cell['point_pass_splits']}/7 | "
            f"{cell['interval_pass_splits']}/7 | {cell['full_pass_splits']}/7 | "
            f"{cell['selected_source_counts']} | {'pass' if cell['site_adaptive_validated'] else 'fail'} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "This is site-adaptive validation, not frozen zero-shot transfer. eICU adaptation hospitals may select a predeclared MIMIC-fitted source and calibrate uncertainty; at least ten different eICU hospitals remain untouched for each evaluation split.",
        "",
        "No causal, treatment-effect, clinical, or automatic runtime-promotion claim is allowed.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    external = json.loads(args.external_report.read_text(encoding="utf-8"))
    original_by_cell = {row["cell"]: row for row in external["cells"]}
    contracts = [
        row for row in development["cells"]
        if not original_by_cell[row["cell"]]["full_external_pass"]
        and row["task_type"] != "binary_event_probability"
    ]
    if args.cells:
        selected = set(filter(None, map(str.strip, args.cells.split(","))))
        contracts = [row for row in contracts if row["cell"] in selected]
    mimic_cache: dict[int, pd.DataFrame] = {}
    cells = []
    for contract in contracts:
        cell = contract["cell"]
        target = contract["target"]
        horizon = int(contract["horizon_hours"])
        print(f"site-adaptive rescue {cell}", flush=True)
        if horizon not in mimic_cache:
            mimic_cache[horizon] = pd.read_parquet(mimic_path(args.project_root, horizon))
        mimic, mimic_support = normalize_cohort(
            mimic_cache[horizon],
            database="MIMIC-IV",
            target=target,
            horizon=horizon,
            max_rows=args.max_rows_per_database,
        )
        eicu, eicu_info = choose_eicu_cohort(
            args.eicu_root,
            target=target,
            horizon=horizon,
            max_rows=args.max_rows_per_database,
        )
        if mimic.empty or eicu.empty:
            cells.append({
                "cell": cell,
                "site_adaptive_validated": False,
                "reason": "insufficient_support",
            })
            continue
        result = audit_cell(
            contract,
            original_by_cell[cell],
            mimic,
            eicu,
        )
        result["mimic_support"] = mimic_support
        result["eicu_cohort"] = eicu_info
        cells.append(result)
        print(json.dumps({
            "cell": cell,
            "point": result["point_pass_splits"],
            "interval": result["interval_pass_splits"],
            "full": result["full_pass_splits"],
            "validated": result["site_adaptive_validated"],
        }), flush=True)
    original_full = int(external["full_external_pass_cells"])
    rescued = sum(bool(row.get("site_adaptive_validated")) for row in cells)
    output = {
        "schema": "mimic_to_eicu_site_adaptive_rescue.v1",
        "original_full_pass_cells": original_full,
        "attempted_cells": len(cells),
        "new_site_adaptive_pass_cells": rescued,
        "combined_usable_cells": original_full + rescued,
        "validated_rescue_cells": sorted(
            row["cell"] for row in cells if row.get("site_adaptive_validated")
        ),
        "cells": cells,
        "hospital_split_seeds": list(SPLIT_SEEDS),
        "treatment_schema_policy": "intersection of target-permitted pre-anchor hist fields with >=30 nonzero rows in both databases",
        "evaluation_labels_used_for_selection": False,
        "interval_calibration": "leave-one-adaptation-hospital-out adaptive patient-cluster conformal",
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "automatic_runtime_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown(output, args.findings)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-report",
        type=Path,
        default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"),
    )
    parser.add_argument(
        "--external-report",
        type=Path,
        default=Path("mimic_to_eicu_49_cell_external_audit_20260816.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_future_label_repaired"))
    parser.add_argument("--max-rows-per-database", type=int, default=15000)
    parser.add_argument("--cells", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({
        "attempted_cells": report["attempted_cells"],
        "new_site_adaptive_pass_cells": report["new_site_adaptive_pass_cells"],
        "combined_usable_cells": report["combined_usable_cells"],
        "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
