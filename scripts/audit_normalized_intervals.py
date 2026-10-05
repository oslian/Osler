#!/usr/bin/env python3
"""Audit patient-conditional interval widths for selected release cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.calibrate_release_intervals import (
    ADAPTIVE_NOMINAL_LEVELS,
    ACCEPTED_COVERAGE,
    LEVEL,
    PATIENT_SPLIT_SEEDS,
    _group_equal_coverage,
    _in_range,
    _patient_cluster_radius,
)
from scripts.validate_subtype_external import _load_cell, stable_seed
from whole_body_jepa.runtime import load_release


def _fit_scale_model(
    features: np.ndarray, residual: np.ndarray, ridge: float = 10.0
) -> dict[str, np.ndarray | float]:
    matrix = np.asarray(features, dtype=np.float64)
    values = np.asarray(residual, dtype=np.float64)
    center = np.nanmedian(matrix, axis=0)
    center = np.nan_to_num(center, nan=0.0)
    matrix = np.where(np.isfinite(matrix), matrix, center)
    scale = np.nanstd(matrix, axis=0)
    scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
    standardized = np.clip((matrix - center) / scale, -8.0, 8.0)
    design = np.concatenate(
        [np.ones((len(standardized), 1)), standardized], axis=1
    )
    positive = values[np.isfinite(values) & (values > 0.0)]
    floor = max(float(np.quantile(positive, 0.02)), 1.0e-6)
    log_values = np.log(np.maximum(np.nan_to_num(values, nan=floor), floor))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ log_values)
    lower = max(float(np.quantile(np.maximum(values, floor), 0.05)), floor)
    upper = max(float(np.quantile(np.maximum(values, floor), 0.95)), lower)
    return {
        "center": center,
        "scale": scale,
        "beta": beta,
        "lower": lower,
        "upper": upper,
    }


def _predict_scale(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, model["center"])
    standardized = np.clip(
        (matrix - model["center"]) / model["scale"], -8.0, 8.0
    )
    design = np.concatenate(
        [np.ones((len(standardized), 1)), standardized], axis=1
    )
    prediction = np.exp(np.clip(design @ model["beta"], -20.0, 20.0))
    return np.clip(prediction, model["lower"], model["upper"])


def _serialize_scale_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "center": np.asarray(model["center"]).tolist(),
        "scale": np.asarray(model["scale"]).tolist(),
        "beta": np.asarray(model["beta"]).tolist(),
        "lower": float(model["lower"]),
        "upper": float(model["upper"]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_widths(
    calibration_residual: np.ndarray,
    calibration_features: np.ndarray,
    calibration_subjects: np.ndarray,
    test_features: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
    adaptive_nominal: bool = False,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    subjects = np.asarray(calibration_subjects).astype(str)
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    patient_folds = np.array_split(shuffled, folds)
    out_of_fold_scale = np.zeros(len(calibration_residual), dtype=np.float64)
    for fold_subjects in patient_folds:
        check = np.isin(subjects, fold_subjects)
        fit = ~check
        model = _fit_scale_model(
            calibration_features[fit], calibration_residual[fit]
        )
        out_of_fold_scale[check] = _predict_scale(
            model, calibration_features[check]
        )
    normalized = calibration_residual / np.maximum(out_of_fold_scale, 1.0e-8)
    selected_nominal = LEVEL
    selection = None
    if adaptive_nominal:
        diagnostics = {}
        for nominal in ADAPTIVE_NOMINAL_LEVELS:
            fold_hits = []
            fold_groups = []
            for heldout_subjects in patient_folds:
                check = np.isin(subjects, heldout_subjects)
                radius = _patient_cluster_radius(
                    normalized[~check], subjects[~check], float(nominal)
                )
                fold_hits.append(normalized[check] <= radius)
                fold_groups.append(subjects[check])
            hits = np.concatenate(fold_hits)
            groups = np.concatenate(fold_groups)
            row_coverage = float(np.mean(hits))
            patient_coverage = _group_equal_coverage(
                1.0 - hits.astype(np.float64), groups, 0.0
            )
            diagnostics[float(nominal)] = (row_coverage, patient_coverage)
        selected_nominal = min(
            diagnostics,
            key=lambda nominal: (
                max(
                    abs(diagnostics[nominal][0] - LEVEL),
                    abs(diagnostics[nominal][1] - LEVEL),
                ),
                abs(nominal - LEVEL),
            ),
        )
        selection = {
            "row_coverage": diagnostics[selected_nominal][0],
            "patient_equal_coverage": diagnostics[selected_nominal][1],
        }
    quantile = _patient_cluster_radius(
        normalized, subjects, selected_nominal
    )
    final_model = _fit_scale_model(calibration_features, calibration_residual)
    widths = quantile * _predict_scale(final_model, test_features)
    return widths, {
        "method": "patient_grouped_crossfit_normalized_conformal",
        "normalized_radius": float(quantile),
        "selected_nominal_coverage": float(selected_nominal),
        "adaptive_nominal_selection": selection,
        "calibration_patients": int(len(unique)),
        "selection_test_labels_used": False,
    }, final_model


def _coverage(residual: np.ndarray, widths: np.ndarray) -> float:
    return float(np.mean(np.asarray(residual) <= np.asarray(widths)))


def _patient_audit(
    payload: dict[str, Any],
    residual: np.ndarray,
    cell: str,
    *,
    adaptive_nominal: bool,
):
    output = []
    subjects = payload["subjects"]
    features = payload["features"]
    for seed in PATIENT_SPLIT_SEEDS:
        unique = np.unique(subjects)
        calibration_subjects = {
            subject
            for subject in unique
            if stable_seed(cell, "normalized_interval", seed, subject) % 2 == 0
        }
        calibration = np.isin(subjects, list(calibration_subjects))
        test = ~calibration
        widths, metadata, _ = _normalized_widths(
            residual[calibration],
            features[calibration],
            subjects[calibration],
            features[test],
            seed=stable_seed(cell, "normalized_fit", seed),
            adaptive_nominal=adaptive_nominal,
        )
        row_coverage = _coverage(residual[test], widths)
        patient_equal = _group_equal_coverage(
            residual[test] - widths, subjects[test], 0.0
        )
        output.append(
            {
                "seed": seed,
                "row_coverage": row_coverage,
                "patient_equal_coverage": patient_equal,
                "median_half_width": float(np.median(widths)),
                "pass": _in_range(row_coverage) and _in_range(patient_equal),
                "calibration": metadata,
            }
        )
    return output


def _careunit_audit(
    payload: dict[str, Any],
    residual: np.ndarray,
    cell: str,
    *,
    adaptive_nominal: bool,
):
    widths_by_row = np.full(len(residual), np.nan, dtype=np.float64)
    folds = []
    for careunit in np.unique(payload["careunits"]):
        test = payload["careunits"] == careunit
        if int(test.sum()) < 100 or int((~test).sum()) < 100:
            continue
        widths, _, _ = _normalized_widths(
            residual[~test],
            payload["features"][~test],
            payload["subjects"][~test],
            payload["features"][test],
            seed=stable_seed(cell, "normalized_careunit", careunit),
            adaptive_nominal=adaptive_nominal,
        )
        widths_by_row[test] = widths
        folds.append(
            {
                "careunit": str(careunit),
                "coverage": _coverage(residual[test], widths),
                "median_half_width": float(np.median(widths)),
            }
        )
    available = np.isfinite(widths_by_row)
    pooled = _coverage(residual[available], widths_by_row[available])
    equal = float(np.mean([fold["coverage"] for fold in folds]))
    return {
        "pooled_coverage": pooled,
        "careunit_equal_coverage": equal,
        "pass": len(folds) >= 2 and _in_range(pooled) and _in_range(equal),
        "folds": folds,
    }


def _time_audit(
    payload: dict[str, Any],
    residual: np.ndarray,
    cell: str,
    *,
    adaptive_nominal: bool,
):
    timestamps = np.asarray(payload["time"]).astype("datetime64[ns]")
    order = np.argsort(timestamps, kind="stable")
    boundary = max(100, int(round(0.70 * len(order))))
    calibration = order[:boundary]
    test = order[boundary:]
    widths, metadata, _ = _normalized_widths(
        residual[calibration],
        payload["features"][calibration],
        payload["subjects"][calibration],
        payload["features"][test],
        seed=stable_seed(cell, "normalized_time"),
        adaptive_nominal=adaptive_nominal,
    )
    coverage = _coverage(residual[test], widths)
    return {
        "coverage": coverage,
        "median_half_width": float(np.median(widths)),
        "pass": _in_range(coverage),
        "calibration": metadata,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_release(args.model_dir)
    by_cell = {str(item["cell"]): item for item in assets.registry["cells"]}
    results = {}
    for cell in args.cells:
        print(f"auditing {cell}", flush=True)
        contract = by_cell[cell]
        payload = _load_cell(
            database="MIMIC",
            project_root=args.project_root,
            eicu_root=Path("."),
            target=str(contract["target"]),
            horizon=int(contract["horizon_hours"]),
            cell_contract=contract,
            assets=assets,
            max_rows=args.max_rows,
            max_rows_per_subject=args.max_rows_per_subject,
            batch_size=args.batch_size,
        )
        if payload is None:
            results[cell] = {"pass": False, "reason": "insufficient_support"}
            continue
        residual = np.abs(payload["truth"] - payload["base"])
        patient = _patient_audit(
            payload, residual, cell, adaptive_nominal=args.adaptive_nominal
        )
        careunit = _careunit_audit(
            payload, residual, cell, adaptive_nominal=args.adaptive_nominal
        )
        time = _time_audit(
            payload, residual, cell, adaptive_nominal=args.adaptive_nominal
        )
        final_widths, final_metadata, final_model = _normalized_widths(
            residual,
            payload["features"],
            payload["subjects"],
            payload["features"],
            seed=stable_seed(cell, "normalized_final"),
            adaptive_nominal=args.adaptive_nominal,
        )
        results[cell] = {
            "patient_splits": patient,
            "careunit_holdout": careunit,
            "forward_time": time,
            "final_calibration": {
                **final_metadata,
                "mean_half_width": float(np.mean(final_widths)),
                "median_half_width": float(np.median(final_widths)),
                "residual_scale_model": _serialize_scale_model(final_model),
            },
            "pass": all(item["pass"] for item in patient)
            and careunit["pass"]
            and time["pass"],
        }
    report = {
        "schema": "whole_body_jepa.normalized_interval_audit.v1",
        "level": LEVEL,
        "accepted_coverage_range": list(ACCEPTED_COVERAGE),
        "cells": results,
        "passed": sorted(cell for cell, result in results.items() if result["pass"]),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.update_calibrator:
        calibrator = json.loads(
            args.update_calibrator.read_text(encoding="utf-8")
        )
        for cell in report["passed"]:
            audit = results[cell]
            final = audit["final_calibration"]
            calibrator["cells"][cell].update(
                {
                    "runtime_eligible": True,
                    "failure_reasons": [],
                    "runtime_interval_method": "normalized_conformal",
                    "normalized_radius": final["normalized_radius"],
                    "residual_scale_model": final["residual_scale_model"],
                    "radius": final["median_half_width"],
                    "mean_width": 2.0 * final["mean_half_width"],
                    "patient_splits": audit["patient_splits"],
                    "careunit_holdout": audit["careunit_holdout"],
                    "forward_time": audit["forward_time"],
                    "calibration": {
                        key: value
                        for key, value in final.items()
                        if key != "residual_scale_model"
                    },
                }
            )
        calibrator["runtime_eligible_cells"] = sorted(
            cell
            for cell, entry in calibrator["cells"].items()
            if entry.get("runtime_eligible")
        )
        args.update_calibrator.write_text(
            json.dumps(calibrator, indent=2), encoding="utf-8"
        )
        manifest_path = args.model_dir / "MODEL_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["externally_validated_interval_cells"] = len(
            calibrator["runtime_eligible_cells"]
        )
        manifest["files"]["interval_calibrator.json"] = {
            "bytes": args.update_calibrator.stat().st_size,
            "sha256": _sha256(args.update_calibrator),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(json.dumps({"passed": report["passed"]}, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--max-rows-per-subject", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--adaptive-nominal", action="store_true")
    parser.add_argument(
        "--update-calibrator",
        type=Path,
        help="Promote only fully passing cells into this release calibrator.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
