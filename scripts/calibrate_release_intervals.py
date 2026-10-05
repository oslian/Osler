#!/usr/bin/env python3
"""Build release-bound 90% split-conformal intervals on MIMIC-IV.

Point models are frozen.  MIMIC patients are split before residual calibration,
and a cell is runtime eligible only when patient, care-unit, and forward-time
coverage all remain inside the declared acceptance range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.validate_subtype_external import _load_cell, stable_seed
from whole_body_jepa.runtime import load_release


LEVEL = 0.90
ACCEPTED_COVERAGE = (0.87, 0.93)
PATIENT_SPLIT_SEEDS = (7, 19, 31, 43, 59, 71, 83)
ADAPTIVE_NOMINAL_LEVELS = tuple(np.round(np.arange(0.82, 0.951, 0.01), 2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _conformal_radius(residual: np.ndarray, level: float = LEVEL) -> float:
    finite = np.asarray(residual, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("conformal calibration requires finite residuals")
    quantile = min(1.0, np.ceil((len(finite) + 1) * level) / len(finite))
    return float(np.quantile(finite, quantile, method="higher"))


def _patient_cluster_radius(
    residual: np.ndarray,
    subjects: np.ndarray,
    level: float,
) -> float:
    """Return a finite-sample radius with equal total weight per patient."""

    scores = np.asarray(residual, dtype=np.float64)
    groups = np.asarray(subjects).astype(str)
    finite = np.isfinite(scores)
    scores = scores[finite]
    groups = groups[finite]
    if not len(scores):
        raise ValueError("patient-cluster conformal requires finite residuals")
    unique, inverse, counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse].astype(np.float64)
    order = np.argsort(scores, kind="stable")
    corrected_level = min(
        1.0, float(level) * (len(unique) + 1.0) / max(len(unique), 1)
    )
    threshold = corrected_level * float(weights.sum())
    index = int(
        np.searchsorted(np.cumsum(weights[order]), threshold, side="left")
    )
    return float(scores[order[min(index, len(order) - 1)]])


def _adaptive_patient_cluster_radius(
    residual: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
) -> tuple[float, dict[str, Any]]:
    """Choose nominal coverage by patient CV without using audit labels."""

    scores = np.asarray(residual, dtype=np.float64)
    groups = np.asarray(subjects).astype(str)
    finite = np.isfinite(scores)
    scores = scores[finite]
    groups = groups[finite]
    unique = np.unique(groups)
    if len(scores) < 120 or len(unique) < folds * 5:
        return _patient_cluster_radius(scores, groups, LEVEL), {
            "method": "patient_cluster_conformal_fallback",
            "selected_nominal_coverage": LEVEL,
            "selection_test_labels_used": False,
        }

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    patient_folds = np.array_split(shuffled, folds)
    diagnostics: dict[float, dict[str, float]] = {}
    for nominal in ADAPTIVE_NOMINAL_LEVELS:
        heldout_hits: list[np.ndarray] = []
        heldout_groups: list[np.ndarray] = []
        for fold_groups in patient_folds:
            test = np.isin(groups, fold_groups)
            calibration = ~test
            radius = _patient_cluster_radius(
                scores[calibration], groups[calibration], float(nominal)
            )
            heldout_hits.append(scores[test] <= radius)
            heldout_groups.append(groups[test])
        hits = np.concatenate(heldout_hits)
        check_groups = np.concatenate(heldout_groups)
        row_coverage = float(np.mean(hits))
        patient_coverage = _group_equal_coverage(
            1.0 - hits.astype(np.float64), check_groups, 0.0
        )
        diagnostics[float(nominal)] = {
            "row_coverage": row_coverage,
            "patient_equal_coverage": patient_coverage,
        }

    selected = min(
        diagnostics,
        key=lambda nominal: (
            max(
                abs(diagnostics[nominal]["row_coverage"] - LEVEL),
                abs(
                    diagnostics[nominal]["patient_equal_coverage"] - LEVEL
                ),
            ),
            abs(nominal - LEVEL),
        ),
    )
    radius = _patient_cluster_radius(scores, groups, selected)
    return radius, {
        "method": "patient_grouped_crossfit_adaptive_conformal",
        "selected_nominal_coverage": float(selected),
        "selection_row_coverage": diagnostics[selected]["row_coverage"],
        "selection_patient_equal_coverage": diagnostics[selected][
            "patient_equal_coverage"
        ],
        "selection_folds": folds,
        "selection_patients": int(len(unique)),
        "selection_test_labels_used": False,
    }


def _coverage(residual: np.ndarray, radius: float) -> float:
    return float(np.mean(np.asarray(residual) <= radius))


def _group_equal_coverage(
    residual: np.ndarray, groups: np.ndarray, radius: float
) -> float:
    values = []
    for group in np.unique(groups):
        mask = groups == group
        if np.any(mask):
            values.append(_coverage(residual[mask], radius))
    return float(np.mean(values))


def _in_range(value: float) -> bool:
    return ACCEPTED_COVERAGE[0] <= value <= ACCEPTED_COVERAGE[1]


def _patient_audit(
    residual: np.ndarray,
    subjects: np.ndarray,
    cell: str,
    *,
    adaptive: bool,
) -> list[dict[str, Any]]:
    output = []
    for seed in PATIENT_SPLIT_SEEDS:
        unique = np.unique(subjects)
        calibration_subjects = {
            subject
            for subject in unique
            if stable_seed(cell, "interval", seed, subject) % 2 == 0
        }
        calibration = np.isin(subjects, list(calibration_subjects))
        test = ~calibration
        if adaptive:
            radius, calibration_metadata = _adaptive_patient_cluster_radius(
                residual[calibration],
                subjects[calibration],
                seed=stable_seed(cell, "adaptive_interval", seed),
            )
        else:
            radius = _conformal_radius(residual[calibration])
            calibration_metadata = {
                "method": "absolute_residual_conformal",
                "selected_nominal_coverage": LEVEL,
                "selection_test_labels_used": False,
            }
        row_coverage = _coverage(residual[test], radius)
        patient_equal = _group_equal_coverage(
            residual[test], subjects[test], radius
        )
        output.append(
            {
                "seed": seed,
                "calibration_patients": int(len(np.unique(subjects[calibration]))),
                "test_patients": int(len(np.unique(subjects[test]))),
                "radius": radius,
                "row_coverage": row_coverage,
                "patient_equal_coverage": patient_equal,
                "pass": _in_range(row_coverage) and _in_range(patient_equal),
                "calibration": calibration_metadata,
            }
        )
    return output


def _careunit_audit(
    residual: np.ndarray,
    careunits: np.ndarray,
    subjects: np.ndarray,
    cell: str,
    *,
    adaptive: bool,
) -> dict[str, Any]:
    folds = []
    hit_parts = []
    for careunit in np.unique(careunits):
        test = careunits == careunit
        if int(test.sum()) < 100 or int((~test).sum()) < 100:
            continue
        if adaptive:
            radius, calibration_metadata = _adaptive_patient_cluster_radius(
                residual[~test],
                subjects[~test],
                seed=stable_seed(cell, "adaptive_careunit", careunit),
            )
        else:
            radius = _conformal_radius(residual[~test])
            calibration_metadata = {
                "method": "absolute_residual_conformal",
                "selected_nominal_coverage": LEVEL,
                "selection_test_labels_used": False,
            }
        hits = residual[test] <= radius
        hit_parts.append(hits)
        folds.append(
            {
                "careunit": str(careunit),
                "rows": int(test.sum()),
                "coverage": float(np.mean(hits)),
                "calibration": calibration_metadata,
            }
        )
    if len(folds) < 2:
        return {"available": False, "pass": False, "folds": folds}
    pooled = float(np.mean(np.concatenate(hit_parts)))
    equal = float(np.mean([fold["coverage"] for fold in folds]))
    return {
        "available": True,
        "pooled_coverage": pooled,
        "careunit_equal_coverage": equal,
        "pass": _in_range(pooled) and _in_range(equal),
        "folds": folds,
    }


def _time_audit(
    residual: np.ndarray,
    time: np.ndarray,
    subjects: np.ndarray,
    cell: str,
    *,
    adaptive: bool,
) -> dict[str, Any]:
    timestamps = np.asarray(time).astype("datetime64[ns]")
    order = np.argsort(timestamps, kind="stable")
    boundary = max(100, int(round(0.70 * len(order))))
    if len(order) - boundary < 100:
        return {"available": False, "pass": False}
    calibration = order[:boundary]
    test = order[boundary:]
    if adaptive:
        radius, calibration_metadata = _adaptive_patient_cluster_radius(
            residual[calibration],
            subjects[calibration],
            seed=stable_seed(cell, "adaptive_forward_time"),
        )
    else:
        radius = _conformal_radius(residual[calibration])
        calibration_metadata = {
            "method": "absolute_residual_conformal",
            "selected_nominal_coverage": LEVEL,
            "selection_test_labels_used": False,
        }
    coverage = _coverage(residual[test], radius)
    return {
        "available": True,
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "radius": radius,
        "coverage": coverage,
        "pass": _in_range(coverage),
        "calibration": calibration_metadata,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_release(args.model_dir)
    selected_cells = set(args.cells or ())
    if selected_cells and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        cells: dict[str, Any] = dict(previous.get("cells", {}))
    else:
        cells = {}
    for index, contract in enumerate(assets.registry["cells"], start=1):
        cell = str(contract["cell"])
        target = str(contract["target"])
        horizon = int(contract["horizon_hours"])
        if not bool(contract.get("can_move")) or target == "hypoxemia_event":
            continue
        if selected_cells and cell not in selected_cells:
            continue
        print(f"[{index}/41] calibrating {cell}", flush=True)
        payload = _load_cell(
            database="MIMIC",
            project_root=args.project_root,
            eicu_root=Path("."),
            target=target,
            horizon=horizon,
            cell_contract=contract,
            assets=assets,
            max_rows=args.max_rows,
            max_rows_per_subject=args.max_rows_per_subject,
            batch_size=args.batch_size,
        )
        if payload is None:
            cells[cell] = {
                "runtime_eligible": False,
                "failure_reasons": ["insufficient_mimic_support"],
            }
            continue
        residual = np.abs(payload["truth"] - payload["base"])
        adaptive = cell in selected_cells
        patient = _patient_audit(
            residual, payload["subjects"], cell, adaptive=adaptive
        )
        careunit = _careunit_audit(
            residual,
            payload["careunits"],
            payload["subjects"],
            cell,
            adaptive=adaptive,
        )
        forward_time = _time_audit(
            residual,
            payload["time"],
            payload["subjects"],
            cell,
            adaptive=adaptive,
        )
        reasons = []
        if not all(item["pass"] for item in patient):
            reasons.append("patient_split_coverage")
        if not careunit["pass"]:
            reasons.append("careunit_coverage")
        if not forward_time["pass"]:
            reasons.append("forward_time_coverage")
        if adaptive:
            final_radius, final_calibration = _adaptive_patient_cluster_radius(
                residual,
                payload["subjects"],
                seed=stable_seed(cell, "adaptive_final"),
            )
        else:
            final_radius = _conformal_radius(residual)
            final_calibration = {
                "method": "absolute_residual_conformal",
                "selected_nominal_coverage": LEVEL,
                "selection_test_labels_used": False,
            }
        cells[cell] = {
            "target": target,
            "horizon_hours": horizon,
            "source_model": contract["source_model"],
            "rows": int(len(residual)),
            "patients": int(len(np.unique(payload["subjects"]))),
            "radius": final_radius,
            "mean_width": 2.0 * final_radius,
            "calibration": final_calibration,
            "patient_splits": patient,
            "careunit_holdout": careunit,
            "forward_time": forward_time,
            "runtime_eligible": not reasons,
            "failure_reasons": reasons,
        }

    model_dir = Path(args.model_dir)
    binding_files = (
        "whole_body_joint_jepa.pt",
        "unified_shared_jepa_head_bundle.joblib",
        "runtime_registry.json",
    )
    result = {
        "schema": "whole_body_jepa.interval_calibrator.v1",
        "method": "patient_split_conformal_with_patient_cluster_rescue",
        "level": LEVEL,
        "accepted_coverage_range": list(ACCEPTED_COVERAGE),
        "calibration_database": "MIMIC-IV",
        "external_test_reused_for_point_selection": True,
        "clinical_promotion_allowed": False,
        "causal_claim_allowed": False,
        "release_binding": {
            name: _sha256(model_dir / name) for name in binding_files
        },
        "cells": cells,
        "runtime_eligible_cells": sorted(
            cell for cell, payload in cells.items() if payload["runtime_eligible"]
        ),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "attempted": len(cells),
                "runtime_eligible": len(result["runtime_eligible_cells"]),
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("models/interval_calibrator.json"))
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--max-rows-per-subject", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--cells",
        nargs="*",
        help="Recalibrate only these cells and merge them into an existing artifact.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
