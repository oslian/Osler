"""MIMIC-only development of a well-calibrated value-conditional interval.

The v1 Mondrian interval (patient weights computed within each value bin)
over-covers: MIMIC patient-fold coverage 0.902-0.944 against a 0.90 target.
Suspected cause: a patient with few rows in a bin gets full weight there, while
the patient-equal evaluation counts each row as 1 / (patient's total rows).

Candidates, all calibrated on MIMIC calibration rows only:

- ``fixed``: one patient-cluster conformal width (reference);
- ``mondrian_v1``: the frozen v1 method;
- ``mondrian_global_weight``: per value bin, a weighted conformal quantile whose
  row weight is 1 / (patient's total calibration rows), matching the
  patient-equal evaluation;
- ``normalized_bin_scale``: residuals divided by the bin's median absolute
  residual, one global patient-cluster quantile, width = quantile * bin scale.

Predeclared selection rule (written before any candidate was scored):

1. most cells with MIMIC patient-fold patient-equal coverage in [0.89, 0.91];
2. tie-break: smaller median over cells of the worst per-bin coverage error;
3. the winner must also have a smaller median worst per-bin error than
   ``fixed``, or no value-conditional method is adopted.

No eICU or NWICU data is read.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import fit_continuous_recipe, normalize_cohort, stable_seed
from target_specific_domain_conformal_audit import _safe_float
from teacher_anchored_joint_jepa_audit import _patient_cluster_conformal_quantile


VALUE_BINS = 5
MIN_BIN_SUBJECTS = 30
CV_FOLDS = 5
NOMINAL = 0.90
CANDIDATES = ("fixed", "mondrian_v1", "mondrian_global_weight", "normalized_bin_scale")


def value_bin_edges(values: np.ndarray, count: int = VALUE_BINS) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return np.unique(np.quantile(finite, np.linspace(0, 1, count + 1)[1:-1]))


def patient_row_weights(subjects: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(np.float64)


def weighted_conformal_quantile(
    scores: np.ndarray, weights: np.ndarray, patients: int, coverage: float = NOMINAL
) -> float:
    """Weighted split-conformal quantile with a patient-count finite-sample correction."""

    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    level = min(1.0, coverage * (patients + 1.0) / max(patients, 1))
    index = min(int(np.searchsorted(cumulative, level * cumulative[-1], side="left")), len(scores) - 1)
    return float(scores[order][index])


def candidate_widths(
    method: str,
    frame: pd.DataFrame,
    rows: np.ndarray,
    residual: np.ndarray,
    bins: np.ndarray,
    query_bins: np.ndarray,
) -> np.ndarray:
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    pooled, _, _ = _patient_cluster_conformal_quantile(frame, rows, residual, coverage=NOMINAL)
    if method == "fixed":
        return np.full(len(query_bins), pooled)
    widths: dict[int, float] = {}
    if method == "normalized_bin_scale":
        overall_scale = max(float(np.median(residual)), 1e-9)
        scale = {}
        for value in np.unique(np.concatenate([bins, query_bins])):
            mask = bins == value
            enough = len(np.unique(subjects[mask])) >= MIN_BIN_SUBJECTS
            scale[int(value)] = max(float(np.median(residual[mask])), 1e-9) if enough else overall_scale
        normalized = residual / np.array([scale[int(value)] for value in bins])
        quantile, _, _ = _patient_cluster_conformal_quantile(frame, rows, normalized, coverage=NOMINAL)
        return np.array([quantile * scale[int(value)] for value in query_bins])
    global_weights = patient_row_weights(subjects)
    for value in np.unique(np.concatenate([bins, query_bins])):
        mask = bins == value
        patients = len(np.unique(subjects[mask]))
        if patients < MIN_BIN_SUBJECTS:
            widths[int(value)] = pooled
        elif method == "mondrian_v1":
            widths[int(value)], _, _ = _patient_cluster_conformal_quantile(
                frame, rows[mask], residual[mask], coverage=NOMINAL
            )
        elif method == "mondrian_global_weight":
            widths[int(value)] = weighted_conformal_quantile(residual[mask], global_weights[mask], patients)
        else:
            raise ValueError(method)
    return np.array([widths[int(value)] for value in query_bins])


def cross_validated(frame: pd.DataFrame, rows: np.ndarray, residual: np.ndarray, values: np.ndarray, seed: int) -> dict:
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    np.random.default_rng(seed).shuffle(unique)
    covered = {method: np.zeros(len(rows), dtype=bool) for method in CANDIDATES}
    width = {method: np.zeros(len(rows)) for method in CANDIDATES}
    eval_bins = np.zeros(len(rows), dtype=int)
    for heldout in np.array_split(unique, CV_FOLDS):
        test = np.isin(subjects, heldout)
        train = ~test
        edges = value_bin_edges(values[train])
        train_bins, test_bins = np.digitize(values[train], edges), np.digitize(values[test], edges)
        eval_bins[test] = test_bins
        for method in CANDIDATES:
            widths = candidate_widths(method, frame, rows[train], residual[train], train_bins, test_bins)
            covered[method][test] = residual[test] <= widths
            width[method][test] = widths
    weights = patient_row_weights(subjects)
    patients = len(unique)
    report = {}
    for method in CANDIDATES:
        bin_errors = []
        for value in np.unique(eval_bins):
            mask = eval_bins == value
            bin_coverage = float(np.sum(weights[mask] * covered[method][mask]) / np.sum(weights[mask]))
            bin_errors.append(abs(bin_coverage - NOMINAL))
        report[method] = {
            "patient_equal_coverage": float(np.sum(weights * covered[method]) / patients),
            "worst_bin_coverage_error": float(max(bin_errors)),
            "median_width": float(np.median(width[method])),
        }
    return report


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = [row for row in development["cells"] if row["task_type"] != "binary_event_probability"]
    by_horizon: dict[int, list[dict]] = {}
    for row in contracts:
        by_horizon.setdefault(int(row["horizon_hours"]), []).append(row)
    cells = []
    for horizon in sorted(by_horizon):
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        for contract in sorted(by_horizon[horizon], key=lambda row: row["target"]):
            cell, target, source = contract["cell"], contract["target"], contract["source_model"]
            print(f"developing interval on {cell} ({source})", flush=True)
            mimic, _ = normalize_cohort(
                raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=args.max_rows
            )
            frame = mimic.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
            train = np.arange(len(frame))
            fitted = fit_continuous_recipe(
                frame, train, train[:1], target=target, source=source,
                seed=stable_seed(cell, "mimic_to_eicu"),
            )
            rows = fitted["calibration_rows"]
            residual = np.abs(fitted["calibration_prediction"] - _safe_float(frame, f"future_{target}")[rows])
            values = _safe_float(frame, f"{target}_t")[rows]
            cells.append({
                "cell": cell,
                "source_model": source,
                "calibration_rows": int(len(rows)),
                "calibration_patients": int(frame.iloc[rows]["subject_id"].nunique()),
                "methods": cross_validated(frame, rows, residual, values, stable_seed(cell, "interval_dev_cv")),
            })
        del raw
    summary = {}
    for method in CANDIDATES:
        coverage = np.array([cell["methods"][method]["patient_equal_coverage"] for cell in cells])
        worst = np.array([cell["methods"][method]["worst_bin_coverage_error"] for cell in cells])
        relative = np.array([
            cell["methods"][method]["median_width"] / cell["methods"]["fixed"]["median_width"] for cell in cells
        ])
        summary[method] = {
            "cells_in_0.89_0.91": int(np.sum((coverage >= 0.89) & (coverage <= 0.91))),
            "median_coverage": float(np.median(coverage)),
            "coverage_range": [float(coverage.min()), float(coverage.max())],
            "median_worst_bin_error": float(np.median(worst)),
            "median_relative_width_vs_fixed": float(np.median(relative)),
        }
    ranked = sorted(
        (method for method in CANDIDATES if method != "fixed"),
        key=lambda method: (-summary[method]["cells_in_0.89_0.91"], summary[method]["median_worst_bin_error"]),
    )
    winner = ranked[0]
    adopted = summary[winner]["median_worst_bin_error"] < summary["fixed"]["median_worst_bin_error"]
    output = {
        "schema": "value_conditional_interval_method_dev.v1",
        "database": "MIMIC-IV only",
        "mimic_template": args.mimic_template,
        "selection_rule": "most cells in [0.89, 0.91]; tie-break median worst per-bin error; must beat fixed on worst per-bin error",
        "summary": summary,
        "ranking": ranked,
        "selected": winner if adopted else None,
        "cells": cells,
        "external_data_read": False,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-report",
        type=Path,
        default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"),
    )
    parser.add_argument(
        "--mimic-template",
        default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet",
    )
    parser.add_argument("--max-rows", type=int, default=15000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"summary": result["summary"], "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
