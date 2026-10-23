"""Frozen MIMIC recipes on an external ICU database, fixed vs value-conditional intervals.

Every catalog recipe is fit and calibrated on MIMIC-IV only and scored once on
an external database (eICU or NWICU).  Two intervals are scored side by side:

- ``fixed``: one patient-cluster conformal half-width (the frozen 2026-08-16
  external audit method);
- ``value_conditional`` (method v2, ``normalized_bin_scale``): residuals are
  divided by the median absolute residual of their current-value quintile bin
  (edges and scales from MIMIC calibration rows only), one patient-cluster
  conformal quantile is taken on the normalized residuals, and the interval
  half-width is that quantile times the bin scale.  Selected on MIMIC only in
  ``value_conditional_interval_method_dev_20261015.json``; it replaces the v1
  per-bin Mondrian width, which over-covered.

Gate axes depend on the external database (declared in the NWICU protocol):

- eICU: point = patient and hospital paired bootstraps; interval = patient- and
  hospital-equal coverage in [0.87, 0.93];
- NWICU (one hospital): point = patient paired bootstrap; interval = patient-
  and care-unit-equal coverage in [0.87, 0.93].

External labels are read only for scoring.  Factual research-only:
``causal_claim_allowed: false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from osler_jepa.recipe_availability import deployable_recipe

from mimic_to_eicu_49_cell_external_audit import (
    COVERAGE_BOUNDS,
    DATABASE_PREFIX,
    MIN_HOSPITALS,
    MIN_ROWS,
    MIN_SUBJECTS,
    choose_eicu_cohort,
    evaluate_event,
    fit_continuous_recipe,
    group_equal_coverage,
    normalize_cohort,
    paired_comparison,
    stable_seed,
)
from target_specific_domain_conformal_audit import _safe_float
from teacher_anchored_joint_jepa_audit import _patient_cluster_conformal_quantile


VALUE_BINS = 5
MIN_BIN_SUBJECTS = 30
MIN_GROUP_SUBJECTS = 20
CV_FOLDS = 5
SINGLE_SITE_AXES = {"point": ("patient",), "interval": ("subject_id", "careunit")}
GATE_AXES = {
    "eICU": {"point": ("patient", "hospital"), "interval": ("subject_id", "hospitalid")},
    "NWICU": SINGLE_SITE_AXES,
    "SICdb": SINGLE_SITE_AXES,
    "HiRID": SINGLE_SITE_AXES,
    "Zigong": SINGLE_SITE_AXES,
    "Emory": SINGLE_SITE_AXES,
}
EXTERNAL_PREFIX = {database: f"{prefix}:" for database, prefix in DATABASE_PREFIX.items() if database != "MIMIC-IV"}


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def value_bin_edges(calibration_values: np.ndarray, count: int = VALUE_BINS) -> np.ndarray:
    finite = calibration_values[np.isfinite(calibration_values)]
    return np.unique(np.quantile(finite, np.linspace(0, 1, count + 1)[1:-1]))


def normalized_bin_widths(
    frame: pd.DataFrame,
    calibration_rows: np.ndarray,
    calibration_residual: np.ndarray,
    calibration_bins: np.ndarray,
    query_bins: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Bin-scaled patient-cluster conformal widths; sparse bins use the overall scale."""

    subjects = frame.iloc[calibration_rows]["subject_id"].astype(str).to_numpy()
    overall_scale = max(float(np.median(calibration_residual)), 1e-9)
    scales = {}
    table = []
    for value in np.unique(np.concatenate([calibration_bins, query_bins])):
        mask = calibration_bins == value
        patients = int(len(np.unique(subjects[mask])))
        if patients >= MIN_BIN_SUBJECTS:
            scale, source = max(float(np.median(calibration_residual[mask])), 1e-9), "bin"
        else:
            scale, source = overall_scale, "overall_scale_fallback"
        scales[int(value)] = scale
        table.append({
            "bin": int(value),
            "calibration_rows": int(mask.sum()),
            "calibration_patients": patients,
            "scale": scale,
            "source": source,
        })
    normalized = calibration_residual / np.array([scales[int(value)] for value in calibration_bins])
    quantile, _, _ = _patient_cluster_conformal_quantile(
        frame, calibration_rows, normalized, coverage=0.90
    )
    for item in table:
        item["half_width"] = float(quantile * item["scale"])
    return np.array([quantile * scales[int(value)] for value in query_bins]), table


def mimic_cross_validated_coverage(
    frame: pd.DataFrame,
    calibration_rows: np.ndarray,
    residual: np.ndarray,
    values: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Patient-fold coverage inside MIMIC calibration for both interval types."""

    subjects = frame.iloc[calibration_rows]["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    np.random.default_rng(seed).shuffle(unique)
    fixed = np.zeros(len(calibration_rows), dtype=bool)
    conditional = np.zeros(len(calibration_rows), dtype=bool)
    for heldout in np.array_split(unique, CV_FOLDS):
        test = np.isin(subjects, heldout)
        train = ~test
        width, _, _ = _patient_cluster_conformal_quantile(
            frame, calibration_rows[train], residual[train], coverage=0.90
        )
        fixed[test] = residual[test] <= width
        edges = value_bin_edges(values[train])
        bin_widths, _ = normalized_bin_widths(
            frame,
            calibration_rows[train],
            residual[train],
            np.digitize(values[train], edges),
            np.digitize(values[test], edges),
        )
        conditional[test] = residual[test] <= bin_widths
    return {
        "fixed": group_equal_coverage(frame, calibration_rows, fixed, "subject_id"),
        "value_conditional": group_equal_coverage(frame, calibration_rows, conditional, "subject_id"),
    }


def interval_report(
    frame: pd.DataFrame,
    rows: np.ndarray,
    covered: np.ndarray,
    widths: np.ndarray,
    axes: tuple[str, ...],
) -> dict:
    report = {
        "patient_equal_coverage": group_equal_coverage(frame, rows, covered, "subject_id"),
        "hospital_equal_coverage": group_equal_coverage(frame, rows, covered, "hospitalid"),
        "careunit_equal_coverage": group_equal_coverage(frame, rows, covered, "careunit"),
        "median_half_width": float(np.median(widths)),
    }
    for column in ("hospitalid", "careunit"):
        groups = frame.iloc[rows][column].astype(str).to_numpy()
        subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
        values = [
            float(np.mean(covered[groups == group]))
            for group in np.unique(groups)
            if len(np.unique(subjects[groups == group])) >= MIN_GROUP_SUBJECTS
        ]
        report[f"{column}_coverage_range"] = [min(values), max(values)] if values else None
    names = {"subject_id": "patient_equal_coverage", "hospitalid": "hospital_equal_coverage", "careunit": "careunit_equal_coverage"}
    report["gate_axes"] = [names[axis] for axis in axes]
    report["pass"] = bool(all(
        COVERAGE_BOUNDS[0] <= report[names[axis]] <= COVERAGE_BOUNDS[1] for axis in axes
    ))
    return report


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def _point_pass(comparisons: dict, axes: tuple[str, ...]) -> bool:
    return bool(all(report[axis]["pass"] for report in comparisons.values() for axis in axes))


def evaluate_continuous_cell(
    mimic: pd.DataFrame,
    external: pd.DataFrame,
    *,
    cell: str,
    target: str,
    source: str,
    database: str,
) -> dict:
    axes = GATE_AXES[database]
    prefix = EXTERNAL_PREFIX[database]
    frame = pd.concat([mimic, external], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    train_rows = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("MIMIC:"))
    test_rows = np.flatnonzero(frame["subject_id"].astype(str).str.startswith(prefix))
    seed = stable_seed(cell, "mimic_to_eicu")
    fitted = fit_continuous_recipe(
        frame, train_rows, test_rows, target=target, source=source, seed=seed
    )
    future = _safe_float(frame, f"future_{target}")
    current = _safe_float(frame, f"{target}_t")
    rows = fitted["test_rows"]
    observed = future[rows]
    candidate_loss = np.abs(fitted["test_prediction"] - observed)
    comparisons = {
        "persistence": paired_comparison(
            frame, rows, candidate_loss, np.abs(current[rows] - observed), seed=seed
        ),
        fitted["comparator_name"]: paired_comparison(
            frame, rows, candidate_loss, np.abs(fitted["comparator"] - observed), seed=seed + 2
        ),
    }

    cal_rows = fitted["calibration_rows"]
    cal_residual = np.abs(fitted["calibration_prediction"] - future[cal_rows])
    fixed_width, _, _ = _patient_cluster_conformal_quantile(
        frame, cal_rows, cal_residual, coverage=0.90
    )
    fixed_covered = candidate_loss <= fixed_width
    edges = value_bin_edges(current[cal_rows])
    test_bins = np.digitize(current[rows], edges)
    widths, bins = normalized_bin_widths(
        frame, cal_rows, cal_residual, np.digitize(current[cal_rows], edges), test_bins
    )
    conditional_covered = candidate_loss <= widths
    for item in bins:
        mask = test_bins == item["bin"]
        item["external_rows"] = int(mask.sum())
        item["external_fixed_coverage"] = float(np.mean(fixed_covered[mask])) if mask.any() else None
        item["external_value_conditional_coverage"] = (
            float(np.mean(conditional_covered[mask])) if mask.any() else None
        )
    point = _point_pass(comparisons, axes["point"])
    fixed = interval_report(frame, rows, fixed_covered, np.full(len(rows), fixed_width), axes["interval"])
    conditional = interval_report(frame, rows, conditional_covered, widths, axes["interval"])
    return {
        "rows": int(len(rows)),
        "subjects": int(frame.iloc[rows]["subject_id"].nunique()),
        "hospitals": int(frame.iloc[rows]["hospitalid"].nunique()),
        "candidate_mae": float(candidate_loss.mean()),
        "persistence_mae": float(np.abs(current[rows] - observed).mean()),
        "comparisons": comparisons,
        "point_pass": point,
        "mimic_cross_validated_coverage": mimic_cross_validated_coverage(
            frame, cal_rows, cal_residual, current[cal_rows],
            seed=stable_seed(cell, "value_conditional_cv"),
        ),
        "bin_edges": [float(edge) for edge in edges],
        "fixed": fixed,
        "value_conditional": conditional,
        "bins": bins,
        "fixed_full_pass": bool(point and fixed["pass"]),
        "value_conditional_full_pass": bool(point and conditional["pass"]),
    }


def evaluate_event_cell(mimic: pd.DataFrame, external: pd.DataFrame, *, cell: str, database: str) -> dict:
    metrics = evaluate_event(
        mimic,
        external,
        seed=stable_seed(cell, "mimic_to_eicu"),
        external_prefix=EXTERNAL_PREFIX[database],
    )
    point = _point_pass(metrics["comparisons"], GATE_AXES[database]["point"])
    return {
        **metrics,
        "point_pass": point,
        "fixed_full_pass": point,
        "value_conditional_full_pass": point,
    }


def external_cohort(args: argparse.Namespace, target: str, horizon: int) -> tuple[pd.DataFrame, dict]:
    if args.database == "eICU":
        return choose_eicu_cohort(
            args.eicu_root, target=target, horizon=horizon, max_rows=args.max_rows_per_database
        )
    if not args.external_template:
        raise ValueError(f"--external-template is required for {args.database}")
    path = Path(args.external_template.format(horizon=horizon))
    if not path.exists():
        return pd.DataFrame(), {"path": str(path), "reason": "file_missing"}
    frame, support = normalize_cohort(
        pd.read_parquet(path), database=args.database, target=target, horizon=horizon,
        max_rows=args.max_rows_per_database,
    )
    return frame, {"path": str(path), "support": support}


def evaluability(database: str, external: pd.DataFrame) -> str | None:
    """Support checks only; targets a site does not record yield an empty cohort."""

    if external.empty:
        return "insufficient_external_support"
    if len(external) < MIN_ROWS or external["subject_id"].nunique() < MIN_SUBJECTS:
        return "insufficient_external_support"
    if database == "eICU" and external["hospitalid"].nunique() < MIN_HOSPITALS:
        return "insufficient_external_hospitals"
    return None


# ---------------------------------------------------------------------------
# Protocol enforcement and reporting
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol(protocol_path: Path) -> dict:
    """Refuse to run unless every frozen code, configuration and training-data file is unchanged."""

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = {**protocol["frozen_code_sha256"], **protocol.get("frozen_training_data_sha256", {})}
    mismatched = sorted(name for name, digest in frozen.items() if sha256(Path(name)) != digest)
    if mismatched:
        raise RuntimeError(f"files changed after protocol freeze: {mismatched}")
    return protocol


def write_markdown(report: dict, path: Path) -> None:
    database = report["external_database"]
    lines = [
        f"# MIMIC-IV to {database}: Fixed vs Value-Conditional Intervals",
        "",
        f"MIMIC data: `{report['mimic_template']}`. Gate axes: point {report['gate_axes']['point']}, "
        f"interval {report['gate_axes']['interval']}, coverage in {report['coverage_bounds']}.",
        "",
        f"- Catalog cells: `{report['catalog_cells']}`; evaluable: `{report['evaluable_cells']}`",
        f"- Point pass: `{report['point_pass_cells']}`",
        f"- Interval pass, fixed → value-conditional: `{report['fixed_interval_pass_cells']}` → "
        f"`{report['value_conditional_interval_pass_cells']}`",
        f"- Full pass, fixed → value-conditional: `{report['fixed_full_pass_cells']}` → "
        f"`{report['value_conditional_full_pass_cells']}`",
        "",
        "| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    group_key = "hospital_equal_coverage" if database == "eICU" else "careunit_equal_coverage"
    for row in report["cells"]:
        if not row["evaluable"]:
            lines.append(f"| {row['cell']} | {row['source_model']} | - | not evaluable: {row['reason']} | | | | |")
            continue
        metrics = row["metrics"]
        if "fixed" not in metrics:
            lines.append(
                f"| {row['cell']} | {row['source_model']} | {metrics['rows']} / {metrics['subjects']} | "
                f"{metrics['point_pass']} | event (Brier) | | | {metrics['fixed_full_pass']} |"
            )
            continue
        fixed, cond, cv = metrics["fixed"], metrics["value_conditional"], metrics["mimic_cross_validated_coverage"]
        lines.append(
            f"| {row['cell']} | {row['source_model']} | {metrics['rows']} / {metrics['subjects']} | "
            f"{metrics['point_pass']} | {fixed['patient_equal_coverage']:.3f} → {cond['patient_equal_coverage']:.3f} | "
            f"{fixed[group_key]:.3f} → {cond[group_key]:.3f} | {cv['fixed']:.3f} / {cv['value_conditional']:.3f} | "
            f"{metrics['fixed_full_pass']} → {metrics['value_conditional_full_pass']} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        f"All fitting, selection and calibration use MIMIC-IV only; {database} labels are read once for scoring.",
        "No result here changes the frozen runtime registry.",
        "`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    protocol = verify_protocol(args.protocol) if args.protocol else None
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = sorted(
        development["cells"], key=lambda row: (int(row["horizon_hours"]), str(row["target"]))
    )
    if args.cells:
        selected = set(filter(None, map(str.strip, args.cells.split(","))))
        contracts = [row for row in contracts if row["cell"] in selected]
    mimic_cache: dict[int, pd.DataFrame] = {}
    data_hashes: dict[str, str] = {}
    reports = []
    for contract in contracts:
        cell, target = str(contract["cell"]), str(contract["target"])
        horizon, source = int(contract["horizon_hours"]), str(contract["source_model"])
        print(f"{args.database}: {cell} ({source})", flush=True)
        base = {"cell": cell, "target": target, "horizon_hours": horizon, "source_model": source}
        if horizon not in mimic_cache:
            path = Path(args.mimic_template.format(horizon=horizon))
            mimic_cache[horizon] = pd.read_parquet(path)
            data_hashes[str(path)] = sha256(path)
        mimic, mimic_support = normalize_cohort(
            mimic_cache[horizon], database="MIMIC-IV", target=target, horizon=horizon,
            max_rows=args.max_rows_per_database,
        )
        external, info = external_cohort(args, target, horizon)
        source, recipe_reason = deployable_recipe(source, external, target)
        base["deployed_recipe"], base["recipe_reason"] = source, recipe_reason
        reason = evaluability(args.database, external)
        if reason is None and mimic.empty:
            reason = "insufficient_mimic_support"
        if reason:
            reports.append({**base, "evaluable": False, "reason": reason})
            continue
        if contract["task_type"] == "binary_event_probability":
            metrics = evaluate_event_cell(mimic, external, cell=cell, database=args.database)
        else:
            metrics = evaluate_continuous_cell(
                mimic, external, cell=cell, target=target, source=source, database=args.database
            )
        reports.append({
            **base,
            "evaluable": True,
            "mimic_support": mimic_support,
            "external_cohort": info,
            "metrics": metrics,
        })
        print(json.dumps({
            "cell": cell,
            "point": metrics["point_pass"],
            "full_fixed": metrics["fixed_full_pass"],
            "full_conditional": metrics["value_conditional_full_pass"],
        }), flush=True)
    evaluable = [row for row in reports if row["evaluable"]]
    continuous = [row for row in evaluable if "fixed" in row["metrics"]]
    output = {
        "schema": "external_value_conditional_interval_audit.v2",
        "value_conditional_method": "normalized_bin_scale.v2",
        "external_database": args.database,
        "protocol": str(args.protocol) if protocol else None,
        "mimic_template": args.mimic_template,
        "mimic_data_sha256": data_hashes,
        "gate_axes": {key: list(value) for key, value in GATE_AXES[args.database].items()},
        "coverage_bounds": list(COVERAGE_BOUNDS),
        "catalog_cells": len(reports),
        "evaluable_cells": len(evaluable),
        "not_evaluable_reasons": dict(Counter(row["reason"] for row in reports if not row["evaluable"])),
        "point_pass_cells": sum(row["metrics"]["point_pass"] for row in evaluable),
        "fixed_interval_pass_cells": sum(row["metrics"]["fixed"]["pass"] for row in continuous),
        "value_conditional_interval_pass_cells": sum(
            row["metrics"]["value_conditional"]["pass"] for row in continuous
        ),
        "fixed_full_pass_cells": sum(row["metrics"]["fixed_full_pass"] for row in evaluable),
        "value_conditional_full_pass_cells": sum(
            row["metrics"]["value_conditional_full_pass"] for row in evaluable
        ),
        "cells": reports,
        "external_labels_used_for_fitting_or_selection": False,
        "promotion_allowed": False,
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown(output, args.findings)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=sorted(GATE_AXES), required=True)
    parser.add_argument(
        "--development-report",
        type=Path,
        default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"),
    )
    parser.add_argument(
        "--mimic-template",
        default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet",
    )
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_future_label_repaired"))
    parser.add_argument("--external-template", default=None, help="parquet path template with {horizon} for single-site databases")
    parser.add_argument("--max-rows-per-database", type=int, default=15000)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--cells", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    args = parser.parse_args()
    if args.database == "NWICU" and args.protocol is None:
        parser.error("NWICU scoring requires --protocol so the frozen code is verified first")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
