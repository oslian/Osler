"""Decompose the eICU coverage gap left after value-conditional intervals (v2).

With v2 intervals (calibrated on MIMIC only) eICU coverage stays below 0.90 at
the same current value.  Hypotheses tested per continuous catalog cell:

- H1 label timing: eICU keeps "near" labels up to 3.6h from anchor + horizon,
  MIMIC only within 2h.  Compare eICU coverage on exact vs all labels.
- H2 missing inputs: eICU module files carry 13-27 state variables and 10-40
  treatment columns versus 81 and 279 in MIMIC; absent inputs are imputed with
  MIMIC medians.  MIMIC-only test: fit on 70% of MIMIC patients, score the other
  30% with and without the eICU-absent inputs masked, same v2 interval.
- H3 target volatility: at the same current-value bin, compare the persistence
  error |future - current| in eICU and MIMIC.
- H4 definitions: compare current-value distributions per target.
- H5 in-window treatment: compare eICU coverage on rows with and without any
  treatment recorded inside the forecast window (``act_*``), and the share of
  such rows in eICU and MIMIC.  Coverage failures live in the residual tail,
  so H3 is also measured at the 90th percentile.

eICU labels are used for analysis only; nothing is fit, selected or promoted.
Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_value_conditional_interval_audit import normalized_bin_widths, value_bin_edges
from mimic_to_eicu_49_cell_external_audit import (
    choose_eicu_cohort,
    fit_continuous_recipe,
    group_equal_coverage,
    normalize_cohort,
    stable_seed,
)
from target_specific_domain_conformal_audit import _safe_float


AVAILABLE_SHARE = 0.01


def input_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns
        if (column.endswith("_t") and not column.startswith(("future_", "_source_")) and column != "t")
        or column.endswith("_age_hr")
        or column.startswith(("_observed_", "hist_"))
    ]


def absent_inputs(mimic: pd.DataFrame, eicu: pd.DataFrame, target: str) -> list[str]:
    absent = []
    for column in input_columns(mimic):
        if column in (f"{target}_t", f"{target}_age_hr"):
            continue
        share = (
            float(pd.to_numeric(eicu[column], errors="coerce").notna().mean())
            if column in eicu else 0.0
        )
        if share < AVAILABLE_SHARE:
            absent.append(column)
    return absent


def v2_widths(frame: pd.DataFrame, fitted: dict, target: str, query_rows: np.ndarray) -> np.ndarray:
    future = _safe_float(frame, f"future_{target}")
    current = _safe_float(frame, f"{target}_t")
    cal = fitted["calibration_rows"]
    residual = np.abs(fitted["calibration_prediction"] - future[cal])
    edges = value_bin_edges(current[cal])
    widths, _ = normalized_bin_widths(
        frame, cal, residual, np.digitize(current[cal], edges), np.digitize(current[query_rows], edges)
    )
    return widths


def patient_coverage(frame: pd.DataFrame, rows: np.ndarray, covered: np.ndarray) -> float | None:
    return group_equal_coverage(frame, rows, covered, "subject_id") if len(rows) else None


def eicu_side(mimic: pd.DataFrame, eicu: pd.DataFrame, *, cell: str, target: str, source: str) -> dict:
    frame = pd.concat([mimic, eicu], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    train = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("MIMIC:"))
    test = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("EICU:"))
    fitted = fit_continuous_recipe(
        frame, train, test, target=target, source=source, seed=stable_seed(cell, "mimic_to_eicu")
    )
    rows = fitted["test_rows"]
    future = _safe_float(frame, f"future_{target}")
    current = _safe_float(frame, f"{target}_t")
    residual = np.abs(fitted["test_prediction"] - future[rows])
    covered = residual <= v2_widths(frame, fitted, target, rows)
    quality = frame.iloc[rows].get(
        f"future_label_quality_{target}", pd.Series("exact", index=frame.index[rows])
    ).astype(str).str.lower().to_numpy()
    exact = quality == "exact"

    cal = fitted["calibration_rows"]
    cal_persistence = np.abs(future[cal] - current[cal])
    cal_residual = np.abs(fitted["calibration_prediction"] - future[cal])
    edges = value_bin_edges(current[cal])
    cal_bins, test_bins = np.digitize(current[cal], edges), np.digitize(current[rows], edges)
    persistence = np.abs(future[rows] - current[rows])
    ratios = {"p50_persistence": [], "p50_residual": [], "p90_persistence": [], "p90_residual": []}
    for value in np.unique(cal_bins):
        a, b = cal_bins == value, (test_bins == value) & exact
        if a.sum() >= 30 and b.sum() >= 30:
            for q in (50, 90):
                ratios[f"p{q}_persistence"].append(
                    np.percentile(persistence[b], q) / max(np.percentile(cal_persistence[a], q), 1e-9)
                )
                ratios[f"p{q}_residual"].append(
                    np.percentile(residual[b], q) / max(np.percentile(cal_residual[a], q), 1e-9)
                )
    act_columns = [
        column for column in frame.columns
        if column.startswith("act_") and not column.endswith(("_evidence_count", "_dose_observed"))
    ]
    shared_act = [
        column for column in act_columns
        if frame.iloc[rows][column].notna().any() and frame.iloc[cal][column].notna().any()
    ]

    def any_act(index: np.ndarray) -> np.ndarray:
        if not shared_act:
            return np.zeros(len(index), dtype=bool)
        values = frame.iloc[index][shared_act].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return (values.to_numpy() > 0).any(axis=1)

    treated_test, treated_cal = any_act(rows), any_act(cal)
    mimic_values = current[cal]
    return {
        "shared_in_window_treatment_columns": shared_act,
        "share_rows_with_in_window_treatment_eicu": float(np.mean(treated_test)),
        "share_rows_with_in_window_treatment_mimic": float(np.mean(treated_cal)),
        "eicu_coverage_with_in_window_treatment": patient_coverage(frame, rows[treated_test], covered[treated_test]),
        "eicu_coverage_without_in_window_treatment": patient_coverage(frame, rows[~treated_test], covered[~treated_test]),
        **{
            f"median_bin_{key}_ratio_eicu_vs_mimic": float(np.median(values)) if values else None
            for key, values in ratios.items()
        },
        "eicu_coverage_all_labels": patient_coverage(frame, rows, covered),
        "eicu_coverage_exact_labels": patient_coverage(frame, rows[exact], covered[exact]),
        "share_near_labels": float(np.mean(~exact)),
        "current_value_quantiles_mimic_10_50_90": [float(x) for x in np.nanquantile(mimic_values, (0.1, 0.5, 0.9))],
        "current_value_quantiles_eicu_10_50_90": [float(x) for x in np.nanquantile(current[rows], (0.1, 0.5, 0.9))],
    }


def mimic_masking(mimic: pd.DataFrame, absent: list[str], *, cell: str, target: str, source: str) -> dict:
    frame = mimic.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
    subjects = frame["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    rng = np.random.default_rng(stable_seed(cell, "mask_split"))
    held_out = set(rng.choice(unique, size=int(round(0.3 * len(unique))), replace=False))
    test = np.flatnonzero(np.array([subject in held_out for subject in subjects]))
    train = np.setdiff1d(np.arange(len(frame)), test)
    kwargs = {"target": target, "source": source, "seed": stable_seed(cell, "mask_fit")}
    base = fit_continuous_recipe(frame, train, test, **kwargs)
    masked_frame = frame.copy()
    masked_frame.loc[masked_frame.index[test], absent] = np.nan
    masked = fit_continuous_recipe(masked_frame, train, test, **kwargs)
    rows = np.intersect1d(base["test_rows"], masked["test_rows"])
    future = _safe_float(frame, f"future_{target}")
    widths = pd.Series(v2_widths(frame, base, target, rows), index=rows)

    def covered(fitted):
        prediction = pd.Series(fitted["test_prediction"], index=fitted["test_rows"])[rows].to_numpy()
        return np.abs(prediction - future[rows]) <= widths.to_numpy(), np.abs(prediction - future[rows])

    base_cov, base_err = covered(base)
    mask_cov, mask_err = covered(masked)
    return {
        "test_patients": int(len(held_out)),
        "coverage_full_inputs": patient_coverage(frame, rows, base_cov),
        "coverage_eicu_absent_inputs_masked": patient_coverage(frame, rows, mask_cov),
        "mae_full_inputs": float(base_err.mean()),
        "mae_masked": float(mask_err.mean()),
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# eICU Coverage Gap Decomposition (v2 intervals)",
        "",
        "Gap = eICU patient-equal coverage − 0.90. H1 = coverage change when only exact-time labels are kept.",
        "H2 = MIMIC-only coverage change when eICU-absent inputs are masked on held-out MIMIC patients.",
        "H3 = eICU/MIMIC ratio of persistence error |future − current| within the same value bin (exact labels), median and 90th percentile.",
        "H5 = eICU coverage without vs with any in-window treatment; treated share eICU / MIMIC.",
        "",
        "| Cell | eICU cov. | Gap | H1 exact-only Δ | H2 Δ (masked inputs) | H3 p50 / p90 volatility | Residual p50 / p90 | H5 cov. untreated / treated | Treated share eICU / MIMIC |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for cell in report["cells"]:
        e, m = cell["eicu"], cell["masking"]
        h1 = (e["eicu_coverage_exact_labels"] or np.nan) - e["eicu_coverage_all_labels"]
        h2 = m["coverage_eicu_absent_inputs_masked"] - m["coverage_full_inputs"]
        fmt = lambda x: "-" if x is None else f"{x:.2f}"
        lines.append(
            f"| {cell['cell']} | {e['eicu_coverage_all_labels']:.3f} | {e['eicu_coverage_all_labels'] - 0.90:+.3f} | "
            f"{h1:+.3f} | {h2:+.3f} | "
            f"{fmt(e['median_bin_p50_persistence_ratio_eicu_vs_mimic'])} / {fmt(e['median_bin_p90_persistence_ratio_eicu_vs_mimic'])} | "
            f"{fmt(e['median_bin_p50_residual_ratio_eicu_vs_mimic'])} / {fmt(e['median_bin_p90_residual_ratio_eicu_vs_mimic'])} | "
            f"{fmt(e['eicu_coverage_without_in_window_treatment'])} / {fmt(e['eicu_coverage_with_in_window_treatment'])} | "
            f"{e['share_rows_with_in_window_treatment_eicu']:.0%} / {e['share_rows_with_in_window_treatment_mimic']:.0%} |"
        )
    lines.extend(["", "`causal_claim_allowed: false`. Diagnostic only; nothing promoted.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = [row for row in development["cells"] if row["task_type"] != "binary_event_probability"]
    if args.cells:
        selected = set(args.cells.split(","))
        contracts = [row for row in contracts if row["cell"] in selected]
    by_horizon: dict[int, list[dict]] = {}
    for row in contracts:
        by_horizon.setdefault(int(row["horizon_hours"]), []).append(row)
    cells = []
    for horizon in sorted(by_horizon):
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        for contract in sorted(by_horizon[horizon], key=lambda row: row["target"]):
            cell, target, source = contract["cell"], contract["target"], contract["source_model"]
            print(f"gap diagnostic {cell}", flush=True)
            mimic, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=15000)
            eicu, info = choose_eicu_cohort(args.eicu_root, target=target, horizon=horizon, max_rows=15000)
            absent = absent_inputs(mimic, eicu, target)
            cells.append({
                "cell": cell,
                "source_model": source,
                "eicu_module": info.get("module"),
                "input_count": len(input_columns(mimic)),
                "absent_input_count": len(absent),
                "eicu": eicu_side(mimic, eicu, cell=cell, target=target, source=source),
                "masking": mimic_masking(mimic, absent, cell=cell, target=target, source=source),
            })
        del raw
    output = {
        "schema": "eicu_within_bin_gap_diagnostic.v1",
        "interval_method": "normalized_bin_scale.v2",
        "cells": cells,
        "eicu_labels_used_for_fitting_or_selection": False,
        "causal_claim_allowed": False,
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
        "--mimic-template",
        default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet",
    )
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_future_label_repaired"))
    parser.add_argument("--cells", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
