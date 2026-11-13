"""Does eICU bedside (point-of-care) glucose explain the glucose coverage gap?

MIMIC-IV glucose comes from laboratory and blood-gas items only.  The eICU
extraction merges ``bedside glucose`` (point-of-care) with laboratory
``glucose``.  This diagnostic re-reads the raw eICU lab table, attributes each
row's current value and future label to its source, and scores the frozen
MIMIC recipe with v2 value-conditional intervals by source.

eICU labels are used for analysis only.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_within_bin_gap_diagnostic import v2_widths
from mimic_to_eicu_49_cell_external_audit import (
    choose_eicu_cohort,
    fit_continuous_recipe,
    group_equal_coverage,
    normalize_cohort,
    stable_seed,
)
from target_specific_domain_conformal_audit import _safe_float


BASE_TIME = pd.Timestamp("2100-01-01")
GLUCOSE_LABS = {"bedside glucose": "bedside", "glucose": "laboratory"}


def load_glucose_sources(eicu_raw: Path) -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(
        eicu_raw / "lab.csv.gz",
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
        chunksize=5_000_000,
    ):
        chunk = chunk[chunk["labname"].isin(GLUCOSE_LABS)]
        pieces.append(chunk)
    labs = pd.concat(pieces, ignore_index=True).dropna(subset=["labresult"])
    labs["source"] = labs["labname"].map(GLUCOSE_LABS)
    return labs.rename(columns={"patientunitstayid": "stay", "labresultoffset": "offset"})


def attribute(rows: pd.DataFrame, labs: pd.DataFrame, offset: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Source of the measurement at ``offset`` minutes whose value matches; else 'unmatched'."""

    query = pd.DataFrame({
        "position": np.arange(len(rows)),
        "stay": pd.to_numeric(rows["stay_id"].astype(str).str.split(":").str[-1], errors="coerce"),
        "offset": np.round(offset).astype("float64"),
        "value": value,
    }).dropna(subset=["stay", "offset", "value"])
    query = query.astype({"stay": "int64", "offset": "int64"})
    joined = query.merge(labs, on=["stay", "offset"], how="inner")
    joined = joined[np.isclose(joined["labresult"], joined["value"], atol=1e-6)]
    sources = joined.groupby("position")["source"].agg(lambda item: "+".join(sorted(set(item))))
    output = np.full(len(rows), "unmatched", dtype=object)
    output[sources.index.to_numpy()] = sources.to_numpy()
    return output


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = [row for row in development["cells"] if row["target"] == "glucose"]
    labs = load_glucose_sources(args.eicu_raw)
    cells = []
    for contract in sorted(contracts, key=lambda row: int(row["horizon_hours"])):
        cell, horizon, source = contract["cell"], int(contract["horizon_hours"]), contract["source_model"]
        print(f"glucose source diagnostic {cell}", flush=True)
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        mimic, _ = normalize_cohort(raw, database="MIMIC-IV", target="glucose", horizon=horizon, max_rows=15000)
        del raw
        eicu, info = choose_eicu_cohort(args.eicu_root, target="glucose", horizon=horizon, max_rows=15000)
        frame = pd.concat([mimic, eicu], ignore_index=True, sort=False)
        frame.sort_values(["subject_id", "anchor_time"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        train = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("MIMIC:"))
        test = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("EICU:"))
        fitted = fit_continuous_recipe(
            frame, train, test, target="glucose", source=source, seed=stable_seed(cell, "mimic_to_eicu")
        )
        rows = fitted["test_rows"]
        view = frame.iloc[rows]
        future = _safe_float(frame, "future_glucose")[rows]
        current = _safe_float(frame, "glucose_t")[rows]
        covered = np.abs(fitted["test_prediction"] - future) <= v2_widths(frame, fitted, "glucose", rows)
        anchor_minutes = (pd.to_datetime(view["anchor_time"]) - BASE_TIME).dt.total_seconds().to_numpy() / 60.0
        current_source = attribute(view, labs, anchor_minutes - _safe_float(view, "glucose_age_hr") * 60.0, current)
        label_source = attribute(view, labs, _safe_float(view, "future_event_time_hr_glucose") * 60.0, future)
        groups = {}
        for name in sorted(set(label_source)):
            mask = label_source == name
            groups[f"label:{name}"] = {
                "rows": int(mask.sum()),
                "coverage": group_equal_coverage(frame, rows[mask], covered[mask], "subject_id") if mask.any() else None,
            }
        for current_name in ("bedside", "laboratory"):
            for label_name in ("bedside", "laboratory"):
                mask = (current_source == current_name) & (label_source == label_name)
                groups[f"current:{current_name}->label:{label_name}"] = {
                    "rows": int(mask.sum()),
                    "coverage": group_equal_coverage(frame, rows[mask], covered[mask], "subject_id") if mask.sum() >= 30 else None,
                }
        cells.append({
            "cell": cell,
            "eicu_module": info.get("module"),
            "rows": int(len(rows)),
            "coverage_all": group_equal_coverage(frame, rows, covered, "subject_id"),
            "share_label_bedside": float(np.mean(label_source == "bedside")),
            "share_current_bedside": float(np.mean(current_source == "bedside")),
            "share_unmatched_label": float(np.mean(label_source == "unmatched")),
            "groups": groups,
        })
    output = {"schema": "eicu_glucose_source_diagnostic.v1", "cells": cells, "causal_claim_allowed": False}
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
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_future_label_repaired"))
    parser.add_argument(
        "--eicu-raw", type=Path,
        default=Path.home() / "Downloads/eicu-collaborative-research-database-2.0",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    for cell in result["cells"]:
        print(json.dumps({k: cell[k] for k in ("cell", "coverage_all", "share_label_bedside", "share_unmatched_label")}))
        for name, item in cell["groups"].items():
            if item["coverage"] is not None:
                print(f"   {name:40s} rows {item['rows']:6d} coverage {item['coverage']:.3f}")
