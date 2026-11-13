"""Does blood-gas glucose explain the eICU short-horizon glucose gap?

MIMIC-IV laboratory glucose mixes chemistry (50931 and variants) with blood-gas
glucose (50809, about 25% of ICU glucose results); eICU laboratory glucose is
chemistry only.  For each glucose cell this script fits the frozen recipe on 70%
of MIMIC patients, applies the v2 interval to the held-out 30%, attributes the
current value and the label to their item source, and reports composition and
coverage by source pair.  MIMIC-only; no external labels are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_within_bin_gap_diagnostic import v2_widths
from mimic_to_eicu_49_cell_external_audit import (
    fit_continuous_recipe,
    group_equal_coverage,
    normalize_cohort,
    stable_seed,
)
from target_specific_domain_conformal_audit import _safe_float


BLOOD_GAS_ITEMS = {50809}


def attribute(frame: pd.DataFrame, rows: np.ndarray, items: pd.DataFrame, times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Item source of the glucose result at the given time with the given value."""

    query = pd.DataFrame({
        "position": np.arange(len(rows)),
        "stay_id": pd.to_numeric(frame.iloc[rows]["stay_id"].astype(str).str.split(":").str[-1], errors="coerce"),
        "charttime": times,
        "value": values,
    }).dropna()
    query = query.astype({"stay_id": "int64"})
    joined = query.merge(items, on=["stay_id", "charttime"], how="inner")
    joined = joined[np.isclose(joined["v"], joined["value"], atol=1e-6)]
    source = joined.assign(source=np.where(joined["itemid"].isin(BLOOD_GAS_ITEMS), "blood_gas", "chemistry"))
    source = source.groupby("position")["source"].agg(lambda item: "+".join(sorted(set(item))))
    output = np.full(len(rows), "unmatched", dtype=object)
    output[source.index.to_numpy()] = source.to_numpy()
    return output


def label_times(frame: pd.DataFrame, rows: np.ndarray, items: pd.DataFrame, horizon: int, values: np.ndarray) -> np.ndarray:
    tolerance = pd.Timedelta(hours=min(2.0, horizon / 2.0))
    query = pd.DataFrame({
        "position": np.arange(len(rows)),
        "stay_id": pd.to_numeric(frame.iloc[rows]["stay_id"].astype(str).str.split(":").str[-1], errors="coerce"),
        "target": pd.to_datetime(frame.iloc[rows]["anchor_time"]).to_numpy() + pd.Timedelta(hours=horizon),
        "value": values,
    }).dropna().astype({"stay_id": "int64"})
    joined = query.merge(items, on="stay_id", how="inner")
    joined = joined[(joined["charttime"] - joined["target"]).abs() <= tolerance]
    joined = joined[np.isclose(joined["v"], joined["value"], atol=1e-6)]
    joined = joined.assign(distance=(joined["charttime"] - joined["target"]).abs())
    joined = joined.sort_values(["position", "distance"]).drop_duplicates("position")
    output = np.full(len(rows), np.datetime64("NaT"), dtype="datetime64[ns]")
    output[joined["position"].to_numpy()] = joined["charttime"].to_numpy()
    return output


def run(args: argparse.Namespace) -> dict:
    items = pd.read_parquet(args.items)
    items["charttime"] = pd.to_datetime(items["charttime"]).astype("datetime64[ns]")
    development = {row["cell"]: row for row in json.loads(args.development_report.read_text())["cells"]}
    cells = []
    for horizon in (1, 3, 6, 12, 24, 48):
        cell = f"glucose@{horizon}h"
        source_model = development[cell]["source_model"]
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        mimic, _ = normalize_cohort(raw, database="MIMIC-IV", target="glucose", horizon=horizon, max_rows=15000)
        del raw
        frame = mimic.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        subjects = frame["subject_id"].astype(str).to_numpy()
        unique = np.unique(subjects)
        held = set(np.random.default_rng(stable_seed(cell, "glucose_source_split")).choice(unique, size=len(unique) * 3 // 10, replace=False))
        test = np.flatnonzero(np.isin(subjects, list(held)))
        train = np.setdiff1d(np.arange(len(frame)), test)
        fitted = fit_continuous_recipe(frame, train, test, target="glucose", source=source_model, seed=stable_seed(cell, "mimic_to_eicu"))
        rows = fitted["test_rows"]
        future = _safe_float(frame, "future_glucose")[rows]
        current = _safe_float(frame, "glucose_t")[rows]
        covered = np.abs(fitted["test_prediction"] - future) <= v2_widths(frame, fitted, "glucose", rows)
        current_time = (
            pd.to_datetime(frame.iloc[rows]["anchor_time"]).to_numpy().astype("datetime64[ns]")
            - (_safe_float(frame, "glucose_age_hr")[rows] * 3.6e12).astype("timedelta64[ns]")
        )
        current_source = attribute(frame, rows, items, pd.Series(current_time).dt.round("min").to_numpy(), current)
        label_source = attribute(frame, rows, items, label_times(frame, rows, items, horizon, future), future)
        pair = np.array([f"{a}->{b}" for a, b in zip(current_source, label_source)], dtype=object)
        groups = {}
        for name in sorted(set(pair)):
            mask = pair == name
            groups[name] = {
                "share": float(np.mean(mask)),
                "rows": int(mask.sum()),
                "coverage": group_equal_coverage(frame, rows[mask], covered[mask], "subject_id") if mask.sum() >= 30 else None,
            }
        cells.append({
            "cell": cell,
            "heldout_rows": int(len(rows)),
            "heldout_coverage": group_equal_coverage(frame, rows, covered, "subject_id"),
            "pairs": groups,
        })
        print(json.dumps({"cell": cell, "coverage": round(cells[-1]["heldout_coverage"], 3),
                          "pairs": {k: [round(v["share"], 3), None if v["coverage"] is None else round(v["coverage"], 3)] for k, v in groups.items()}}), flush=True)
    output = {"schema": "glucose_source_mimic_diagnostic.v1", "cells": cells, "external_data_read": False}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"))
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v3.parquet")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
