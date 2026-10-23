"""Diagnostic: is external glucose under-coverage concentrated in volatile patients?

Deploys the current frozen form (selected recipe, point recalibration, DtACI
with the frozen starting multiplier) for MIMIC sample 0 and reports coverage by
pre-anchor volatility markers: |change since the previous anchor|, recent
point-of-care glucose, insulin exposure, and the current value; plus, per
hospital/care unit, coverage against the group's share of volatile rows.
Inputs only define the strata.  Development databases only.  Research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_recipe_selection_dev import PREFIX, SOURCE_ROWS, TEST_ROWS
from mimic_to_eicu_49_cell_external_audit import choose_eicu_cohort, fit_continuous_recipe, normalize_cohort, stable_seed
from multi_sample_external_audit import mimic_sample
from site_adaptive_external_audit import DTACI_METHOD, frozen_dtaci_config, frozen_point_config, scored_only, simulate_sites
from target_specific_domain_conformal_audit import _feature_frame, _safe_float

TEMPLATES = {"NWICU": "nwicu_observation_transitions_{horizon}h_v5.parquet"}


def strata(name: str, values: pd.Series, covered: pd.Series) -> dict:
    table = covered.groupby(values, observed=True).agg(["size", "mean"])
    return {"stratum": name, "rows": {str(k): [int(v["size"]), round(float(v["mean"]), 3)] for k, v in table.iterrows()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="database:cell,...")
    parser.add_argument("--recipes", type=Path, default=Path("external_recipe_selection_dev_20261019.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipes = json.loads(args.recipes.read_text(encoding="utf-8"))["recipes"]
    _, point_config = frozen_point_config(Path("site_adaptive_point_dev_20261016.json"))
    _, dtaci_config = frozen_dtaci_config(Path("site_adaptive_dtaci_dev_20261017.json"), Path("site_adaptive_dtaci_prior_dev_20261019.json"))
    results = []
    for case in args.cases.split(","):
        database, cell = case.split(":")
        target, horizon = cell.split("@")[0], int(cell.split("@")[1].rstrip("h"))
        full = normalize_cohort(pd.read_parquet(f"mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet"),
                                database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)[0]
        if database == "eICU":
            site = choose_eicu_cohort(Path("eicu_harmonized_20261022"), target=target, horizon=horizon, max_rows=TEST_ROWS)[0]
        else:
            site = normalize_cohort(pd.read_parquet(TEMPLATES[database].format(horizon=horizon)), database=database,
                                    target=target, horizon=horizon, max_rows=TEST_ROWS)[0]
        train = mimic_sample(full, target=target, horizon=horizon, sample=0, rows=SOURCE_ROWS)
        frame = pd.concat([train, site], ignore_index=True, sort=False).sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        subject = frame["subject_id"].astype(str)
        fitted = fit_continuous_recipe(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                                       np.flatnonzero(subject.str.startswith(PREFIX[database])), target=target,
                                       source=recipes[cell], seed=stable_seed(cell, "mimic_to_eicu"))
        rows = fitted["test_rows"]
        rows, values = scored_only(rows, simulate_sites(frame, rows, fitted, target, horizon, cell, 0, point_config,
                                                        dtaci_config)[DTACI_METHOD])
        observed = _safe_float(frame, f"future_{target}")[rows]
        features = _feature_frame(frame).iloc[rows].reset_index(drop=True)
        view = frame.iloc[rows][["subject_id", "hospitalid", "careunit"]].reset_index(drop=True)
        covered = pd.Series(np.abs(values["prediction"] - observed) <= values["width"])
        change = (features[f"{target}_t"] - features[f"lag1__{target}_t"]).abs()
        poc_recent = features.get("glucose_poc_age_hr", pd.Series(np.nan, index=features.index)) <= 2
        insulin = pd.to_numeric(frame.iloc[rows].get("hist_insulin", pd.Series(np.nan, index=rows)), errors="coerce").reset_index(drop=True)
        current = features[f"{target}_t"]
        volatile = change > 50
        group = "hospitalid" if database == "eICU" else "careunit"
        per_group = pd.DataFrame({"group": view[group], "covered": covered, "volatile": volatile, "no_lag": change.isna()}) \
            .groupby("group").agg(rows=("covered", "size"), coverage=("covered", "mean"),
                                  volatile_share=("volatile", "mean"), no_lag_share=("no_lag", "mean"))
        big = per_group[per_group["rows"] >= 20]
        entry = {
            "database": database, "cell": cell, "recipe": recipes[cell], "rows": int(len(rows)),
            "row_coverage": round(float(covered.mean()), 3),
            "strata": [
                strata("abs_change_since_previous_anchor",
                       pd.cut(change, [-0.1, 10, 25, 50, 100, 1e9]).astype(str).where(change.notna(), "no_previous_anchor"), covered),
                strata("poc_glucose_within_2h", poc_recent.map({True: "yes", False: "no"}), covered),
                strata("insulin_recorded_before_anchor", insulin.fillna(0).gt(0).map({True: "yes", False: "no"}), covered),
                strata("current_value", pd.cut(current, [0, 100, 140, 180, 250, 1e9]).astype(str), covered),
            ],
            "groups_with_20_rows": len(big),
            "corr_group_coverage_vs_volatile_share": round(float(big["coverage"].corr(big["volatile_share"])), 3) if len(big) > 2 else None,
            "corr_group_coverage_vs_no_lag_share": round(float(big["coverage"].corr(big["no_lag_share"])), 3) if len(big) > 2 else None,
            "lowest_groups": big.sort_values("coverage").head(6).round(3).reset_index().to_dict("records"),
        }
        print(json.dumps(entry), flush=True)
        results.append(entry)
    args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
