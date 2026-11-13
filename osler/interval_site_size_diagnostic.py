"""Diagnostic: where does DtACI under-cover at external sites?

For one cell and MIMIC sample 0, deploy the selected recipe in the frozen form
and report coverage by group size (hospital or care unit), by position in the
group's stream, and by patient row count.  Development databases only.
Research-only.
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
from site_adaptive_interval_dev import replay
from target_specific_domain_conformal_audit import _safe_float

TEMPLATES = {"NWICU": "nwicu_observation_transitions_{horizon}h_v5.parquet", "Zigong": "zigong_observation_transitions_{horizon}h_v2.parquet"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="database:cell:recipe,...")
    parser.add_argument("--dtaci-prior", type=Path, default=Path("site_adaptive_dtaci_prior_dev_20261019.json"))
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, point_config = frozen_point_config(Path("site_adaptive_point_dev_20261016.json"))
    _, dtaci_config = frozen_dtaci_config(Path("site_adaptive_dtaci_dev_20261017.json"), args.dtaci_prior)
    results = []
    for case in args.cases.split(","):
        database, cell, recipe = case.split(":")
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
                                       source=recipe, seed=stable_seed(cell, "mimic_to_eicu"))
        rows = fitted["test_rows"]
        all_rows = rows
        rows, values = scored_only(rows, simulate_sites(frame, rows, fitted, target, horizon, cell, 0, point_config,
                                                        dtaci_config)[DTACI_METHOD])
        observed = _safe_float(frame, f"future_{target}")[rows]
        view = frame.iloc[rows][["subject_id", "stay_id", "hospitalid", "careunit"]].reset_index(drop=True)
        view["covered"] = np.abs(values["prediction"] - observed) <= values["width"]
        view["width"] = values["width"]
        # stream position of each row within its hospital's deployment-clock replay (as in simulate_sites)
        position = pd.Series(np.nan, index=all_rows)
        all_hospitals = frame.iloc[all_rows]["hospitalid"].astype(str).to_numpy()
        for hospital in np.unique(all_hospitals):
            site_rows = replay(frame, all_rows[all_hospitals == hospital], target, stable_seed(cell, 0, hospital, "clock")).rows
            position[site_rows] = np.arange(len(site_rows))
        view["position"] = position[rows].to_numpy()
        group = "hospitalid" if database == "eICU" else "careunit"
        by_group = view.groupby(group).agg(rows=("covered", "size"), coverage=("covered", "mean"))
        by_group["size_bin"] = pd.qcut(by_group["rows"].rank(method="first"), 4, labels=["q1_small", "q2", "q3", "q4_large"]) if len(by_group) >= 4 else "all"
        size = by_group.groupby("size_bin", observed=True).agg(groups=("rows", "size"), median_rows=("rows", "median"), mean_coverage=("coverage", "mean"))
        early = view["position"] < 200
        patient_rows = view.groupby("subject_id")["covered"].transform("size")
        patient_bin = pd.cut(patient_rows, [0, 2, 5, 10, 1e9], labels=["1-2", "3-5", "6-10", ">10"])
        per_patient = view.assign(bin=patient_bin).groupby(["bin", "subject_id"], observed=True)["covered"].mean().groupby("bin", observed=True).agg(["size", "mean"])
        # Noise check: shuffle coverage across rows, keep group sizes; is the observed group-equal mean unusually low?
        rng = np.random.default_rng(0)
        codes = pd.factorize(view[group])[0]
        sizes = np.bincount(codes)
        flags = view["covered"].to_numpy(dtype=float)
        null = np.array([np.mean(np.bincount(codes, weights=rng.permutation(flags)) / sizes) for _ in range(args.permutations)])
        observed = float(by_group["coverage"].mean())
        entry = {
            "database": database, "cell": cell, "recipe": recipe,
            "group_equal_null_mean": float(null.mean()),
            "group_equal_null_2.5pct": float(np.quantile(null, 0.025)),
            "group_equal_permutation_p_low": float(np.mean(null <= observed)),
            "groups": int(len(sizes)),
            "groups_under_20_rows": int((sizes < 20).sum()),
            "patient_equal": float(view.groupby("subject_id")["covered"].mean().mean()),
            "group_equal": float(by_group["coverage"].mean()),
            "row_coverage": float(view["covered"].mean()),
            "by_group_size": size.reset_index().astype({"size_bin": str}).to_dict("records"),
            "first_200_rows_of_site": float(view.loc[early, "covered"].mean()),
            "after_200_rows_of_site": float(view.loc[~early, "covered"].mean()) if (~early).any() else None,
            "by_patient_row_count": per_patient.reset_index().astype({"bin": str}).to_dict("records"),
        }
        print(json.dumps(entry), flush=True)
        results.append(entry)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
