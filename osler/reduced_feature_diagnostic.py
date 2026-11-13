"""Diagnostic: imputed-missing features vs a reduced-feature refit at the deployment site.

For selected cells, the deployed recipe is fit on MIMIC (5 samples) either with
all MIMIC features (site-missing ones median-imputed, the current behaviour) or
with only the raw input columns the site records (>= 1% non-missing), and
deployed in the frozen form.  Development databases only.  Research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_recipe_selection_dev import TEST_ROWS, SOURCE_ROWS, deploy
from mimic_to_eicu_49_cell_external_audit import choose_eicu_cohort, normalize_cohort
from multi_sample_external_audit import mimic_sample
from site_adaptive_external_audit import frozen_dtaci_config, frozen_point_config

TEMPLATES = {"NWICU": "nwicu_observation_transitions_{horizon}h_v4.parquet", "Zigong": "zigong_observation_transitions_{horizon}h.parquet"}
MIN_SHARE = 0.01


def feature_inputs(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if (c.endswith("_t") and c != "t") or c.endswith("_age_hr") or c.startswith("hist_")]


def unavailable(site: pd.DataFrame, train: pd.DataFrame, target: str) -> list[str]:
    keep = {f"{target}_t"}
    return [c for c in feature_inputs(train) if c not in keep
            and (c not in site or pd.to_numeric(site[c], errors="coerce").notna().mean() < MIN_SHARE)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="database:cell:recipe,...")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, point_config = frozen_point_config(Path("site_adaptive_point_dev_20261016.json"))
    _, dtaci_config = frozen_dtaci_config(Path("site_adaptive_dtaci_dev_20261017.json"))
    results = []
    for case in args.cases.split(","):
        database, cell, recipe = case.split(":")
        target, horizon = cell.split("@")[0], int(cell.split("@")[1].rstrip("h"))
        full = normalize_cohort(pd.read_parquet(f"mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v4.parquet"),
                                database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)[0]
        if database == "eICU":
            site = choose_eicu_cohort(Path("eicu_harmonized_20261015"), target=target, horizon=horizon, max_rows=TEST_ROWS)[0]
        else:
            site = normalize_cohort(pd.read_parquet(TEMPLATES[database].format(horizon=horizon)), database=database,
                                    target=target, horizon=horizon, max_rows=TEST_ROWS)[0]
        entry = {"database": database, "cell": cell, "recipe": recipe}
        for variant in ("imputed", "reduced"):
            samples = []
            for sample in range(5):
                train = mimic_sample(full, target=target, horizon=horizon, sample=sample, rows=SOURCE_ROWS)
                test = site
                if variant == "reduced":
                    drop = unavailable(site, train, target)
                    train = train.assign(**{c: np.nan for c in drop})
                    test = site.assign(**{c: np.nan for c in drop if c in site})
                    entry["dropped_inputs"] = len(drop)
                samples.append(deploy(train, test, database=database, cell=cell, target=target, horizon=horizon,
                                      recipe=recipe, sample=sample, point_config=point_config, dtaci_config=dtaci_config))
            entry[variant] = {k: float(np.mean([s[k] for s in samples])) for k in samples[0]}
        print(json.dumps(entry), flush=True)
        results.append(entry)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
