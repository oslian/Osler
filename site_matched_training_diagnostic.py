"""Diagnostic: does an importance-resampled (site-matched) MIMIC training sample help point prediction?

For each case, MIMIC sample 0 (uniform, the frozen form) is compared with a
site-matched sample of the same size: MIMIC subjects drawn without replacement
with probability proportional to their mean covariate-shift weight, from a
MIMIC-vs-site classifier on the site's day-one inputs only (first min(200, 25%)
stays; no outcomes), restricted to covariates the site records.  Both are
deployed in the frozen form.  Development databases only.  Research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_recipe_selection_dev import PREFIX, SOURCE_ROWS, TEST_ROWS
from external_value_conditional_interval_audit import GATE_AXES, interval_report
from mimic_to_eicu_49_cell_external_audit import choose_eicu_cohort, fit_continuous_recipe, normalize_cohort, paired_comparison, stable_seed
from multi_sample_external_audit import mimic_sample
from osler_jepa.recipe_availability import deployable_recipe
from osler_jepa.site_adaptive_conformal import covariate_shift_weights
from site_adaptive_external_audit import (DTACI_METHOD, frozen_dtaci_config, frozen_network_config, frozen_point_config,
                                          scored_only, simulate_sites)
from site_adaptive_interval_dev import COVARIATES, replay
from target_specific_domain_conformal_audit import _safe_float

TEMPLATES = {"NWICU": "nwicu_observation_transitions_{horizon}h_v5.parquet", "Zigong": "zigong_observation_transitions_{horizon}h_v2.parquet"}
MIN_SHARE = 0.01


def recorded_covariates(site: pd.DataFrame, target: str) -> list[str]:
    columns = [f"{target}_t", f"{target}_age_hr"] + [f"{var}_t" for var in COVARIATES if var != target]
    return [c for c in columns if c in site and pd.to_numeric(site[c], errors="coerce").notna().mean() >= MIN_SHARE]


def matrix(frame: pd.DataFrame, columns: list[str], reference: pd.DataFrame) -> np.ndarray:
    output = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) if column in frame else np.full(len(frame), np.nan)
        ref = pd.to_numeric(reference[column], errors="coerce").dropna().to_numpy()
        center = float(np.median(ref)) if len(ref) else 0.0
        spread = float(np.subtract(*np.percentile(ref, [75, 25]))) if len(ref) else 1.0
        spread = spread if spread > 0 else 1.0
        observed = np.isfinite(values)
        output.append(np.where(observed, (values - center) / spread, 0.0))
        output.append(observed.astype(float))
    return np.column_stack(output)


def site_matched_sample(full: pd.DataFrame, site: pd.DataFrame, target: str, seed: int) -> tuple[pd.DataFrame, dict]:
    day_one = site.iloc[replay(site, np.arange(len(site)), target, seed).day_one]
    columns = recorded_covariates(site, target)
    weights = covariate_shift_weights(matrix(full, columns, full), matrix(day_one, columns, full))
    per_subject = pd.Series(weights, index=full.index).groupby(full["subject_id"].astype(str)).mean()
    rng = np.random.default_rng(seed)
    subjects = rng.choice(per_subject.index.to_numpy(), size=len(per_subject), replace=False, p=(per_subject / per_subject.sum()).to_numpy())
    counts = full.groupby(full["subject_id"].astype(str)).size()
    keep = subjects[: int(np.searchsorted(np.cumsum(counts[subjects].to_numpy()), SOURCE_ROWS)) + 1]
    effective = float(weights.sum() ** 2 / np.sum(weights ** 2))
    return full[full["subject_id"].astype(str).isin(set(keep))].reset_index(drop=True), {"covariates": len(columns), "effective_sample_size": effective}


def deploy(train, site, database, cell, target, horizon, recipe, point_config, dtaci_config, network_config):
    frame = pd.concat([train, site], ignore_index=True, sort=False).sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
    subject = frame["subject_id"].astype(str)
    seed = stable_seed(cell, "mimic_to_eicu")
    fitted = fit_continuous_recipe(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                                   np.flatnonzero(subject.str.startswith(PREFIX[database])), target=target, source=recipe, seed=seed)
    rows = fitted["test_rows"]
    rows, values = scored_only(rows, simulate_sites(frame, rows, fitted, target, horizon, cell, 0, point_config, dtaci_config,
                                                    labels=(DTACI_METHOD,), network_config=network_config)[DTACI_METHOD])
    observed = _safe_float(frame, f"future_{target}")[rows]
    current = _safe_float(frame, f"{target}_t")[rows]
    loss = np.abs(values["prediction"] - observed)
    persistence = paired_comparison(frame, rows, loss, np.abs(current - observed), seed=seed)["patient"]
    comparator = paired_comparison(frame, rows, loss, np.abs(values["comparator"] - observed), seed=seed + 2)["patient"]
    report = interval_report(frame, rows, loss <= values["width"], values["width"], GATE_AXES[database]["interval"])
    return {"mae_over_persistence": round(float(loss.mean() / np.abs(current - observed).mean()), 4),
            "vs_persistence_ci": [round(persistence["bootstrap_low"], 4), round(persistence["bootstrap_high"], 4)],
            "point_pass_patient_axis": bool(persistence["pass"] and comparator["pass"]),
            "interval_pass": report["pass"], "coverage": [round(report[a], 3) for a in report["gate_axes"]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="database:cell,...")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipes = json.loads(Path("external_recipe_selection_dev_20261019.json").read_text())["recipes"]
    _, point_config = frozen_point_config(Path("site_adaptive_point_dev_20261016.json"))
    _, dtaci_config = frozen_dtaci_config(Path("site_adaptive_dtaci_dev_20261017.json"), Path("site_adaptive_dtaci_prior_dev_20261019.json"))
    network_config = frozen_network_config(Path("site_adaptive_network_dev_20261020.json"), dtaci_config)
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
        recipe, _ = deployable_recipe(recipes[cell], site, target)
        if recipe == "persistence":
            continue
        uniform = mimic_sample(full, target=target, horizon=horizon, sample=0, rows=SOURCE_ROWS)
        matched, info = site_matched_sample(full, site, target, stable_seed(cell, database, "site_matched"))
        entry = {"database": database, "cell": cell, "recipe": recipe, **info,
                 "uniform": deploy(uniform, site, database, cell, target, horizon, recipe, point_config, dtaci_config, network_config),
                 "site_matched": deploy(matched, site, database, cell, target, horizon, recipe, point_config, dtaci_config, network_config)}
        print(json.dumps(entry), flush=True)
        results.append(entry)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
