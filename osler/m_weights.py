"""Fitted weights of the online marginal M in its v13 form (lockbox parts B-D), for the research API.

Per MIMIC training sample K (0-4), as ``stacked_lockbox_audit.py`` scored it under
``online_tail_dev_rule_20261031.json``:

- expert anchors: the deployment recipe of each of the 36 cells (6 targets x 6
  horizons; persistence outside the catalog), fitted as ``build_expert_anchor_columns``
  fits them (MIMIC-IV v5 transitions without the JEPA cohort's MIMIC patients,
  sample K, seed ``stable_seed(cell, "mimic_to_eicu")``).  Each bundle must
  reproduce the frozen ``expert__anchor_<target>`` column on a random set of
  stays of every database in the examples (float32 tolerance of
  ``adaptive_cell_weights.reproduces``);
- own-trajectory residuals: refitted as ``joint_forecast_dev.cell_rows`` fits
  them (``osler_jepa.stacked_expert.fit_cell``); each must reproduce the frozen
  point of every MIMIC calibration row exactly;
- day-one marginal of each cell, as ``jepa_deployment_form_dev.deploy_distribution``
  derives it from the MIMIC calibration rows: point-correction scale, v2 quintile
  scale with its recording-resolution floor, ``tail8`` category edges and each
  category's day-one normalized quantiles (own, or pooled below ``MIN_PATIENTS``).
  No subject identifiers or scores are stored, only the quantiles;
- the Gaussian copula's latent correlation (``online_tail_dev.dependence_inputs``).

Research runtime only; ``causal_claim_allowed`` and ``clinical_promotion_allowed`` are false.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn

import adaptive_cell_weights
from joint_forecast_dev import cell_rows
from mimic_to_eicu_49_cell_external_audit import fit_continuous_recipe, normalize_cohort, stable_seed
from multi_sample_external_audit import mimic_sample
from online_joint_dev import frozen_multiqt
from online_tail_dev import dependence_inputs
from osler_jepa.frozen_protocol import sha256
from osler_jepa.interval_scale import MIN_PATIENTS, fit_interval_scale
from osler_jepa.lockbox_guard import read_non_lockbox
from osler_jepa.patient_split import selection_patients
from osler_jepa.predictive_distribution import base_quantiles, value_bins
from osler_jepa.stacked_expert import feature_columns, fit_cell
from site_adaptive_external_audit import frozen_point_config
from stacked_expert_dev import anchor_features, examples_path, labelled_rows, load_cohort, load_rule
from target_specific_domain_conformal_audit import add_history_features
from threshold_event_form import resolution
from whole_body_joint_gate import _split_by_column

RULE = Path("online_tail_dev_rule_20261031.json")
RECIPES = Path("deployment_recipes_20261024.json")
MIMIC_TEMPLATE = "mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet"
SOURCE_ROWS = 15000                                   # build_expert_anchor_columns.SOURCE_ROWS
VERIFY_STAYS = 150                                    # random stays per database for the anchor check


def residuals_and_marginals(frame: pd.DataFrame, sample: int, rule: dict) -> tuple[dict, dict, dict]:
    """Own-trajectory residual and day-one marginal per cell, and the copula; ``frame``: the MIMIC rows of the cohort."""

    targets, horizons = tuple(rule["data"]["targets"]), tuple(rule["data"]["horizons_hours"])
    lags = tuple(rule["trajectory"]["lags_hours"])
    frozen = cell_rows(frame, sample, {**rule, "marginal": {"system": "own_trajectory"}})
    features = anchor_features(frame, targets, horizons, lags)
    _, calibration_index = _split_by_column(frame, "subject_id", stable_seed("stacked_expert_split", sample), 0.20)
    calibration_subjects = set(frame["subject_id"].iloc[calibration_index])
    residuals, marginals = {}, {}
    for target in targets:
        for horizon in horizons:
            cell = f"{target}@{horizon}h"
            rows = labelled_rows(frame, target, horizon)
            x = features.reindex(pd.MultiIndex.from_arrays([rows["stay_id"], rows["t"]])).reset_index(drop=True)
            fit = ~rows["subject_id"].isin(calibration_subjects).to_numpy()
            columns = feature_columns(target, targets, horizons, lags, exchange=False)
            model = fit_cell(x[fit], rows["anchor"].to_numpy()[fit], rows["future"].to_numpy()[fit], columns)
            point = model.predict(x, rows["anchor"].to_numpy())[~fit]
            if not np.array_equal(point, frozen[cell]["point"].to_numpy(), equal_nan=True):
                raise RuntimeError(f"{cell} sample {sample}: refitted residual differs from the frozen point")
            residuals[cell] = model
            marginals[cell] = day_one_marginal(frozen[cell])
    correlation, _ = dependence_inputs(frozen)
    return residuals, marginals, {"cells": list(frozen), "correlation": correlation}


def day_one_marginal(rows: pd.DataFrame) -> dict:
    """``deploy_distribution``'s calibration-side quantities for one cell, read from its MIMIC calibration rows."""

    signed = rows["future"].to_numpy() - rows["point"].to_numpy()
    subjects, current = rows["subject_id"].to_numpy(), rows["current"].to_numpy()
    choosing = selection_patients(subjects)
    no_change = lambda n: np.full(n, np.nan)
    fitted = fit_interval_scale("quintile", current[choosing], no_change(int(choosing.sum())), np.abs(signed[choosing]),
                                subjects[choosing])
    floor = resolution(current)
    scores = signed[~choosing] / np.maximum(fitted(current[~choosing], no_change(int((~choosing).sum()))), floor)
    edges = value_bins(current)
    groups, conformal_subjects = np.digitize(current[~choosing], edges), subjects[~choosing].astype(str)
    levels = np.asarray(frozen_multiqt().levels)
    bases = {}
    for group in range(len(edges) + 1):
        own = groups == group
        if len(np.unique(conformal_subjects[own])) < MIN_PATIENTS:
            own = np.ones(len(scores), dtype=bool)
        bases[group] = base_quantiles(scores[own], np.ones(int(own.sum())), levels)
    return {"point_scale": float(np.median(np.abs(signed[choosing]))), "scale": fitted, "floor": float(floor),
            "edges": edges, "bases": bases}


def verification_events(path: Path, sample: int) -> pd.DataFrame:
    """Every example row (all horizons) of ``VERIFY_STAYS`` random stays per database, without provenance strings."""

    keys = read_non_lockbox(path, columns=["database", "stay_id"]).drop_duplicates()
    rng = np.random.default_rng(stable_seed("m_weights_verify", sample))
    chosen = pd.concat([group.sample(min(VERIFY_STAYS, len(group)), random_state=rng.integers(2**31))
                        for _, group in keys.groupby("database")])
    names = [n for n in pq.read_schema(path).names if not n.startswith("_source_")]
    events = read_non_lockbox(path, columns=names, filters=[("stay_id", "in", sorted(set(chosen["stay_id"])))])
    wanted = set(zip(chosen["database"], chosen["stay_id"]))
    keep = np.array([pair in wanted for pair in zip(events["database"], events["stay_id"])])
    return events[keep].reset_index(drop=True)


def expert_bundle(events: pd.DataFrame, mimic_raw: pd.DataFrame, target: str, horizon: int, recipe: str,
                  exclude_subjects: set[str], sample: int) -> tuple[dict, float]:
    """The anchor recipe fitted as ``build_expert_anchor_columns.anchor_predictions`` fits it, and the largest
    difference from the frozen anchor column on ``events`` (one horizon)."""

    cell = f"{target}@{horizon}h"
    frozen_anchor = pd.to_numeric(events[f"expert__anchor_{target}"], errors="coerce").to_numpy()
    if recipe == "persistence":
        bundle = {"recipe": recipe, "target": target}
        current = pd.to_numeric(events[f"{target}_t"], errors="coerce").to_numpy()
        return bundle, float(np.nanmax(np.abs(current - frozen_anchor), initial=0.0))
    full = normalize_cohort(mimic_raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)[0]
    full = full[~full["subject_id"].str.removeprefix("MIMIC:").isin(exclude_subjects)].reset_index(drop=True)
    train = mimic_sample(full, target=target, horizon=horizon, sample=sample, rows=SOURCE_ROWS)
    rows = events.copy()
    rows["_event_index"] = rows.index
    rows["subject_id"] = "EVENT:" + rows["subject_id"].astype(str)
    rows["stay_id"] = "EVENT:" + rows["database"].astype(str) + ":" + rows["stay_id"].astype(str)
    rows["anchor_time"] = pd.to_datetime(rows["anchor_time"])
    rows = add_history_features(rows)
    rows = rows[pd.to_numeric(events[f"{target}_t"], errors="coerce").notna()]
    frame = pd.concat([train, rows], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    subject = frame["subject_id"].astype(str)
    mimic_rows, event_rows = np.flatnonzero(subject.str.startswith("MIMIC:")), np.flatnonzero(subject.str.startswith("EVENT:"))
    seed = stable_seed(cell, "mimic_to_eicu")
    fitted = fit_continuous_recipe(frame, mimic_rows, event_rows, target=target, source=recipe, seed=seed,
                                   require_test_labels=False)
    bundle = adaptive_cell_weights.fit_predictor(frame, mimic_rows, target=target, recipe=recipe, seed=seed,
                                                 policy=fitted.get("mimic_selected_policy"))
    predicted = adaptive_cell_weights.predict(bundle, frame.iloc[fitted["test_rows"]].reset_index(drop=True))
    if not adaptive_cell_weights.reproduces(predicted, fitted["test_prediction"]):
        raise RuntimeError(f"{cell} sample {sample}: exported recipe differs from the frozen fit")
    index = frame.loc[fitted["test_rows"], "_event_index"].astype(int).to_numpy()
    if not adaptive_cell_weights.reproduces(predicted, frozen_anchor[index]):
        raise RuntimeError(f"{cell} sample {sample}: exported recipe differs from the frozen anchor column")
    return bundle, float(np.max(np.abs(predicted - frozen_anchor[index]), initial=0.0))


def export_sample(sample: int, rule: dict, recipes: dict) -> tuple[dict, dict]:
    started = time.time()
    path = examples_path(sample)
    targets, horizons = tuple(rule["data"]["targets"]), tuple(rule["data"]["horizons_hours"])
    cohort = load_cohort(path, targets, horizons)
    cohort = cohort[cohort["database"] == rule["data"]["training_database"]].reset_index(drop=True)
    residuals, marginals, copula = residuals_and_marginals(cohort, sample, rule)
    print(json.dumps({"sample": sample, "residuals": len(residuals), "seconds": round(time.time() - started)}), flush=True)
    del cohort
    keys = read_non_lockbox(path, columns=["database", "subject_id"])
    exclude = set(keys.loc[keys["database"].astype(str).str.lower() == "mimiciv", "subject_id"].astype(str))
    events = verification_events(path, sample)
    experts, differences = {}, {}
    for horizon in horizons:
        mimic_raw = read_non_lockbox(MIMIC_TEMPLATE.format(horizon=horizon))
        at_horizon = events[events["horizon_hours"] == float(horizon)].reset_index(drop=True)   # positions = labels
        for target in targets:
            cell = f"{target}@{horizon}h"
            experts[cell], differences[cell] = expert_bundle(at_horizon, mimic_raw, target, horizon,
                                                             recipes.get(cell, "persistence"), exclude, sample)
        del mimic_raw
        print(json.dumps({"sample": sample, "horizon": horizon, "seconds": round(time.time() - started)}), flush=True)
    bundle = {"sample": sample, "targets": list(targets), "horizons": list(horizons),
              "lags": list(rule["trajectory"]["lags_hours"]), "experts": experts, "residuals": residuals,
              "marginals": marginals, "copula": copula}
    check = {"sample": sample, "verify_stays_per_database": VERIFY_STAYS, "verify_rows": int(len(events)),
             "anchor_max_abs_difference": differences, "residual_points": "identical to joint_forecast_dev.cell_rows"}
    return bundle, check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("m_weights_20261108"))
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    rule, recipes = load_rule(RULE), json.loads(RECIPES.read_text(encoding="utf-8"))["recipes"]
    args.output.mkdir(exist_ok=False)
    entries = []
    for sample in args.samples:
        bundle, check = export_sample(sample, rule, recipes)
        path = args.output / f"m_sample{sample}.joblib"
        joblib.dump(bundle, path, compress=3)
        entries.append({**check, "file": path.name, "sha256": sha256(path)})
        print(json.dumps({"sample": sample, "file": path.name,
                          "anchor_max": max(check["anchor_max_abs_difference"].values())}), flush=True)
    configs = [RULE, Path(rule["inherits"]), RECIPES, Path("threshold_event_candidates_20261029.json"),
               Path("site_adaptive_point_amendment_20261024.json")]
    manifest = {
        "schema": "m_weights.v1",
        "form": "v13 lockbox parts B-D: expert anchor -> own-trajectory residual -> online point correction + Mondrian "
                "MultiQT per hospital -> Gaussian copula",
        "configs_sha256": {str(p): sha256(p) for p in configs},
        "examples_sha256": {examples_path(s).name: sha256(examples_path(s)) for s in args.samples},
        "point_config": repr(frozen_point_config(Path("site_adaptive_point_amendment_20261024.json"))),
        "multiqt": repr(frozen_multiqt()),
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "joblib": joblib.__version__},
        "samples": entries,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
