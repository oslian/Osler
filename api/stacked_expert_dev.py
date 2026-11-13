"""Development gate of the stacked expert on eICU / NWICU / Zigong (``--rule``; schema stacked_expert_dev_rule.v2).

Per MIMIC training sample K: the JEPA event-example cohort (same rows and
labels as the JEPA export, so the two are comparable), frozen expert anchors
fitted outside the cohort, one fixed 80/20 patient split of the MIMIC rows; per
target x horizon one residual (``osler_jepa.stacked_expert``) per system of the
rule (exchange across targets or own target only; with or without each
target's trajectory) is fitted on the fit patients.  Every system and the
anchor go through the frozen deployment form (``jepa_deployment_form_dev.deploy``)
on each test database; the rule's candidate passes there if it beats every
required comparator on the database's point axes and its interval gate passes.
``--summarize`` combines the five samples (robust = passing in >= 80%).
Development evidence only; research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from external_value_conditional_interval_audit import GATE_AXES, interval_report
from jepa_deployment_form_dev import deploy
from jepa_gate import DATABASE_NAMES, MIN_ROWS, MIN_SUBJECTS, ROBUST_SHARE
from mimic_to_eicu_49_cell_external_audit import paired_comparison, stable_seed
from osler_jepa.lockbox_guard import read_non_lockbox
from osler_jepa.stacked_expert import feature_columns, fit_cell, group_importance, target_columns
from site_adaptive_external_audit import frozen_dtaci_config, frozen_network_config, frozen_point_config
from whole_body_joint_gate import _split_by_column
from osler_jepa.organ_schema import assert_disjoint_contracts, canonical_organ_contracts

MIN_MODULES = 2


def examples_path(sample: int) -> Path:
    return Path(f"four_db_jepa_examples_harmonized_anchor{'' if sample == 0 else f'_s{sample}'}_20261025.parquet")


def load_cohort(path: Path, targets: tuple[str, ...], horizons: tuple[int, ...], reader=read_non_lockbox) -> pd.DataFrame:
    """The JEPA loader's cohort (>= 2 canonical organ modules observed), only the columns the stacked expert uses.

    ``reader(path, columns=...)`` reads the file: ``read_non_lockbox`` for development data; a lockbox scorer passes
    its authorized reader."""

    names = pq.read_schema(path).names
    available = set(names)
    variables = tuple(sorted(c[len("future_"):] for c in names if c.startswith("future_") and f"{c[len('future_'):]}_t" in available
                             and not c.startswith(("future_event_time_", "future_delta_t_hr_", "future_label_quality_",
                                                   "future_label_weight_"))))
    contracts = canonical_organ_contracts(variables)          # the JEPA loader's canonical_organs contracts
    assert_disjoint_contracts(contracts)
    presence = {v: f"_observed_{v}" if f"_observed_{v}" in available else f"{v}_t" for vs in contracts.values() for v in vs}
    per_target = [f"{prefix}{t}{suffix}" for t in targets for prefix, suffix in
                  (("", "_t"), ("", "_age_hr"), ("future_", ""), ("future_label_weight_", ""), ("future_delta_t_hr_", ""),
                   ("expert__anchor_", ""))]
    columns = sorted({"subject_id", "stay_id", "database", "hospitalid", "careunit", "t", "horizon_hours",
                      *presence.values(), *per_target})
    frame = reader(path, columns=columns)
    frame = frame[frame["horizon_hours"].isin([float(h) for h in horizons])]
    count = np.zeros(len(frame), dtype=np.int16)
    for module_variables in contracts.values():
        observed = np.zeros(len(frame), dtype=bool)
        for variable in module_variables:
            values = pd.to_numeric(frame[presence[variable]], errors="coerce")
            observed |= (values.fillna(0.0).to_numpy() > 0.5) if presence[variable].startswith("_observed_") else values.notna().to_numpy()
        count += observed
    frame = frame[count >= MIN_MODULES].copy()
    for column in ("subject_id", "stay_id", "hospitalid", "careunit", "database"):
        frame[column] = frame[column].astype(str)
    # Stay ids are unique only within a database (NWICU and Zigong both number their stays from 1): qualify them so
    # onsets, trajectories and deployment stays never mix databases.
    frame["stay_id"] = frame["database"] + ":" + frame["stay_id"]
    frame["t"] = pd.to_datetime(frame["t"])
    frame["hours_since_onset"] = (frame["t"] - frame.groupby("stay_id")["t"].transform("min")).dt.total_seconds() / 3600.0
    if frame.duplicated(["stay_id", "t", "horizon_hours"]).any():
        raise ValueError(f"{path}: duplicate (stay, anchor time, horizon) rows")
    return frame.drop(columns=[c for c in set(presence.values()) if c not in per_target]).reset_index(drop=True)


def dev_cohort(sample: int, rule: dict) -> pd.DataFrame:
    """The development cohort of MIMIC training sample ``sample`` (MIMIC and the rule's test databases)."""

    return load_cohort(examples_path(sample), tuple(rule["data"]["targets"]), tuple(rule["data"]["horizons_hours"]))


def load_rule(path: Path) -> dict:
    """The rule with its inherited sections filled in from the rule it inherits."""

    rule = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads(Path(rule["inherits"]).read_text(encoding="utf-8"))
    return {**{section: base[section] for section in rule["inherited_sections"]}, **rule, "path": str(path)}


def trajectories(features: pd.DataFrame, targets: tuple[str, ...], lags: tuple[int, ...]) -> pd.DataFrame:
    """Per target and lag L: current value minus the value at the stay's latest anchor at or before t - L.

    Only earlier anchors of the same stay are read, and the anchor set does not depend on labels.
    """

    anchors = features[[f"current__{target}" for target in targets]].reset_index()
    history = anchors.sort_values("t")
    output = pd.DataFrame(index=features.index)
    for lag in lags:
        lagged = anchors.assign(t=anchors["t"] - pd.Timedelta(hours=lag)).reset_index().sort_values("t")
        earlier = pd.merge_asof(lagged, history, on="t", by="stay_id", direction="backward", suffixes=("", "__then"))
        earlier = earlier.sort_values("index")
        for target in targets:
            output[f"trend__{target}@-{lag}h"] = (earlier[f"current__{target}"] - earlier[f"current__{target}__then"]).to_numpy()
    return output


def anchor_features(frame: pd.DataFrame, targets: tuple[str, ...], horizons: tuple[int, ...],
                    lags: tuple[int, ...] = ()) -> pd.DataFrame:
    """One row per (stay, anchor time): every target's current value, measurement age, forecast change per horizon
    and trajectory per lookback."""

    first = frame.drop_duplicates(["stay_id", "t"]).set_index(["stay_id", "t"])
    features = pd.DataFrame(index=first.index)
    for target in targets:
        features[f"current__{target}"] = pd.to_numeric(first[f"{target}_t"], errors="coerce")
        features[f"age__{target}"] = pd.to_numeric(first[f"{target}_age_hr"], errors="coerce")
    for horizon in horizons:
        rows = frame[frame["horizon_hours"] == float(horizon)].set_index(["stay_id", "t"])
        for target in targets:
            change = pd.to_numeric(rows[f"expert__anchor_{target}"], errors="coerce") - pd.to_numeric(rows[f"{target}_t"], errors="coerce")
            features[f"change__{target}@{horizon}h"] = change.reindex(features.index)
    return pd.concat([features, trajectories(features, targets, lags)], axis=1) if lags else features


def labelled_rows(frame: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    rows = frame[frame["horizon_hours"] == float(horizon)]
    weight = pd.to_numeric(rows[f"future_label_weight_{target}"], errors="coerce")
    labelled = rows[f"future_{target}"].notna() & rows[f"{target}_t"].notna() & (weight.isna() | (weight > 0))
    rows = rows[labelled]
    delay = pd.to_numeric(rows[f"future_delta_t_hr_{target}"], errors="coerce").to_numpy()
    return pd.DataFrame({
        "subject_id": rows["subject_id"].to_numpy(), "stay_id": rows["stay_id"].to_numpy(),
        "hospitalid": rows["hospitalid"].to_numpy(), "careunit": rows["careunit"].to_numpy(),
        "database": rows["database"].to_numpy(), "t": rows["t"].to_numpy(),
        "hours_since_onset": rows["hours_since_onset"].to_numpy(),
        "label_available_hr": horizon + np.abs(delay - horizon),
        "current": pd.to_numeric(rows[f"{target}_t"], errors="coerce").to_numpy(),
        "future": pd.to_numeric(rows[f"future_{target}"], errors="coerce").to_numpy(),
        "anchor": pd.to_numeric(rows[f"expert__anchor_{target}"], errors="coerce").to_numpy(),
    })


def score_database(calibration: pd.DataFrame, test: pd.DataFrame, name: str, cell: str, configs, rule: dict) -> dict:
    axes, candidate = GATE_AXES[name], rule["candidate"]
    systems = ("anchor", *rule["systems"])
    outputs = {system: deploy(calibration, test, system, cell, *configs) for system in systems}
    index = np.flatnonzero(outputs["anchor"][2])
    future = test["future"].to_numpy()
    loss = {system: np.abs(outputs[system][0] - future)[index] for system in systems}
    loss["persistence"] = np.abs(test["current"].to_numpy() - future)[index]
    pairs = [(candidate, other) for other in rule["required_comparators"]] + [(a, b) for a, b, _ in rule["secondary_comparisons"]]
    comparisons = {}
    for system, other in pairs:
        result = paired_comparison(test, index, loss[system], loss[other], seed=stable_seed(cell, name, system, other))
        comparisons[f"{system}_vs_{other}"] = {
            "pass": all(result[axis]["pass"] for axis in axes["point"]),
            "ci": {axis: [result[axis]["bootstrap_low"], result[axis]["bootstrap_high"]] for axis in axes["point"]}}
    width = outputs[candidate][1][index]
    interval = interval_report(test, index, loss[candidate] <= width, width, axes["interval"])
    required = [f"{candidate}_vs_{other}" for other in rule["required_comparators"]]
    return {
        "scored_rows": int(len(index)),
        "corrected_mae": {system: float(values.mean()) for system, values in loss.items()},
        "comparisons": comparisons,
        "interval": {"pass": bool(interval["pass"]), "coverage": [round(interval[a], 4) for a in interval["gate_axes"]]},
        "pass": bool(all(comparisons[name]["pass"] for name in required) and interval["pass"]),
    }


def run_sample(frame: pd.DataFrame, sample: int, rule: dict, configs) -> dict:
    """``frame``: the cohort (``load_cohort``) holding the training database and the rule's test databases."""

    targets, horizons = tuple(rule["data"]["targets"]), tuple(rule["data"]["horizons_hours"])
    lags = tuple(rule["trajectory"]["lags_hours"])
    features = anchor_features(frame, targets, horizons, lags)
    mimic = frame[frame["database"] == rule["data"]["training_database"]].reset_index(drop=True)
    _, calibration_index = _split_by_column(mimic, "subject_id", stable_seed("stacked_expert_split", sample), 0.20)
    calibration_subjects = set(mimic["subject_id"].iloc[calibration_index])
    groups = {target: target_columns(target, horizons, lags) for target in targets}
    cells = {}
    for target in targets:
        for horizon in horizons:
            cell = f"{target}@{horizon}h"
            rows = labelled_rows(frame, target, horizon)
            x = features.reindex(pd.MultiIndex.from_arrays([rows["stay_id"], rows["t"]])).reset_index(drop=True)
            is_mimic = (rows["database"] == rule["data"]["training_database"]).to_numpy()
            is_calibration = is_mimic & rows["subject_id"].isin(calibration_subjects).to_numpy()
            fit = is_mimic & ~is_calibration
            anchor, future = rows["anchor"].to_numpy(), rows["future"].to_numpy()
            if not np.isfinite(anchor).all():
                raise ValueError(f"{cell}: labelled rows without a frozen expert forecast")
            models = {name: fit_cell(x[fit], anchor[fit], future[fit],
                                     feature_columns(target, targets, horizons, lags if spec["trajectory"] else (),
                                                     exchange=spec["exchange"]))
                      for name, spec in rule["systems"].items()}
            for name, model in models.items():
                rows[name] = model.predict(x, anchor)
            calibration = rows[is_calibration].reset_index(drop=True)
            entry = {"fit_rows": int(fit.sum()), "calibration_rows": int(is_calibration.sum()),
                     "importance": group_importance(models[rule["candidate"]], x[is_calibration].reset_index(drop=True),
                                                    anchor[is_calibration], future[is_calibration], groups,
                                                    seed=stable_seed(cell, sample, "importance")),
                     "databases": {}}
            for database in rule["data"]["test_databases"]:
                test = rows[(rows["database"] == database).to_numpy()].reset_index(drop=True)
                if len(test) < MIN_ROWS or test["subject_id"].nunique() < MIN_SUBJECTS:
                    continue
                entry["databases"][DATABASE_NAMES[database]] = score_database(calibration, test, DATABASE_NAMES[database], cell,
                                                                              configs, rule)
            cells[cell] = entry
            print(json.dumps({"sample": sample, "cell": cell,
                              **{db: [r["pass"]] + [round(1 - r["corrected_mae"][s] / r["corrected_mae"]["anchor"], 4)
                                                    for s in rule["systems"]]
                                 for db, r in entry["databases"].items()}}), flush=True)
    return {"schema": "stacked_expert_dev.v2", "rule": rule["path"], "candidate": rule["candidate"], "sample": sample,
            "examples": str(examples_path(sample)),
            "cells": cells, "development_evidence_only": True, "causal_claim_allowed": False}


def summarize(paths: list[Path]) -> dict:
    return summary_of([json.loads(path.read_text(encoding="utf-8")) for path in paths], [str(p) for p in paths])


def summary_of(samples: list[dict], sources: list[str]) -> dict:
    """Robust cells over the samples' results (a cell missing from a sample counts as not passing there)."""

    recipes = json.loads(Path("deployment_recipes_20261024.json").read_text(encoding="utf-8"))["recipes"]
    keys = sorted({f"{db}:{cell}" for s in samples for cell, e in s["cells"].items() for db in e["databases"]})
    cells = {}
    for key in keys:
        database, cell = key.split(":", 1)
        results = [s["cells"].get(cell, {}).get("databases", {}).get(database) for s in samples]
        passes = [bool(r and r["pass"]) for r in results]
        gains = [1 - r["corrected_mae"][s["candidate"]] / r["corrected_mae"]["anchor"] for s, r in zip(samples, results) if r]
        cells[key] = {"expert_recipe": recipes.get(cell, "persistence") != "persistence",
                      "sample_pass_share": float(np.mean(passes)), "robust": bool(np.mean(passes) >= ROBUST_SHARE),
                      "median_gain_vs_anchor": float(np.median(gains)) if gains else None}
    return {"schema": "stacked_expert_dev_summary.v2", "rule": samples[0]["rule"], "candidate": samples[0]["candidate"],
            "samples": sources,
            "robust_expert_cells": sorted(k for k, v in cells.items() if v["robust"] and v["expert_recipe"]),
            "robust_persistence_anchor_cells": sorted(k for k, v in cells.items() if v["robust"] and not v["expert_recipe"]),
            "cells": cells, "development_evidence_only": True, "causal_claim_allowed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int)
    group.add_argument("--summarize", nargs="+", type=Path)
    parser.add_argument("--rule", type=Path, default=Path("stacked_expert_trajectory_dev_rule_20261031.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.summarize)
    else:
        rule = load_rule(args.rule)
        jepa_rule = json.loads(Path("jepa_gate_rule_20261022.json").read_text(encoding="utf-8"))
        _, point_config = frozen_point_config(Path(jepa_rule["point_dev"]))
        _, dtaci_config = frozen_dtaci_config(Path(jepa_rule["dtaci_dev"]), Path(jepa_rule["dtaci_prior"]))
        configs = (point_config, dtaci_config, frozen_network_config(Path(jepa_rule["network_dev"]), dtaci_config))
        result = run_sample(dev_cohort(args.sample, rule), args.sample, rule, configs)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
