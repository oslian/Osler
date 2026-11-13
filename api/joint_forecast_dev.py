"""Development test of the joint dependence layer (``--rule``; schema joint_forecast_dev_rule.v3).

Per MIMIC training sample K: the stacked-expert cohort, labels and patient
split (``stacked_expert_dev``).  Marginals: the rule's point system (the
frozen expert anchor, or a residual system of ``osler_jepa.stacked_expert``
fitted on MIMIC fit patients) + empirical quantiles of its signed errors on
MIMIC conformal patients.  Copulas (``osler_jepa.joint_copula``): gaussian
(correlation of the errors on MIMIC selection patients), t (the same
correlation, degrees of freedom by composite pseudo-likelihood on the same
patients) and independent (the same base draws with the identity
correlation).  On each test database and block (cross_target: one vector per
anchor and horizon over the targets; across_horizons: one per anchor and
target over the horizons) the variogram score (primary) and energy score of
the rule's copula pairs are compared per patient (patient-cluster bootstrap);
same-horizon joint events are scored by O/E and Brier.  ``--summarize`` combines the five samples.
Static forecasts (no deployment form).  Development evidence only; research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jepa_gate import DATABASE_NAMES, ROBUST_SHARE
from mimic_to_eicu_49_cell_external_audit import paired_comparison, stable_seed
from osler_jepa.joint_copula import (energy_score, latent_correlation, latent_draws, marginal_members, t_degrees_of_freedom,
                                     t_latent, variogram_score)
from osler_jepa.patient_split import selection_patients
from osler_jepa.stacked_expert import feature_columns, fit_cell
from stacked_expert_dev import anchor_features, dev_cohort, labelled_rows, load_rule
from whole_body_joint_gate import _split_by_column

MIN_PAIRS, MIN_JOINT_EVENTS = 200, 30
T_DF_GRID = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0, 50.0, 100.0)
EVENTS = {
    "map < 65": ("map", lambda future, current: future < 65.0),
    "heart_rate > 120": ("heart_rate", lambda future, current: future > 120.0),
    "potassium >= 6.0": ("potassium", lambda future, current: future >= 6.0),
    "bicarbonate < 18": ("bicarbonate", lambda future, current: future < 18.0),
    "creatinine rise >= 0.3": ("creatinine", lambda future, current: future - current >= 0.3),
    "glucose > 180": ("glucose", lambda future, current: future > 180.0),
}
EVENT_PAIRS = (("map < 65", "heart_rate > 120"), ("potassium >= 6.0", "bicarbonate < 18"),
               ("creatinine rise >= 0.3", "potassium >= 6.0"), ("glucose > 180", "bicarbonate < 18"))


def cell_rows(frame: pd.DataFrame, sample: int, rule: dict) -> dict[str, pd.DataFrame]:
    """Per cell: its labelled rows outside the MIMIC fit patients, with the rule's point forecast and each row's role
    (``selection`` / ``conformal`` MIMIC calibration patients, or the test database).  ``frame``: the cohort
    (``stacked_expert_dev.load_cohort``)."""

    targets, horizons = tuple(rule["data"]["targets"]), tuple(rule["data"]["horizons_hours"])
    system = rule["marginal"]["system"]
    if system != "anchor":
        spec, lags = rule["systems"][system], tuple(rule["trajectory"]["lags_hours"])
        features = anchor_features(frame, targets, horizons, lags if spec["trajectory"] else ())
    training = rule["data"]["training_database"]
    mimic = frame[frame["database"] == training].reset_index(drop=True)
    _, calibration_index = _split_by_column(mimic, "subject_id", stable_seed("stacked_expert_split", sample), 0.20)
    calibration_subjects = set(mimic["subject_id"].iloc[calibration_index])
    output = {}
    for target in targets:
        for horizon in horizons:
            cell = f"{target}@{horizon}h"
            rows = labelled_rows(frame, target, horizon)
            is_mimic = (rows["database"] == training).to_numpy()
            fit = is_mimic & ~rows["subject_id"].isin(calibration_subjects).to_numpy()
            point = rows["anchor"].to_numpy()
            if system != "anchor":
                x = features.reindex(pd.MultiIndex.from_arrays([rows["stay_id"], rows["t"]])).reset_index(drop=True)
                columns = feature_columns(target, targets, horizons, lags if spec["trajectory"] else (), exchange=spec["exchange"])
                point = fit_cell(x[fit], point[fit], rows["future"].to_numpy()[fit], columns).predict(x, point)
            rows = rows.assign(point=point)[~fit].reset_index(drop=True)                 # MIMIC fit patients only fit
            choosing = selection_patients(rows["subject_id"].to_numpy())
            rows["role"] = np.where(rows["database"] != training, rows["database"], np.where(choosing, "selection", "conformal"))
            output[cell] = rows
    return output


def forecasts(sample: int, rule: dict) -> tuple[list[str], pd.DataFrame]:
    """Cells and one wide frame per (stay, anchor time): role, subject, hospital and point/future/current per cell."""

    per_cell = cell_rows(dev_cohort(sample, rule), sample, rule)
    pieces, meta = [], []
    for cell, rows in per_cell.items():
        index = pd.MultiIndex.from_arrays([rows["stay_id"], rows["t"]])
        pieces.append(pd.DataFrame({f"point:{cell}": rows["point"].to_numpy(), f"future:{cell}": rows["future"].to_numpy(),
                                    f"current:{cell}": rows["current"].to_numpy()}, index=index))
        meta.append(rows[["subject_id", "hospitalid", "role"]].set_index(index))
    wide = pd.concat(pieces, axis=1).join(pd.concat(meta).groupby(level=[0, 1]).first())
    return list(per_cell), wide.reset_index()


def errors_of(wide: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    return pd.DataFrame({cell: wide[f"future:{cell}"] - wide[f"point:{cell}"] for cell in cells})


def block_vectors(cells: list[str], targets, horizons) -> dict[str, list[list[str]]]:
    return {"cross_target": [[f"{t}@{h}h" for t in targets] for h in horizons],
            "across_horizons": [[f"{t}@{h}h" for h in horizons] for t in targets]}


def score_block(test: pd.DataFrame, group: list[str], cells: list[str], correlation, df: float, sorted_errors, scale,
                members: int, seed: int) -> dict | None:
    """Independent, gaussian and t ensembles of one vector of cells on the rows labelled for >= 2 of them."""

    labelled = np.column_stack([test[f"future:{c}"].notna().to_numpy() for c in group])
    rows = np.flatnonzero(labelled.sum(axis=1) >= 2)
    if not len(rows):
        return None
    labelled = labelled[rows]
    position = [cells.index(c) for c in group]
    point = np.column_stack([test[f"point:{c}"].to_numpy()[rows] for c in group])
    observed = np.column_stack([test[f"future:{c}"].to_numpy()[rows] for c in group])
    current = np.column_stack([test[f"current:{c}"].to_numpy()[rows] for c in group])
    independent, gaussian = latent_draws(len(rows), members, correlation[np.ix_(position, position)], seed=seed)
    latents = {"independent": independent, "gaussian": gaussian, "t": t_latent(gaussian, df, seed=seed + 1)}
    errors = [sorted_errors[p] for p in position]
    ensembles = {name: marginal_members(np.nan_to_num(point), errors, latent) for name, latent in latents.items()}
    units = np.array([scale[p] for p in position])
    standardized_observed = np.nan_to_num(observed) / units
    scores = {name: {"variogram": variogram_score(values / units, standardized_observed, labelled),
                     "energy": energy_score(values / units, standardized_observed, labelled)}
              for name, values in ensembles.items()}
    return {"rows": rows, "labelled": labelled, "observed": observed, "current": current, "ensembles": ensembles,
            "scores": scores}


def joint_events(block: dict, group: list[str]) -> dict:
    output = {}
    targets = [c.split("@")[0] for c in group]
    for first, second in EVENT_PAIRS:
        (ta, fa), (tb, fb) = EVENTS[first], EVENTS[second]
        if ta not in targets or tb not in targets:
            continue
        a, b = targets.index(ta), targets.index(tb)
        both = block["labelled"][:, a] & block["labelled"][:, b]
        happened = fa(block["observed"][both, a], block["current"][both, a]) & fb(block["observed"][both, b], block["current"][both, b])
        if happened.sum() < MIN_JOINT_EVENTS:
            output[f"{first} & {second}"] = {"labelled_rows": int(both.sum()), "events": int(happened.sum()), "scored": False}
            continue
        entry = {"labelled_rows": int(both.sum()), "events": int(happened.sum()), "observed_rate": float(happened.mean()), "scored": True}
        marginal = block["ensembles"]["independent"][both]                                 # both systems share the marginals
        for label, column, event in ((first, a, fa), (second, b, fb)):
            observed = event(block["observed"][both, column], block["current"][both, column])
            predicted = np.mean(event(marginal[:, :, column], block["current"][both, column][:, None]), axis=1)
            entry[f"marginal {label}"] = {"observed_rate": float(observed.mean()), "o_e": float(observed.mean() / max(predicted.mean(), 1e-12))}
        for name, values in block["ensembles"].items():
            members = values[both]
            probability = np.mean(fa(members[:, :, a], block["current"][both, a][:, None])
                                  & fb(members[:, :, b], block["current"][both, b][:, None]), axis=1)
            entry[name] = {"mean_probability": float(probability.mean()), "o_e": float(happened.mean() / max(probability.mean(), 1e-12)),
                           "brier": float(np.mean((probability - happened) ** 2))}
        output[f"{first} & {second}"] = entry
    return output


def run_sample(sample: int, rule: dict) -> dict:
    targets, horizons = tuple(rule["data"]["targets"]), tuple(rule["data"]["horizons_hours"])
    members = int(rule["marginal"]["members"])
    cells, wide = forecasts(sample, rule)
    errors = errors_of(wide, cells)
    selection = errors[(wide["role"] == "selection").to_numpy()]
    correlation = latent_correlation(selection, MIN_PAIRS)
    df, profile = t_degrees_of_freedom(selection, correlation, MIN_PAIRS, T_DF_GRID)
    comparisons = [tuple(pair) for pair in rule["comparisons"]]
    conformal = errors[(wide["role"] == "conformal").to_numpy()]
    sorted_errors = [np.sort(conformal[c].dropna().to_numpy()) for c in cells]
    scale = [float(np.subtract(*np.quantile(e, [0.75, 0.25]))) for e in sorted_errors]
    result = {"schema": "joint_forecast_dev.v3", "rule": rule["path"], "marginal": rule["marginal"]["system"], "sample": sample, "cells": cells,
              "latent_correlation": correlation.round(4).tolist(), "t_degrees_of_freedom": df,
              "t_profile": {f"{k:g}": v for k, v in profile.items()}, "databases": {}}
    for database in rule["data"]["test_databases"]:
        test = wide[(wide["role"] == database).to_numpy()].reset_index(drop=True)
        name = DATABASE_NAMES[database]
        report = {}
        for block_name, groups in block_vectors(cells, targets, horizons).items():
            losses = {system: {score: [] for score in ("variogram", "energy")} for system in ("independent", "gaussian", "t")}
            subjects, hospitals, events = [], [], {}
            for group in groups:
                block = score_block(test, group, cells, correlation, df, sorted_errors, scale, members,
                                    seed=stable_seed("joint", sample, database, block_name, *group))
                if block is None:
                    continue
                for system in losses:
                    for score in losses[system]:
                        losses[system][score].append(block["scores"][system][score])
                subjects.append(test["subject_id"].to_numpy()[block["rows"]])
                hospitals.append(test["hospitalid"].to_numpy()[block["rows"]])
                if block_name == "cross_target":
                    events[group[0].split("@")[1]] = joint_events(block, group)
            frame = pd.DataFrame({"subject_id": np.concatenate(subjects), "hospitalid": np.concatenate(hospitals)})
            index = np.arange(len(frame))
            entry = {"vectors": int(len(frame)), "patients": int(frame["subject_id"].nunique())}
            for score in ("variogram", "energy"):
                totals = {system: np.concatenate(losses[system][score]) for system in losses}
                entry[score] = {"mean": {system: float(values.mean()) for system, values in totals.items()}}
                for candidate, comparator in comparisons:
                    patient = paired_comparison(frame, index, totals[candidate], totals[comparator],
                                                seed=stable_seed("joint", sample, database, block_name, score))["patient"]
                    entry[score][f"{candidate}_vs_{comparator}"] = {
                        "relative": float(totals[candidate].mean() / totals[comparator].mean() - 1.0),
                        "ci": [patient["bootstrap_low"], patient["bootstrap_high"]],
                        "better": bool(patient["bootstrap_high"] is not None and patient["bootstrap_high"] < 0),
                        "worse": bool(patient["bootstrap_low"] is not None and patient["bootstrap_low"] > 0)}
            if events:
                entry["joint_events"] = events
            report[block_name] = entry
            print(json.dumps({"sample": sample, "database": name, "block": block_name, "vectors": entry["vectors"], "df": df,
                              **{f"{s}:{a}_vs_{b}": [round(entry[s][f"{a}_vs_{b}"]["relative"], 4), entry[s][f"{a}_vs_{b}"]["better"]]
                                 for s in ("variogram", "energy") for a, b in comparisons}}), flush=True)
        result["databases"][name] = report
    return {**result, "development_evidence_only": True, "causal_claim_allowed": False}


def summarize(paths: list[Path], rule: dict) -> dict:
    samples = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    comparisons = [f"{a}_vs_{b}" for a, b in rule["comparisons"]]
    blocks, events = {}, {}
    for database in samples[0]["databases"]:
        for block in samples[0]["databases"][database]:
            entries = [s["databases"][database][block] for s in samples]
            blocks[f"{database}:{block}"] = {
                score: {comparison: {"better_share": float(np.mean([e[score][comparison]["better"] for e in entries])),
                                     "worse_share": float(np.mean([e[score][comparison]["worse"] for e in entries])),
                                     "median_relative": float(np.median([e[score][comparison]["relative"] for e in entries])),
                                     "robust_better": bool(np.mean([e[score][comparison]["better"] for e in entries]) >= ROBUST_SHARE),
                                     "robust_worse": bool(np.mean([e[score][comparison]["worse"] for e in entries]) >= ROBUST_SHARE)}
                        for comparison in comparisons}
                for score in ("variogram", "energy")}
        for horizon in samples[0]["databases"][database]["cross_target"].get("joint_events", {}):
            for pair in samples[0]["databases"][database]["cross_target"]["joint_events"][horizon]:
                scored = [s["databases"][database]["cross_target"]["joint_events"][horizon][pair] for s in samples]
                scored = [e for e in scored if e["scored"]]
                if not scored:
                    continue
                closer = [abs(np.log(e["t"]["o_e"])) < abs(np.log(e["gaussian"]["o_e"])) for e in scored]
                events[f"{database}:{horizon}:{pair}"] = {
                    "scored_samples": len(scored), "t_closer_share": float(np.mean(closer)),
                    "t_robustly_closer": bool(np.mean(closer) >= ROBUST_SHARE),
                    **{f"{system}_o_e": [round(min(e[system]["o_e"] for e in scored), 3), round(max(e[system]["o_e"] for e in scored), 3)]
                       for system in ("independent", "gaussian", "t")}}
    decided = [entry for entry in events.values() if entry["scored_samples"] >= 4]
    events_pass = bool(decided) and np.mean([entry["t_robustly_closer"] for entry in decided]) >= 0.5
    variogram_pass = not any(entry["variogram"]["t_vs_gaussian"]["robust_worse"] for entry in blocks.values())
    return {"schema": "joint_forecast_dev_summary.v2", "rule": samples[0]["rule"], "samples": [str(p) for p in paths],
            "t_degrees_of_freedom": [s["t_degrees_of_freedom"] for s in samples], "blocks": blocks, "joint_events": events,
            "decision": {"events_decided": len(decided), "events_t_robustly_closer": int(sum(e["t_robustly_closer"] for e in decided)),
                         "events_pass": bool(events_pass), "variogram_not_robustly_worse": bool(variogram_pass),
                         "t_replaces_gaussian": bool(events_pass and variogram_pass)},
            "development_evidence_only": True, "causal_claim_allowed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int)
    group.add_argument("--summarize", nargs="+", type=Path)
    parser.add_argument("--rule", type=Path, default=Path("t_copula_joint_dev_rule_20261031.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rule = load_rule(args.rule)
    result = summarize(args.summarize, rule) if args.summarize else run_sample(args.sample, rule)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
