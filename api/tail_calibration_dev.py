"""Development test of a state-conditional marginal under the dependence layer (``--rule``; tail_calibration_dev_rule.v1).

Per MIMIC training sample K: the own_trajectory static forecasts of ``joint_forecast_dev.forecasts``.  Two marginals per
cell: ``unconditional`` (point + empirical quantiles of the conformal errors) and ``mondrian`` (point + s(x) * empirical
quantiles of the normalized conformal errors of the row's current-value category; s = v2 quintile scale floored at the
recording resolution).  Single threshold events are scored per cell on every labelled test row (O/E); the multivariate
blocks of ``joint_forecast_dev`` are scored for both marginals under the Gaussian copula and independence (energy and
variogram score, patient-cluster bootstrap), with same-horizon joint events.  ``--summarize`` combines the five samples.
Development evidence only; research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from joint_forecast_dev import EVENT_PAIRS, EVENTS, MIN_JOINT_EVENTS, MIN_PAIRS, block_vectors, errors_of, forecasts
from jepa_gate import DATABASE_NAMES, ROBUST_SHARE
from mimic_to_eicu_49_cell_external_audit import paired_comparison, stable_seed
from osler_jepa.interval_scale import MIN_PATIENTS, fit_interval_scale
from osler_jepa.joint_copula import (conditional_members, energy_score, latent_correlation, latent_draws, marginal_members,
                                     variogram_score)
from osler_jepa.predictive_distribution import value_bins
from stacked_expert_dev import load_rule
from threshold_event_form import resolution

MARGINALS, COPULAS = ("unconditional", "mondrian"), ("independent", "gaussian")
BAND = (0.8, 1.25)


def calibrate(wide: pd.DataFrame, cell: str) -> dict:
    """Both marginals' ingredients for one cell from its MIMIC selection and conformal patients."""

    labelled = wide[f"future:{cell}"].notna().to_numpy()
    selection, conformal = labelled & (wide["role"] == "selection").to_numpy(), labelled & (wide["role"] == "conformal").to_numpy()
    error = (wide[f"future:{cell}"] - wide[f"point:{cell}"]).to_numpy()
    current = wide[f"current:{cell}"].to_numpy()
    subjects = wide["subject_id"].astype(str).to_numpy()
    floor = resolution(current[selection | conformal])
    fitted = fit_interval_scale("quintile", current[selection], np.zeros(selection.sum()), np.abs(error[selection]), subjects[selection])
    scale = lambda values: np.maximum(fitted(values, np.zeros(len(values))), floor)
    edges = value_bins(current[selection | conformal])
    groups = np.digitize(current[conformal], edges)
    scores = error[conformal] / scale(current[conformal])
    pooled = np.sort(scores)
    by_group = {}
    for group in range(len(edges) + 1):
        mask = groups == group
        enough = len(np.unique(subjects[conformal][mask])) >= MIN_PATIENTS
        by_group[group] = np.sort(scores[mask]) if enough else pooled
    errors = np.sort(error[conformal])
    return {"errors": errors, "scale": scale, "edges": edges, "scores": by_group,
            "unit": float(np.subtract(*np.quantile(errors, [0.75, 0.25])))}


def members(marginal: str, calibrations: list[dict], point: np.ndarray, current: np.ndarray, latent: np.ndarray) -> np.ndarray:
    point = np.nan_to_num(point)
    if marginal == "unconditional":
        return marginal_members(point, [c["errors"] for c in calibrations], latent)
    current = np.nan_to_num(current)
    scale = np.column_stack([c["scale"](current[:, k]) for k, c in enumerate(calibrations)])
    groups = np.column_stack([np.digitize(current[:, k], c["edges"]) for k, c in enumerate(calibrations)])
    return conditional_members(point, scale, groups, [c["scores"] for c in calibrations], latent)


def member_latent(rows: int, count: int) -> np.ndarray:
    return np.broadcast_to(norm.ppf((np.arange(1, count + 1) - 0.5) / count)[None, :, None], (rows, count, 1))


def single_events(test: pd.DataFrame, cells: list[str], calibrations: dict, count: int) -> dict:
    """Per cell whose target has an event: O/E of both marginals on its labelled rows, plus central-90% tail exceedance."""

    output = {}
    for cell in cells:
        rows = test[f"future:{cell}"].notna().to_numpy()
        if not rows.any():
            continue
        target = cell.split("@")[0]
        point, observed = test[f"point:{cell}"].to_numpy()[rows, None], test[f"future:{cell}"].to_numpy()[rows]
        current = test[f"current:{cell}"].to_numpy()[rows, None]
        entry = {"rows": int(rows.sum())}
        for marginal in MARGINALS:
            values = members(marginal, [calibrations[cell]], point, current, member_latent(int(rows.sum()), count))[:, :, 0]
            low, high = np.quantile(values, [0.05, 0.95], axis=1, method="linear")
            entry[f"{marginal}_tails"] = [float(np.mean(observed < low)), float(np.mean(observed > high))]
            for label, (event_target, event) in EVENTS.items():
                if event_target != target:
                    continue
                happened = event(observed, current[:, 0])
                if happened.sum() < MIN_JOINT_EVENTS:
                    entry[label] = {"events": int(happened.sum()), "scored": False}
                    continue
                probability = np.mean(event(values, current), axis=1)
                entry.setdefault(label, {"events": int(happened.sum()), "scored": True, "observed_rate": float(happened.mean())})
                entry[label][marginal] = {"o_e": float(happened.mean() / max(probability.mean(), 1e-12)),
                                          "brier": float(np.mean((probability - happened) ** 2))}
        output[cell] = entry
    return output


def joint_events(observed, current, labelled, ensembles, group) -> dict:
    output = {}
    targets = [c.split("@")[0] for c in group]
    for first, second in EVENT_PAIRS:
        (ta, fa), (tb, fb) = EVENTS[first], EVENTS[second]
        if ta not in targets or tb not in targets:
            continue
        a, b = targets.index(ta), targets.index(tb)
        both = labelled[:, a] & labelled[:, b]
        happened = fa(observed[both, a], current[both, a]) & fb(observed[both, b], current[both, b])
        entry = {"labelled_rows": int(both.sum()), "events": int(happened.sum()), "scored": bool(happened.sum() >= MIN_JOINT_EVENTS)}
        if entry["scored"]:
            for name, values in ensembles.items():
                probability = np.mean(fa(values[both][:, :, a], current[both, a][:, None])
                                      & fb(values[both][:, :, b], current[both, b][:, None]), axis=1)
                entry[name] = {"o_e": float(happened.mean() / max(probability.mean(), 1e-12)),
                               "brier": float(np.mean((probability - happened) ** 2))}
        output[f"{first} & {second}"] = entry
    return output


def compare(frame: pd.DataFrame, candidate: np.ndarray, comparator: np.ndarray, seed: int) -> dict:
    patient = paired_comparison(frame, np.arange(len(frame)), candidate, comparator, seed=seed)["patient"]
    high, low = patient["bootstrap_high"], patient["bootstrap_low"]
    return {"relative": float(candidate.mean() / comparator.mean() - 1.0), "ci": [low, high],
            "better": bool(high is not None and high < 0), "worse": bool(low is not None and low > 0)}


COMPARISONS = (("mondrian:gaussian", "unconditional:gaussian"), ("mondrian:gaussian", "mondrian:independent"),
               ("unconditional:gaussian", "unconditional:independent"))


def run_sample(sample: int, rule: dict) -> dict:
    targets, horizons = tuple(rule["data"]["targets"]), tuple(rule["data"]["horizons_hours"])
    count = int(rule["marginal"]["members"])
    cells, wide = forecasts(sample, rule)
    correlation = latent_correlation(errors_of(wide, cells)[(wide["role"] == "selection").to_numpy()], MIN_PAIRS)
    calibrations = {cell: calibrate(wide, cell) for cell in cells}
    result = {"schema": "tail_calibration_dev.v1", "rule": rule["path"], "sample": sample, "databases": {}}
    for database in rule["data"]["test_databases"]:
        test = wide[(wide["role"] == database).to_numpy()].reset_index(drop=True)
        name = DATABASE_NAMES[database]
        report = {"single_events": single_events(test, cells, calibrations, count)}
        for block_name, groups in block_vectors(cells, targets, horizons).items():
            losses = {f"{m}:{c}": {"variogram": [], "energy": []} for m in MARGINALS for c in COPULAS}
            subjects, hospitals, events = [], [], {}
            for group in groups:
                labelled = np.column_stack([test[f"future:{c}"].notna().to_numpy() for c in group])
                rows = np.flatnonzero(labelled.sum(axis=1) >= 2)
                if not len(rows):
                    continue
                labelled = labelled[rows]
                point, observed, current = (np.column_stack([test[f"{kind}:{c}"].to_numpy()[rows] for c in group])
                                            for kind in ("point", "future", "current"))
                position = [cells.index(c) for c in group]
                latents = dict(zip(COPULAS, latent_draws(len(rows), count, correlation[np.ix_(position, position)],
                                                         seed=stable_seed("joint", sample, database, block_name, *group))))
                units = np.array([calibrations[c]["unit"] for c in group])
                ensembles = {f"{m}:{c}": members(m, [calibrations[g] for g in group], point, current, latents[c])
                             for m in MARGINALS for c in COPULAS}
                for system, values in ensembles.items():
                    losses[system]["variogram"].append(variogram_score(values / units, np.nan_to_num(observed) / units, labelled))
                    losses[system]["energy"].append(energy_score(values / units, np.nan_to_num(observed) / units, labelled))
                subjects.append(test["subject_id"].to_numpy()[rows])
                hospitals.append(test["hospitalid"].to_numpy()[rows])
                if block_name == "cross_target":
                    events[group[0].split("@")[1]] = joint_events(observed, current, labelled, ensembles, group)
            frame = pd.DataFrame({"subject_id": np.concatenate(subjects), "hospitalid": np.concatenate(hospitals)})
            entry = {"vectors": int(len(frame)), "patients": int(frame["subject_id"].nunique())}
            for score in ("variogram", "energy"):
                totals = {system: np.concatenate(values[score]) for system, values in losses.items()}
                entry[score] = {f"{a}_vs_{b}": compare(frame, totals[a], totals[b], stable_seed("tail", sample, database, block_name, score))
                                for a, b in COMPARISONS}
            if events:
                entry["joint_events"] = events
            report[block_name] = entry
            print(json.dumps({"sample": sample, "database": name, "block": block_name,
                              **{f"{s}:{k}": [round(v["relative"], 4), v["better"], v["worse"]]
                                 for s in ("variogram", "energy") for k, v in entry[s].items()}}), flush=True)
        result["databases"][name] = report
    return {**result, "development_evidence_only": True, "causal_claim_allowed": False}


def summarize(paths: list[Path]) -> dict:
    samples = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    blocks, single, joint = {}, {}, {}
    share = lambda flags: float(np.mean(flags))
    for database in samples[0]["databases"]:
        for block in ("cross_target", "across_horizons"):
            entries = [s["databases"][database][block] for s in samples]
            blocks[f"{database}:{block}"] = {
                score: {comparison: {"better_share": share([e[score][comparison]["better"] for e in entries]),
                                     "worse_share": share([e[score][comparison]["worse"] for e in entries]),
                                     "median_relative": float(np.median([e[score][comparison]["relative"] for e in entries]))}
                        for comparison in entries[0][score]}
                for score in ("variogram", "energy")}
        for cell in samples[0]["databases"][database]["single_events"]:
            per_sample = [s["databases"][database]["single_events"].get(cell, {}) for s in samples]
            tails = {m: np.round(np.mean([p[f"{m}_tails"] for p in per_sample if p], axis=0), 4).tolist() for m in MARGINALS}
            for label in EVENTS:
                scored = [p[label] for p in per_sample if p.get(label, {}).get("scored")]
                if not scored:
                    continue
                closer = [abs(np.log(e["mondrian"]["o_e"])) < abs(np.log(e["unconditional"]["o_e"])) for e in scored]
                single[f"{database}:{cell}:{label}"] = {
                    "scored_samples": len(scored), "mondrian_closer_share": share(closer),
                    "mondrian_robustly_closer": share(closer) >= ROBUST_SHARE,
                    **{f"{m}_o_e": [round(min(e[m]["o_e"] for e in scored), 3), round(max(e[m]["o_e"] for e in scored), 3)] for m in MARGINALS},
                    **{f"{m}_in_band": share([BAND[0] <= e[m]["o_e"] <= BAND[1] for e in scored]) >= ROBUST_SHARE for m in MARGINALS},
                    "tails": tails}
        for horizon in samples[0]["databases"][database]["cross_target"].get("joint_events", {}):
            for pair in samples[0]["databases"][database]["cross_target"]["joint_events"][horizon]:
                scored = [s["databases"][database]["cross_target"]["joint_events"][horizon][pair] for s in samples]
                scored = [e for e in scored if e["scored"]]
                if scored:
                    joint[f"{database}:{horizon}:{pair}"] = {
                        "scored_samples": len(scored),
                        **{f"{system}_o_e": [round(min(e[system]["o_e"] for e in scored), 3), round(max(e[system]["o_e"] for e in scored), 3)]
                           for system in scored[0] if ":" in system}}
    decided = [entry for entry in single.values() if entry["scored_samples"] >= 4]
    events_pass = bool(decided) and share([e["mondrian_robustly_closer"] for e in decided]) >= 0.5
    energy_pass = not any(entry["energy"]["mondrian:gaussian_vs_unconditional:gaussian"]["worse_share"] >= ROBUST_SHARE
                          for entry in blocks.values())
    return {"schema": "tail_calibration_dev_summary.v1", "rule": samples[0]["rule"], "samples": [str(p) for p in paths],
            "blocks": blocks, "single_events": single, "joint_events": joint,
            "decision": {"events_decided": len(decided), "mondrian_robustly_closer": int(sum(e["mondrian_robustly_closer"] for e in decided)),
                         "in_band": {m: int(sum(e[f"{m}_in_band"] for e in decided)) for m in MARGINALS},
                         "events_pass": bool(events_pass), "energy_not_robustly_worse": bool(energy_pass),
                         "mondrian_replaces_unconditional": bool(events_pass and energy_pass)},
            "development_evidence_only": True, "causal_claim_allowed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int)
    group.add_argument("--summarize", nargs="+", type=Path)
    parser.add_argument("--rule", type=Path, default=Path("tail_calibration_dev_rule_20261031.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.summarize) if args.summarize else run_sample(args.sample, load_rule(args.rule))
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
