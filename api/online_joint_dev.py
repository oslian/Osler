"""Online marginals under the dependence layer: development test (rule: online_joint_dev_rule_20261031.json).

Per MIMIC training sample K and cell: the own_trajectory point (``joint_forecast_dev.cell_rows``) goes through the
online distributional deployment form (``jepa_deployment_form_dev.deploy_distribution``: frozen online point
correction + Mondrian MultiQT with the frozen v12 part C settings) on each test database; the static marginal is the
same point + empirical quantiles of its MIMIC conformal errors.  Each marginal gives 50 member values per row.  The
dependence layer couples them by ensemble copula coupling (``osler_jepa.joint_copula.couple``) with the Gaussian
copula of the MIMIC selection errors; independent cells use identity-correlation ranks.  Readouts: the online 90%
interval gate per database:cell, joint vs independent multivariate scores over the online marginals, online vs
static, single and joint event O/E.  ``--summarize`` combines the five samples.  Development evidence only;
research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_value_conditional_interval_audit import GATE_AXES, interval_report
from jepa_deployment_form_dev import deploy_distribution
from jepa_gate import DATABASE_NAMES, MIN_ROWS, MIN_SUBJECTS, ROBUST_SHARE
from joint_forecast_dev import EVENT_PAIRS, EVENTS, MIN_JOINT_EVENTS, MIN_PAIRS, block_vectors, cell_rows
from mimic_to_eicu_49_cell_external_audit import paired_comparison, stable_seed
from osler_jepa.joint_copula import couple, energy_score, latent_correlation, latent_draws, variogram_score
from osler_jepa.predictive_distribution import MultiQTConfig
from site_adaptive_external_audit import frozen_point_config
from stacked_expert_dev import dev_cohort, load_rule

MEMBERS = 50
MEMBER_LEVELS = (np.arange(MEMBERS) + 0.5) / MEMBERS
SYSTEMS = ("online_joint", "online_independent", "static_joint", "static_independent")


def frozen_multiqt(path: Path = Path("threshold_event_candidates_20261029.json")) -> MultiQTConfig:
    record = json.loads(path.read_text(encoding="utf-8"))["multiqt"]
    return MultiQTConfig(**{key: tuple(value) if isinstance(value, list) else value for key, value in record.items()})


def at_member_levels(quantiles: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Each row's issued quantiles linearly interpolated at the member levels (all inside the level grid)."""

    upper = np.searchsorted(levels, MEMBER_LEVELS).clip(1, len(levels) - 1)
    weight = (MEMBER_LEVELS - levels[upper - 1]) / (levels[upper] - levels[upper - 1])
    return quantiles[:, upper - 1] * (1.0 - weight) + quantiles[:, upper] * weight


def cell_marginals(rows: pd.DataFrame, database: str, point_config, multiqt: MultiQTConfig, cell: str) -> dict | None:
    calibration = rows[rows["role"].isin(["selection", "conformal"])].reset_index(drop=True)
    test = rows[rows["role"] == database].reset_index(drop=True)
    if len(test) < MIN_ROWS or test["subject_id"].nunique() < MIN_SUBJECTS:
        return None
    _, quantiles, scored = deploy_distribution(calibration, test, "point", cell, point_config, multiqt)
    levels = np.asarray(multiqt.levels)
    conformal = calibration[calibration["role"] == "conformal"]
    static_offsets = np.quantile(conformal["future"].to_numpy() - conformal["point"].to_numpy(), MEMBER_LEVELS)
    index = np.flatnonzero(scored)
    future = test["future"].to_numpy()
    low, high = quantiles[:, levels.tolist().index(0.05)], quantiles[:, levels.tolist().index(0.95)]
    interval = interval_report(test, index, ((future >= low) & (future <= high))[index], ((high - low) / 2.0)[index],
                               GATE_AXES[DATABASE_NAMES[database]]["interval"])
    kept = test.iloc[index]
    return {
        "index": pd.MultiIndex.from_arrays([kept["stay_id"], kept["t"]]),
        "online": at_member_levels(quantiles[index], levels).astype(np.float32),
        "static": (kept["point"].to_numpy()[:, None] + static_offsets[None, :]).astype(np.float32),
        "future": future[index], "current": kept["current"].to_numpy(),
        "subject_id": kept["subject_id"].to_numpy(), "hospitalid": kept["hospitalid"].to_numpy(),
        "interval": {"pass": bool(interval["pass"]), "coverage": [round(interval[a], 4) for a in interval["gate_axes"]],
                     "median_half_width": interval["median_half_width"]},
    }


def assemble(marginals: dict[str, dict], group: list[str], kinds: tuple[str, ...]) -> dict | None:
    """The group's vectors (member values of each marginal in ``kinds``) on anchors where >= 2 of its cells are
    labelled and scored; None when fewer than 2 cells are present or no anchor has 2 of them."""

    present = [c for c in group if c in marginals]
    if len(present) < 2:
        return None
    union = marginals[present[0]]["index"]
    for cell in present[1:]:
        union = union.union(marginals[cell]["index"])
    positions = {cell: union.get_indexer(marginals[cell]["index"]) for cell in present}
    labelled = np.zeros((len(union), len(present)), dtype=bool)
    for j, cell in enumerate(present):
        labelled[positions[cell], j] = True
    keep = np.flatnonzero(labelled.sum(axis=1) >= 2)
    if not len(keep):
        return None
    labelled = labelled[keep]
    shape = (len(union), MEMBERS, len(present))
    values = {kind: np.zeros(shape, np.float32) for kind in kinds}
    observed, current = np.full((len(union), len(present)), np.nan), np.full((len(union), len(present)), np.nan)
    subjects, hospitals = np.empty(len(union), dtype=object), np.empty(len(union), dtype=object)
    for j, cell in enumerate(present):
        at, m = positions[cell], marginals[cell]
        for kind in values:
            values[kind][at, :, j] = m[kind]
        observed[at, j], current[at, j] = m["future"], m["current"]
        subjects[at], hospitals[at] = m["subject_id"], m["hospitalid"]            # same stay -> same patient and hospital
    return {"cells": present, "labelled": labelled, "observed": observed[keep], "current": current[keep],
            **{kind: values[kind][keep] for kind in kinds},
            "subject_id": subjects[keep].astype(str), "hospitalid": hospitals[keep].astype(str)}


def event_entry(vectors: dict, ensembles: dict[str, np.ndarray]) -> dict:
    targets = [c.split("@")[0] for c in vectors["cells"]]
    output = {}
    for first, second in EVENT_PAIRS:
        (ta, fa), (tb, fb) = EVENTS[first], EVENTS[second]
        if ta not in targets or tb not in targets:
            continue
        a, b = targets.index(ta), targets.index(tb)
        both = vectors["labelled"][:, a] & vectors["labelled"][:, b]
        observed, current = vectors["observed"][both], vectors["current"][both]
        happened = fa(observed[:, a], current[:, a]) & fb(observed[:, b], current[:, b])
        entry = {"labelled_rows": int(both.sum()), "events": int(happened.sum()), "scored": bool(happened.sum() >= MIN_JOINT_EVENTS)}
        if entry["scored"]:
            entry["observed_rate"] = float(happened.mean())
            for kind in ("static", "online"):                                            # marginals: coupling-free
                for label, column, event in ((first, a, fa), (second, b, fb)):
                    single = event(observed[:, column], current[:, column])
                    predicted = np.mean(event(vectors[kind][both][:, :, column], current[:, column][:, None]), axis=1)
                    entry[f"{kind} marginal {label}"] = float(single.mean() / max(predicted.mean(), 1e-12))
            for system, members in ensembles.items():
                part = members[both]
                probability = np.mean(fa(part[:, :, a], current[:, a][:, None]) & fb(part[:, :, b], current[:, b][:, None]), axis=1)
                entry[system] = {"o_e": float(happened.mean() / max(probability.mean(), 1e-12)),
                                 "brier": float(np.mean((probability - happened) ** 2))}
        output[f"{first} & {second}"] = entry
    return output


def compare(frame: pd.DataFrame, candidate: np.ndarray, comparator: np.ndarray, seed: int) -> dict:
    patient = paired_comparison(frame, np.arange(len(frame)), candidate, comparator, seed=seed)["patient"]
    high, low = patient["bootstrap_high"], patient["bootstrap_low"]
    return {"relative": float(candidate.mean() / comparator.mean() - 1.0), "ci": [low, high],
            "better": bool(high is not None and high < 0), "worse": bool(low is not None and low > 0)}


def run_sample(sample: int, rule: dict, point_config, multiqt: MultiQTConfig) -> dict:
    targets, horizons = tuple(rule["data"]["targets"]), tuple(rule["data"]["horizons_hours"])
    per_cell = cell_rows(dev_cohort(sample, rule), sample, {**rule, "marginal": {"system": "own_trajectory"}})
    cells = list(per_cell)
    selection = pd.concat({cell: rows[rows["role"] == "selection"].set_index(["stay_id", "t"])["future"]
                           - rows[rows["role"] == "selection"].set_index(["stay_id", "t"])["point"]
                           for cell, rows in per_cell.items()}, axis=1)[cells]
    correlation = latent_correlation(selection, MIN_PAIRS)
    units = {}
    for cell, rows in per_cell.items():
        conformal = rows[rows["role"] == "conformal"]
        units[cell] = float(np.subtract(*np.quantile(conformal["future"] - conformal["point"], [0.75, 0.25])))
    result = {"schema": "online_joint_dev.v1", "rule": rule["path"], "sample": sample, "intervals": {}, "databases": {}}
    for database in rule["data"]["test_databases"]:
        name = DATABASE_NAMES[database]
        marginals = {}
        for cell in cells:
            entry = cell_marginals(per_cell[cell], database, point_config, multiqt, cell)
            if entry is not None:
                marginals[cell] = entry
                result["intervals"][f"{name}:{cell}"] = entry["interval"]
        report = {}
        for block, groups in block_vectors(cells, targets, horizons).items():
            scores = {system: {"variogram": [], "energy": []} for system in SYSTEMS}
            frames, events = [], {}
            for group in groups:
                vectors = assemble(marginals, group, ("online", "static"))
                if vectors is None:
                    continue
                position = [cells.index(c) for c in vectors["cells"]]
                independent, joint = latent_draws(len(vectors["labelled"]), MEMBERS, correlation[np.ix_(position, position)],
                                                  seed=stable_seed("online_joint", sample, database, block, *vectors["cells"]))
                ensembles = {f"{kind}_{coupling}": couple(vectors[kind], draws)
                             for kind in ("online", "static") for coupling, draws in (("joint", joint), ("independent", independent))}
                scale = np.array([units[c] for c in vectors["cells"]])
                observed = np.nan_to_num(vectors["observed"]) / scale
                for system, members in ensembles.items():
                    scores[system]["variogram"].append(variogram_score(members / scale, observed, vectors["labelled"]))
                    scores[system]["energy"].append(energy_score(members / scale, observed, vectors["labelled"]))
                frames.append(pd.DataFrame({"subject_id": vectors["subject_id"], "hospitalid": vectors["hospitalid"]}))
                if block == "cross_target":
                    events[group[0].split("@")[1]] = event_entry(vectors, ensembles)
            frame = pd.concat(frames, ignore_index=True)
            pooled = {system: {score: np.concatenate(values) for score, values in by.items()} for system, by in scores.items()}
            entry = {"vectors": int(len(frame)), "patients": int(frame["subject_id"].nunique())}
            for score in ("variogram", "energy"):
                entry[score] = {
                    "joint_vs_independent": compare(frame, pooled["online_joint"][score], pooled["online_independent"][score],
                                                    stable_seed("online_joint", sample, database, block, score, "ji")),
                    "online_vs_static": compare(frame, pooled["online_joint"][score], pooled["static_joint"][score],
                                                stable_seed("online_joint", sample, database, block, score, "os"))}
            if events:
                entry["joint_events"] = events
            report[block] = entry
            print(json.dumps({"sample": sample, "database": name, "block": block, "vectors": entry["vectors"],
                              **{s: [round(entry[s]["joint_vs_independent"]["relative"], 4), entry[s]["joint_vs_independent"]["better"],
                                     round(entry[s]["online_vs_static"]["relative"], 4)] for s in ("variogram", "energy")}}), flush=True)
        result["databases"][name] = report
    return {**result, "development_evidence_only": True, "causal_claim_allowed": False}


def summarize(paths: list[Path]) -> dict:
    samples = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    dtaci = [json.loads(Path(f"own_trajectory_dev_s{s['sample']}_20261031.json").read_text(encoding="utf-8")) for s in samples]
    intervals = {}
    for key in sorted(set.intersection(*(set(s["intervals"]) for s in samples))):
        database, cell = key.split(":", 1)
        online = [s["intervals"][key]["pass"] for s in samples]
        reference = [d["cells"][cell]["databases"].get(database, {}).get("interval", {}).get("pass", False) for d in dtaci]
        intervals[key] = {"online_pass_share": float(np.mean(online)), "online_robust": bool(np.mean(online) >= ROBUST_SHARE),
                          "dtaci_pass_share": float(np.mean(reference)), "dtaci_robust": bool(np.mean(reference) >= ROBUST_SHARE)}
    blocks = {}
    for database in samples[0]["databases"]:
        for block in samples[0]["databases"][database]:
            entries = [s["databases"][database][block] for s in samples]
            blocks[f"{database}:{block}"] = {
                score: {comparison: {"better_share": float(np.mean([e[score][comparison]["better"] for e in entries])),
                                     "worse_share": float(np.mean([e[score][comparison]["worse"] for e in entries])),
                                     "median_relative": float(np.median([e[score][comparison]["relative"] for e in entries])),
                                     "robust_better": bool(np.mean([e[score][comparison]["better"] for e in entries]) >= ROBUST_SHARE)}
                        for comparison in ("joint_vs_independent", "online_vs_static")}
                for score in ("variogram", "energy")}
    return {"schema": "online_joint_dev_summary.v1", "rule": samples[0]["rule"], "samples": [str(p) for p in paths],
            "interval_robust": {"online": sum(v["online_robust"] for v in intervals.values()),
                                "dtaci": sum(v["dtaci_robust"] for v in intervals.values()), "cells": len(intervals)},
            "intervals": intervals, "blocks": blocks, "development_evidence_only": True, "causal_claim_allowed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int)
    group.add_argument("--summarize", nargs="+", type=Path)
    parser.add_argument("--rule", type=Path, default=Path("online_joint_dev_rule_20261031.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.summarize)
    else:
        jepa_rule = json.loads(Path("jepa_gate_rule_20261022.json").read_text(encoding="utf-8"))
        _, point_config = frozen_point_config(Path(jepa_rule["point_dev"]))
        result = run_sample(args.sample, load_rule(args.rule), point_config, frozen_multiqt())
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
