"""Per-hospital online adaptation of the Mondrian marginal: development test (``--rule``; online_tail_dev_rule.v1).

Per MIMIC training sample K and cell: the own_trajectory point (``joint_forecast_dev.cell_rows``) on each test
database's deployment clock.  Three marginals on the same scored rows: S = static Mondrian (static point + the
category-normalized conformal errors of ``tail_calibration_dev.calibrate``), P = the frozen online point correction +
the same static shape, M = ``jepa_deployment_form_dev.deploy_distribution`` (online point correction + Mondrian
MultiQT with the frozen v12 part C settings).  Single threshold events per cell (O/E), and the multivariate blocks
under the Gaussian copula by ensemble copula coupling (energy and variogram score, patient-cluster bootstrap) with
same-horizon joint events.  ``--summarize`` combines the five samples.  Development evidence only; research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jepa_deployment_form_dev import deploy_distribution
from jepa_gate import DATABASE_NAMES, MIN_ROWS, MIN_SUBJECTS, ROBUST_SHARE
from joint_forecast_dev import EVENTS, MIN_JOINT_EVENTS, MIN_PAIRS, block_vectors, cell_rows
from mimic_to_eicu_49_cell_external_audit import stable_seed
from online_joint_dev import MEMBERS, assemble, at_member_levels, frozen_multiqt
from osler_jepa.joint_copula import couple, energy_score, latent_correlation, latent_draws, variogram_score
from site_adaptive_external_audit import frozen_point_config
from stacked_expert_dev import dev_cohort, load_rule
from tail_calibration_dev import BAND, calibrate, compare, joint_events, member_latent, members

MARGINALS, CANDIDATES, COPULAS = ("S", "P", "M"), ("P", "M"), ("independent", "gaussian")
COMPARISONS = (("P:gaussian", "S:gaussian"), ("M:gaussian", "S:gaussian"), ("S:gaussian", "S:independent"),
               ("P:gaussian", "P:independent"), ("M:gaussian", "M:independent"))


def cell_marginals(rows: pd.DataFrame, database: str, point_config, multiqt, cell: str) -> dict | None:
    """S, P and M member values of one cell on the scored rows of one test database."""

    calibration = rows[rows["role"].isin(["selection", "conformal"])].reset_index(drop=True)
    test = rows[rows["role"] == database].reset_index(drop=True)
    if len(test) < MIN_ROWS or test["subject_id"].nunique() < MIN_SUBJECTS:
        return None
    corrected, quantiles, scored = deploy_distribution(calibration, test, "point", cell, point_config, multiqt)
    shape = calibrate(pd.DataFrame({"subject_id": calibration["subject_id"], "role": calibration["role"],
                                    f"point:{cell}": calibration["point"], f"current:{cell}": calibration["current"],
                                    f"future:{cell}": calibration["future"]}), cell)
    index = np.flatnonzero(scored)
    kept = test.iloc[index]
    current = kept["current"].to_numpy()[:, None]
    latent = member_latent(len(index), MEMBERS)
    static = lambda point: members("mondrian", [shape], point[:, None], current, latent)[:, :, 0].astype(np.float32)
    return {"index": pd.MultiIndex.from_arrays([kept["stay_id"], kept["t"]]),
            "S": static(kept["point"].to_numpy()), "P": static(corrected[index]),
            "M": at_member_levels(quantiles[index], np.asarray(multiqt.levels)).astype(np.float32),
            "future": kept["future"].to_numpy(), "current": current[:, 0],
            "subject_id": kept["subject_id"].to_numpy(), "hospitalid": kept["hospitalid"].to_numpy()}


def single_events(marginals: dict[str, dict]) -> dict:
    output = {}
    for cell, m in marginals.items():
        target, observed, current = cell.split("@")[0], m["future"], m["current"]
        entry = {"rows": int(len(observed))}
        for kind in MARGINALS:
            low, high = np.quantile(m[kind], [0.05, 0.95], axis=1)
            entry[f"{kind}_tails"] = [float(np.mean(observed < low)), float(np.mean(observed > high))]
        for label, (event_target, event) in EVENTS.items():
            if event_target != target:
                continue
            happened = event(observed, current)
            entry[label] = {"events": int(happened.sum()), "scored": bool(happened.sum() >= MIN_JOINT_EVENTS)}
            if entry[label]["scored"]:
                for kind in MARGINALS:
                    probability = np.mean(event(m[kind], current[:, None]), axis=1)
                    entry[label][kind] = {"o_e": float(happened.mean() / max(probability.mean(), 1e-12)),
                                          "brier": float(np.mean((probability - happened) ** 2))}
        output[cell] = entry
    return output


def dependence_inputs(per_cell: dict[str, pd.DataFrame]) -> tuple[np.ndarray, dict[str, float]]:
    """The Gaussian copula's correlation (point errors on MIMIC selection patients) and each cell's scale unit (IQR of
    the errors on MIMIC conformal patients)."""

    cells = list(per_cell)
    selection = pd.concat({cell: rows[rows["role"] == "selection"].set_index(["stay_id", "t"])["future"]
                           - rows[rows["role"] == "selection"].set_index(["stay_id", "t"])["point"]
                           for cell, rows in per_cell.items()}, axis=1)[cells]
    units = {}
    for cell, rows in per_cell.items():
        conformal = rows[rows["role"] == "conformal"]
        units[cell] = float(np.subtract(*np.quantile(conformal["future"] - conformal["point"], [0.75, 0.25])))
    return latent_correlation(selection, MIN_PAIRS), units


def run_sample(frame: pd.DataFrame, sample: int, rule: dict, point_config, multiqt) -> dict:
    """``frame``: the cohort (``stacked_expert_dev.load_cohort``) holding MIMIC and the rule's test databases."""

    targets, horizons = tuple(rule["data"]["targets"]), tuple(rule["data"]["horizons_hours"])
    per_cell = cell_rows(frame, sample, {**rule, "marginal": {"system": "own_trajectory"}})
    correlation, units = dependence_inputs(per_cell)
    result = {"schema": "online_tail_dev.v1", "rule": rule["path"], "sample": sample, "databases": {}}
    for database in rule["data"]["test_databases"]:
        marginals = database_marginals(per_cell, database, point_config, multiqt)
        result["databases"][DATABASE_NAMES[database]] = {
            "single_events": single_events(marginals),
            **block_reports(marginals, list(per_cell), database, sample, targets, horizons, correlation, units, MARGINALS,
                            COMPARISONS)}
    return {**result, "development_evidence_only": True, "causal_claim_allowed": False}


def database_marginals(per_cell: dict[str, pd.DataFrame], database: str, point_config, multiqt) -> dict[str, dict]:
    """S, P and M members of every cell with enough support on one test database."""

    marginals = {}
    for cell, rows in per_cell.items():
        entry = cell_marginals(rows, database, point_config, multiqt, cell)
        if entry is not None:
            marginals[cell] = entry
    return marginals


def block_reports(marginals: dict[str, dict], cells: list[str], database: str, sample: int, targets, horizons, correlation,
                  units, kinds: tuple[str, ...], comparisons) -> dict:
    """Per block: the variogram and energy scores of the ``kinds`` marginals under each copula, compared as
    ``comparisons`` (patient-cluster bootstrap), with same-horizon joint events (cross_target)."""

    name, report = DATABASE_NAMES[database], {}
    for block, groups in block_vectors(cells, targets, horizons).items():
        losses = {f"{k}:{c}": {"variogram": [], "energy": []} for k in kinds for c in COPULAS}
        frames, events, covered = [], {}, set()
        for group in groups:
            vectors = assemble(marginals, group, kinds)
            if vectors is None:
                continue
            covered |= {cell for j, cell in enumerate(vectors["cells"]) if vectors["labelled"][:, j].any()}
            position = [cells.index(c) for c in vectors["cells"]]
            latents = dict(zip(COPULAS, latent_draws(len(vectors["labelled"]), MEMBERS, correlation[np.ix_(position, position)],
                                                     seed=stable_seed("online_tail", sample, database, block, *vectors["cells"]))))
            ensembles = {f"{k}:{c}": couple(vectors[k], latents[c]) for k in kinds for c in COPULAS}
            scale = np.array([units[c] for c in vectors["cells"]])
            observed = np.nan_to_num(vectors["observed"]) / scale
            for system, values in ensembles.items():
                losses[system]["variogram"].append(variogram_score(values / scale, observed, vectors["labelled"]))
                losses[system]["energy"].append(energy_score(values / scale, observed, vectors["labelled"]))
            frames.append(pd.DataFrame({"subject_id": vectors["subject_id"], "hospitalid": vectors["hospitalid"]}))
            if block == "cross_target":
                events[group[0].split("@")[1]] = joint_events(vectors["observed"], vectors["current"], vectors["labelled"],
                                                              ensembles, vectors["cells"])
        if not frames:                                   # no vector of this block has >= 2 labelled cells
            report[block] = {"vectors": 0, "patients": 0, "evaluable": False}
            continue
        frame = pd.concat(frames, ignore_index=True)
        entry = {"vectors": int(len(frame)), "patients": int(frame["subject_id"].nunique()), "evaluable": True,
                 "groups": len(frames), "covered_cells": sorted(covered)}
        for score in ("variogram", "energy"):
            totals = {system: np.concatenate(values[score]) for system, values in losses.items()}
            entry[score] = {f"{a}_vs_{b}": compare(frame, totals[a], totals[b], stable_seed("online_tail", sample, database, block, score))
                            for a, b in comparisons}
        if events:
            entry["joint_events"] = events
        report[block] = entry
        print(json.dumps({"sample": sample, "database": name, "block": block,
                          **{f"{s}:{k}": [round(v["relative"], 4), v["better"], v["worse"]]
                             for s in ("variogram", "energy") for k, v in entry[s].items()}}), flush=True)
    return report


def summarize(paths: list[Path]) -> dict:
    samples = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    share = lambda flags: float(np.mean(flags))
    blocks, single, joint = {}, {}, {}
    for database in samples[0]["databases"]:
        for block in ("cross_target", "across_horizons"):
            entries = [s["databases"][database][block] for s in samples if s["databases"][database][block].get("evaluable", True)]
            if not entries:
                continue
            blocks[f"{database}:{block}"] = {
                score: {comparison: {"better_share": share([e[score][comparison]["better"] for e in entries]),
                                     "worse_share": share([e[score][comparison]["worse"] for e in entries]),
                                     "median_relative": float(np.median([e[score][comparison]["relative"] for e in entries]))}
                        for comparison in entries[0][score]}
                for score in ("variogram", "energy")}
        cells = set.intersection(*(set(s["databases"][database]["single_events"]) for s in samples))
        for cell in sorted(cells):
            per_sample = [s["databases"][database]["single_events"][cell] for s in samples]
            tails = {k: np.round(np.mean([p[f"{k}_tails"] for p in per_sample], axis=0), 4).tolist() for k in MARGINALS}
            for label in EVENTS:
                scored = [p[label] for p in per_sample if p.get(label, {}).get("scored")]
                if not scored:
                    continue
                entry = {"scored_samples": len(scored), "tails": tails}
                for kind in MARGINALS:
                    entry[f"{kind}_o_e"] = [round(min(e[kind]["o_e"] for e in scored), 3), round(max(e[kind]["o_e"] for e in scored), 3)]
                    entry[f"{kind}_in_band"] = share([BAND[0] <= e[kind]["o_e"] <= BAND[1] for e in scored]) >= ROBUST_SHARE
                for kind in CANDIDATES:
                    closer = share([abs(np.log(e[kind]["o_e"])) < abs(np.log(e["S"]["o_e"])) for e in scored])
                    entry[f"{kind}_closer_share"], entry[f"{kind}_robustly_closer"] = closer, closer >= ROBUST_SHARE
                single[f"{database}:{cell}:{label}"] = entry
        for horizon in samples[0]["databases"][database]["cross_target"].get("joint_events", {}):
            for pair in samples[0]["databases"][database]["cross_target"]["joint_events"][horizon]:
                scored = [s["databases"][database]["cross_target"]["joint_events"].get(horizon, {}).get(pair, {}) for s in samples]
                scored = [e for e in scored if e.get("scored")]
                if scored:
                    joint[f"{database}:{horizon}:{pair}"] = {
                        "scored_samples": len(scored),
                        **{f"{system}_o_e": [round(min(e[system]["o_e"] for e in scored), 3), round(max(e[system]["o_e"] for e in scored), 3)]
                           for system in scored[0] if ":" in system}}
    decided = [entry for entry in single.values() if entry["scored_samples"] >= 4]
    decision = {"events_decided": len(decided), "in_band": {k: int(sum(e[f"{k}_in_band"] for e in decided)) for k in MARGINALS}}
    passing = []
    for kind in CANDIDATES:
        closer = int(sum(e[f"{kind}_robustly_closer"] for e in decided))
        energy_ok = not any(entry["energy"][f"{kind}:gaussian_vs_S:gaussian"]["worse_share"] >= ROBUST_SHARE for entry in blocks.values())
        passed = bool(decided) and closer >= len(decided) / 2 and energy_ok
        decision[kind] = {"robustly_closer": closer, "energy_not_robustly_worse": energy_ok, "passes": passed}
        if passed:
            passing.append((closer, kind == "P", kind))
    decision["replaces_S"] = max(passing)[2] if passing else None
    return {"schema": "online_tail_dev_summary.v1", "rule": samples[0]["rule"], "samples": [str(p) for p in paths],
            "blocks": blocks, "single_events": single, "joint_events": joint, "decision": decision,
            "development_evidence_only": True, "causal_claim_allowed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int)
    group.add_argument("--summarize", nargs="+", type=Path)
    parser.add_argument("--rule", type=Path, default=Path("online_tail_dev_rule_20261031.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.summarize)
    else:
        jepa_rule = json.loads(Path("jepa_gate_rule_20261022.json").read_text(encoding="utf-8"))
        _, point_config = frozen_point_config(Path(jepa_rule["point_dev"]))
        rule = load_rule(args.rule)
        result = run_sample(dev_cohort(args.sample, rule), args.sample, rule, point_config, frozen_multiqt())
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
