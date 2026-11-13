"""Per-database takeover gate for the system-organ-body JEPA over the frozen expert anchors.

Inputs: for each MIMIC training sample, the 7 per-seed exports of
``teacher_anchored_joint_jepa_audit.py --export-predictions`` (anchor and every
independently trained matched hierarchy model, ``HIERARCHY_MODELS``) and that
run's audit JSON (latent SVD check).  Rule: ``jepa_gate_rule_20261022.json``.

Per seed and cell:

1. path choice on the calibration set's selection patients only
   (``osler_jepa.patient_split``): a hierarchy model is eligible when its static
   prediction beats each of its matched ablations (``MATCHED_ABLATIONS``) on the
   patient axis; the eligible model with the lowest MAE is chosen.
   ``system_only`` has no organ route, so it can never pass the required
   no-route ablation and never takes over;
2. per test database, every system goes through the frozen deployment form
   (``jepa_deployment_form_dev.deploy``: shared deployment clock, online point
   correction, DtACI; warm-up stays unscored); the chosen model passes there
   only if, on that database's point axes, it beats the anchor, persistence,
   each matched ablation and ``system_only``, its interval gate passes and the
   latent SVD check passes.

A cell passes a database in a sample only if every seed passes; it is robust
on that database if it passes in >= 80% of MIMIC samples, and promotion
requires >= 5 samples (AGENTS.md).  The time validation axis is not
implemented and is reported as not evaluated, never as passed.  Development
evidence only; research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_value_conditional_interval_audit import GATE_AXES, interval_report
from jepa_deployment_form_dev import deploy
from mimic_to_eicu_49_cell_external_audit import paired_comparison, stable_seed
from osler_jepa.adjacent_hierarchy import HIERARCHY_MODELS, MATCHED_ABLATIONS
from osler_jepa.patient_split import selection_patients
from site_adaptive_external_audit import frozen_dtaci_config, frozen_network_config, frozen_point_config


DATABASE_NAMES = {"eicu": "eICU", "nwicu": "NWICU", "zigong": "Zigong", "sicdb": "SICdb", "hirid": "HiRID", "emory": "Emory"}
CANDIDATES = tuple(name for name in HIERARCHY_MODELS if name != "system_only")
ROBUST_SHARE = 0.8
PROMOTION_SAMPLES = 5
MIN_ROWS, MIN_SUBJECTS = 100, 30


def required_ablations(model: str) -> tuple[str, ...]:
    """Matched ablations plus the trained no-route model."""

    return tuple(dict.fromkeys(MATCHED_ABLATIONS[model] + ("system_only",)))


def choose_model(calibration: pd.DataFrame, cell: str) -> dict:
    """Path choice on selection patients only; ``None`` when no hierarchy model earns its components."""

    rows = calibration[selection_patients(calibration["subject_id"].to_numpy())].reset_index(drop=True)
    future, index = rows["future"].to_numpy(), np.arange(len(rows))
    mae = {name: float(np.abs(rows[name].to_numpy() - future).mean()) for name in HIERARCHY_MODELS}
    evidence, eligible = {}, []
    for model in CANDIDATES:
        checks = {}
        for ablation in MATCHED_ABLATIONS[model]:
            result = paired_comparison(rows, index, np.abs(rows[model].to_numpy() - future), np.abs(rows[ablation].to_numpy() - future),
                                       seed=stable_seed(cell, "select", model, ablation))["patient"]
            checks[ablation] = {"pass": bool(result["pass"]), "ci": [result["bootstrap_low"], result["bootstrap_high"]]}
        evidence[model] = checks
        if all(check["pass"] for check in checks.values()):
            eligible.append(model)
    chosen = min(eligible, key=lambda name: mae[name]) if eligible else None
    return {"chosen": chosen, "selection_rows": int(len(rows)), "selection_mae": mae, "evidence": evidence}


def score_database(calibration: pd.DataFrame, test: pd.DataFrame, name: str, cell: str, chosen: str, configs) -> dict:
    axes = GATE_AXES[name]
    systems = ("anchor", chosen) + required_ablations(chosen)
    outputs = {system: deploy(calibration, test, system, cell, *configs) for system in systems}
    index = np.flatnonzero(outputs["anchor"][2])                    # scored rows: identical clock for every system
    future = test["future"].to_numpy()
    loss = {system: np.abs(outputs[system][0] - future)[index] for system in systems}
    loss["persistence"] = np.abs(test["current"].to_numpy() - future)[index]
    comparisons = {}
    for other in ("anchor", "persistence") + required_ablations(chosen):
        result = paired_comparison(test, index, loss[chosen], loss[other], seed=stable_seed(cell, name, chosen, other))
        comparisons[other] = {"pass": all(result[axis]["pass"] for axis in axes["point"]),
                              "ci": {axis: [result[axis]["bootstrap_low"], result[axis]["bootstrap_high"]] for axis in axes["point"]}}
    width = outputs[chosen][1][index]
    interval = interval_report(test, index, loss[chosen] <= width, width, axes["interval"])
    return {
        "scored_rows": int(len(index)),
        "corrected_mae": {system: float(loss[system].mean()) for system in loss},
        "beats": comparisons,
        "interval": {"pass": bool(interval["pass"]), "coverage": [round(interval[a], 4) for a in interval["gate_axes"]]},
        "pass": bool(all(c["pass"] for c in comparisons.values()) and interval["pass"]),
    }


def gate_seed(predictions: Path, svd_pass: bool, configs) -> dict[str, dict]:
    """{"<database>:<cell>": result} for one seed of one MIMIC sample."""

    rows = pd.read_parquet(predictions)
    output = {}
    for cell, cell_rows in rows.groupby("cell", sort=False):
        calibration = cell_rows[cell_rows["split"] == "calibration"].reset_index(drop=True)
        if len(calibration) < MIN_ROWS:
            continue
        choice = choose_model(calibration, cell)
        for database, test in cell_rows[cell_rows["split"] == "test"].groupby("database"):
            test = test.reset_index(drop=True)
            name = DATABASE_NAMES[str(database).lower()]
            if len(test) < MIN_ROWS or test["subject_id"].nunique() < MIN_SUBJECTS:
                continue
            entry = {"chosen": choice["chosen"], "selection": choice, "latent_svd_pass": svd_pass}
            if choice["chosen"] is None:
                entry.update(pass_=False, reason="no hierarchy model beats its matched ablations on selection patients")
            else:
                scored = score_database(calibration, test, name, cell, choice["chosen"], configs)
                entry.update(scored, pass_=bool(scored["pass"] and svd_pass))
            entry["pass"] = entry.pop("pass_")
            output[f"{name}:{cell}"] = entry
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", type=Path, default=Path("jepa_gate_rule_20261022.json"))
    parser.add_argument("--runs", nargs="+", required=True,
                        help="one '<audit.json>:<predictions prefix>' per MIMIC training sample")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    _, point_config = frozen_point_config(Path(rule["point_dev"]))
    _, dtaci_config = frozen_dtaci_config(Path(rule["dtaci_dev"]), Path(rule["dtaci_prior"]))
    configs = (point_config, dtaci_config, frozen_network_config(Path(rule["network_dev"]), dtaci_config))
    samples = []
    for run in args.runs:
        audit_path, prefix = run.split(":", 1)
        reports = {int(r["seed"]): r for r in json.loads(Path(audit_path).read_text(encoding="utf-8"))["reports"]}
        missing = [seed for seed in rule["seeds"] if seed not in reports]
        if missing:
            raise ValueError(f"{audit_path} lacks seeds {missing}")
        per_seed = {seed: gate_seed(Path(f"{prefix}.seed{seed}.parquet"), bool(reports[seed]["latent_svd"]["pass"]), configs)
                    for seed in rule["seeds"]}
        keys = sorted(set.intersection(*(set(v) for v in per_seed.values())))
        samples.append({"run": run, "per_seed": {str(k): v for k, v in per_seed.items()},
                        "cells": {key: {"seeds_passing": int(sum(per_seed[s][key]["pass"] for s in rule["seeds"])),
                                        "pass": bool(all(per_seed[s][key]["pass"] for s in rule["seeds"])),
                                        "chosen": sorted({str(per_seed[s][key]["chosen"]) for s in rule["seeds"]})}
                                  for key in keys}})
        print(json.dumps({"run": run, "passing": sorted(k for k, v in samples[-1]["cells"].items() if v["pass"])}), flush=True)
    keys = sorted(set.intersection(*(set(sample["cells"]) for sample in samples)))
    cells = {}
    for key in keys:
        share = float(np.mean([sample["cells"][key]["pass"] for sample in samples]))
        cells[key] = {"sample_pass_share": share, "robust": bool(share >= ROBUST_SHARE)}
    output = {
        "schema": "jepa_gate.v1", "rule": str(args.rule), "mimic_samples": len(samples),
        "promotion_sample_requirement_met": len(samples) >= PROMOTION_SAMPLES,
        "robust_cells": sorted(k for k, v in cells.items() if v["robust"]), "cells": cells, "samples": samples,
        "time_axis": "not_evaluated", "agents_complete_gate": False,
        "development_candidates_only": True, "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"robust_cells": output["robust_cells"], "mimic_samples": len(samples)}, indent=1))


if __name__ == "__main__":
    main()
