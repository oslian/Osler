"""Select the normalized-conformal residual scale on the external development databases.

External glucose under-coverage concentrates in extreme current values
(> 250 mg/dL: 0.75-0.83) and volatile rows (|change since previous anchor| >
100: 0.81-0.84): the v2 quintile scale lumps the whole upper fifth into one
bin, and eICU's laboratory-glucose cohort has seven times MIMIC's share of
values > 350.  Candidates (``osler_jepa.interval_scale``): ``quintile``
(current), ``tail8`` (tail-refined value bins) and ``tail8_change`` (plus a
separable volatility factor).  All run on the same fitted predictions and
streams in the frozen deployment form (selected recipes, point recalibration,
DtACI with the frozen starting multiplier), MIMIC-only training, 5 samples.
Development databases: eICU, NWICU, Zigong (all already used).

Predeclared rule (written before any candidate was scored):

- interval-robust = interval gate passed in >= 80% of samples;
- select the scheme with the most interval-robust cells summed over the three
  databases, ties to the simpler (order above); adopt only if strictly more
  than ``quintile``;
- reported alongside: leave-one-database-out, choosing on two databases and
  scoring the third.

Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from external_recipe_selection_dev import PREFIX, SOURCE_ROWS, TEST_ROWS, load_cohorts
from external_value_conditional_interval_audit import GATE_AXES, evaluability, interval_report
from mimic_to_eicu_49_cell_external_audit import fit_continuous_recipe, stable_seed
from multi_sample_external_audit import mimic_sample
from osler_jepa.recipe_availability import deployable_recipe
from osler_jepa.interval_scale import SCHEMES
from osler_jepa.site_adaptive_point import PointCorrectionConfig
from site_adaptive_external_audit import DTACI_METHOD, frozen_dtaci_config, frozen_point_config, scored_only, simulate_sites
from target_specific_domain_conformal_audit import _safe_float


DATABASES = ("eICU", "NWICU", "Zigong")
CANDIDATES = SCHEMES
ROBUST_SHARE = 0.8


def label(scheme: str) -> str:
    return scheme


def audit_group(args: argparse.Namespace, contracts: list[dict]) -> list[dict]:
    _, point_config = frozen_point_config(args.point_dev)
    _, dtaci_config = frozen_dtaci_config(args.dtaci_dev, args.dtaci_prior)
    variants = {label(scheme): scheme for scheme in CANDIDATES}
    recipes = json.loads(args.recipes.read_text(encoding="utf-8"))["recipes"]
    horizon = int(contracts[0]["horizon_hours"])
    raw = {
        "MIMIC-IV": pd.read_parquet(args.mimic_template.format(horizon=horizon)),
        "NWICU": pd.read_parquet(args.nwicu_template.format(horizon=horizon)),
        "Zigong": pd.read_parquet(args.zigong_template.format(horizon=horizon)),
    }
    cells = []
    for contract in contracts:
        cell, target = contract["cell"], contract["target"]
        cohorts = load_cohorts(args, raw, target, horizon)
        trains = [mimic_sample(cohorts["MIMIC-IV"], target=target, horizon=horizon, sample=s, rows=SOURCE_ROWS)
                  for s in range(args.samples)]
        entry = {"cell": cell, "databases": {}}
        for database in DATABASES:
            site = cohorts[database]
            if evaluability(database, site) is not None:
                continue
            recipe, _ = deployable_recipe(recipes[cell], site, target)
            cell_point = PointCorrectionConfig() if recipe == "persistence" else point_config
            axes = GATE_AXES[database]
            passes = {name: [] for name in variants}
            coverage = {name: [] for name in variants}
            widths = {name: [] for name in variants}
            for sample, train in enumerate(trains):
                frame = pd.concat([train, site], ignore_index=True, sort=False)
                frame.sort_values(["subject_id", "anchor_time"], inplace=True)
                frame.reset_index(drop=True, inplace=True)
                subject = frame["subject_id"].astype(str)
                fitted = fit_continuous_recipe(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                                               np.flatnonzero(subject.str.startswith(PREFIX[database])),
                                               target=target, source=recipe, seed=stable_seed(cell, "mimic_to_eicu"))
                rows = fitted["test_rows"]
                simulated = {name: simulate_sites(frame, rows, fitted, target, horizon, cell, sample, cell_point,
                                                  dtaci_config, scale_scheme=scheme, labels=(DTACI_METHOD,))[DTACI_METHOD]
                             for name, scheme in variants.items()}
                for name in variants:
                    scored_rows, values = scored_only(rows, simulated[name])
                    observed = _safe_float(frame, f"future_{target}")[scored_rows]
                    loss = np.abs(values["prediction"] - observed)
                    report = interval_report(frame, scored_rows, loss <= values["width"], values["width"], axes["interval"])
                    passes[name].append(report["pass"])
                    coverage[name].append([report[axis] for axis in report["gate_axes"]])
                    widths[name].append(report["median_half_width"])
            entry["databases"][database] = {
                "recipe": recipe,
                **{name: {
                    "interval_pass_share": float(np.mean(passes[name])),
                    "interval_robust": bool(np.mean(passes[name]) >= ROBUST_SHARE),
                    "mean_gate_coverage": np.mean(coverage[name], axis=0).round(4).tolist(),
                    "median_half_width": float(np.mean(widths[name])),
                } for name in variants},
            }
        print(json.dumps({"cell": cell, **{d: [round(v[label(t)]["interval_pass_share"], 1) for t in CANDIDATES]
                                           for d, v in entry["databases"].items()}}), flush=True)
        cells.append(entry)
    return cells


def robust_count(cells: list[dict], database: str, scheme: str) -> int:
    return sum(c["databases"][database][label(scheme)]["interval_robust"] for c in cells if database in c["databases"])


def choose(cells: list[dict], databases) -> str:
    return max(CANDIDATES, key=lambda scheme: (sum(robust_count(cells, d, scheme) for d in databases), -CANDIDATES.index(scheme)))


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = sorted((r for r in development["cells"] if r["task_type"] != "binary_event_probability"),
                       key=lambda r: (int(r["horizon_hours"]), r["target"]))
    if args.cells:
        contracts = [r for r in contracts if r["cell"] in set(args.cells.split(","))]
    groups = [[r for r in contracts if int(r["horizon_hours"]) == h]
              for h in sorted({int(r["horizon_hours"]) for r in contracts})]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cells = [cell for group in pool.map(audit_group, [args] * len(groups), groups) for cell in group]
    lodo = {}
    for held_out in DATABASES:
        scheme = choose(cells, [d for d in DATABASES if d != held_out])
        lodo[held_out] = {"scheme": scheme, "held_out_robust": robust_count(cells, held_out, scheme),
                          "baseline_robust": robust_count(cells, held_out, "quintile")}
    lodo_total = sum(v["held_out_robust"] for v in lodo.values())
    baseline_total = sum(v["baseline_robust"] for v in lodo.values())
    totals = {scheme: sum(robust_count(cells, d, scheme) for d in DATABASES) for scheme in CANDIDATES}
    best = choose(cells, DATABASES)
    adopted = totals[best] > totals["quintile"]
    output = {
        "schema": "interval_scale_dev.v1",
        "totals": totals,
        "candidates": list(CANDIDATES),
        "robust_counts": {d: {label(t): robust_count(cells, d, t) for t in CANDIDATES} for d in DATABASES},
        "lodo": lodo,
        "lodo_total": lodo_total,
        "baseline_total": baseline_total,
        "adopted": bool(adopted),
        "selected_scheme": best if adopted else "quintile",
        "cells": cells,
        "independent_validation": False,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", type=Path, default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"))
    parser.add_argument("--recipes", type=Path, default=Path("external_recipe_selection_dev_20261019.json"))
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_harmonized_20261022"))
    parser.add_argument("--nwicu-template", default="nwicu_observation_transitions_{horizon}h_v5.parquet")
    parser.add_argument("--zigong-template", default="zigong_observation_transitions_{horizon}h_v2.parquet")
    parser.add_argument("--point-dev", type=Path, default=Path("site_adaptive_point_dev_20261016.json"))
    parser.add_argument("--dtaci-dev", type=Path, default=Path("site_adaptive_dtaci_dev_20261017.json"))
    parser.add_argument("--dtaci-prior", type=Path, default=Path("site_adaptive_dtaci_prior_dev_20261019.json"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cells", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({k: result[k] for k in ("totals", "robust_counts", "lodo", "lodo_total", "baseline_total", "adopted", "selected_scheme")}, indent=1))
