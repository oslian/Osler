"""Network-shared DtACI for multi-hospital deployments (development, eICU only).

Small eICU hospitals under-cover because each hospital adapts on its own few
outcomes.  In a network deployment (a health system or vendor network) the
hospitals go live together, so a hospital can borrow the network's adaptation.
Hierarchical online quantile tracking with the existing DtACI levels:

    log multiplier = network term (DtACI site level, learned from every hospital)
                   + hospital term (DtACI unit level keyed by hospital; used once
                     the hospital has ``min_unit_history`` resolved outcomes)

All hospitals go live together on the shared deployment clock: each hospital's
replay (``site_adaptive_interval_dev.replay``) is merged into one network stream
(``network_replay``), and an outcome resolves once the clock passes its
availability time.  Point predictions are the frozen per-hospital online
corrections (unchanged); day-one weights use the network's warm-up inputs
received before the first scored stay; warm-up stays are not scored.  Same frozen
deployment otherwise (selected recipes, DtACI meta 20, theta0 = 0.15), MIMIC
training, 5 samples.  Candidates: per-hospital (current), network with hospital
step ratio 0.5, network with ratio 1.0.

Only eICU has many hospitals, and no untouched multi-hospital database is
available, so the predeclared rule (written before any candidate was scored)
splits eICU hospitals into two fixed halves by hash:

- interval-robust = interval gate (patient-equal and hospital-equal coverage
  over that half's rows) passed in >= 80% of samples;
- select on one half (most interval-robust cells, ties to current, then ratio
  0.5), score the other half; do both directions;
- adopt only if the held-out total over both halves is strictly larger than
  the current form's.

Factual research-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from external_recipe_selection_dev import PREFIX, SOURCE_ROWS, TEST_ROWS
from external_value_conditional_interval_audit import GATE_AXES, evaluability, interval_report
from mimic_to_eicu_49_cell_external_audit import choose_eicu_cohort, fit_continuous_recipe, normalize_cohort, stable_seed
from multi_sample_external_audit import mimic_sample
from osler_jepa.interval_scale import fit_interval_scale
from osler_jepa.patient_split import selection_patients
from osler_jepa.recipe_availability import deployable_recipe
from osler_jepa.site_adaptive_conformal import covariate_shift_weights, dtaci_stream_widths
from osler_jepa.site_adaptive_point import PointCorrectionConfig
from site_adaptive_external_audit import DTACI_METHOD, SCALE_SCHEME, frozen_dtaci_config, frozen_point_config, simulate_sites
from site_adaptive_interval_dev import covariate_matrix, network_replay, replay
from target_specific_domain_conformal_audit import _safe_float


DATABASE = "eICU"
CANDIDATES = ("current", "network_0.5", "network_1.0")
ROBUST_SHARE = 0.8


def half(hospital: str) -> str:
    return "A" if int(hashlib.sha256(f"network_split:{hospital}".encode()).hexdigest(), 16) % 2 == 0 else "B"


def network_widths(frame, rows, fitted, corrected, target, horizon, cell, sample, config) -> pd.Series:
    """DtACI on the network stream (all hospitals on one clock); the unit level is keyed by hospital."""

    future, current = _safe_float(frame, f"future_{target}"), _safe_float(frame, f"{target}_t")
    held_out = fitted["calibration_rows"]
    residual = np.abs(fitted["calibration_prediction"] - future[held_out])
    choosing = selection_patients(frame.iloc[held_out]["subject_id"].to_numpy())
    sel, cal = held_out[choosing], held_out[~choosing]
    change = np.abs(current - _safe_float(frame, f"lag1__{target}_t"))
    scale = fit_interval_scale(SCALE_SCHEME, current[sel], change[sel], residual[choosing], frame.iloc[sel]["subject_id"].to_numpy())
    cal_scores = residual[~choosing] / scale(current[cal], change[cal])
    hospitals = frame.iloc[rows]["hospitalid"].astype(str).to_numpy()
    network = network_replay(frame, [replay(frame, rows[hospitals == h], target, stable_seed(cell, sample, h, "clock"))
                                     for h in np.unique(hospitals)])
    weights = covariate_shift_weights(covariate_matrix(frame, cal, target, frame.iloc[cal]),
                                      covariate_matrix(frame, network.day_one, target, frame.iloc[cal]))
    result = dtaci_stream_widths(
        calibration_scores=cal_scores, calibration_weights=weights, **network.common(frame, {"care_unit": "hospitalid"}),
        scales=scale(current[network.rows], change[network.rows]), predictions=corrected[network.rows].to_numpy(),
        observed=future[network.rows], config=config,
    )
    return pd.Series(result.widths, index=network.rows)


def audit_group(args: argparse.Namespace, contracts: list[dict]) -> list[dict]:
    _, point_config = frozen_point_config(args.point_dev)
    _, dtaci_config = frozen_dtaci_config(args.dtaci_dev, args.dtaci_prior)
    networks = {"network_0.5": replace(dtaci_config, unit_ratio=0.5), "network_1.0": replace(dtaci_config, unit_ratio=1.0)}
    recipes = json.loads(args.recipes.read_text(encoding="utf-8"))["recipes"]
    horizon = int(contracts[0]["horizon_hours"])
    mimic_raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
    cells = []
    for contract in contracts:
        cell, target = contract["cell"], contract["target"]
        site = choose_eicu_cohort(args.eicu_root, target=target, horizon=horizon, max_rows=TEST_ROWS)[0]
        if evaluability(DATABASE, site) is not None:
            continue
        full = normalize_cohort(mimic_raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)[0]
        recipe, _ = deployable_recipe(recipes[cell], site, target)
        cell_point = PointCorrectionConfig() if recipe == "persistence" else point_config
        passes = {name: {h: [] for h in "AB"} for name in CANDIDATES}
        coverage = {name: {h: [] for h in "AB"} for name in CANDIDATES}
        for sample in range(args.samples):
            train = mimic_sample(full, target=target, horizon=horizon, sample=sample, rows=SOURCE_ROWS)
            frame = pd.concat([train, site], ignore_index=True, sort=False).sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
            subject = frame["subject_id"].astype(str)
            fitted = fit_continuous_recipe(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                                           np.flatnonzero(subject.str.startswith(PREFIX[DATABASE])),
                                           target=target, source=recipe, seed=stable_seed(cell, "mimic_to_eicu"))
            rows = fitted["test_rows"]
            current = simulate_sites(frame, rows, fitted, target, horizon, cell, sample, cell_point, dtaci_config,
                                     labels=(DTACI_METHOD,))[DTACI_METHOD]
            corrected = pd.Series(current["prediction"], index=rows)
            widths = {"current": pd.Series(current["width"], index=rows)}
            for name, config in networks.items():
                widths[name] = network_widths(frame, rows, fitted, corrected, target, horizon, cell, sample, config)
            rows = rows[current["scored"]]                      # warm-up stays are never scored
            observed = _safe_float(frame, f"future_{target}")[rows]
            loss = np.abs(corrected[rows].to_numpy() - observed)
            halves = np.array([half(h) for h in frame.iloc[rows]["hospitalid"].astype(str)])
            for name in CANDIDATES:
                width = widths[name][rows].to_numpy()
                for h in "AB":
                    mask = halves == h
                    report = interval_report(frame, rows[mask], (loss <= width)[mask], width[mask], GATE_AXES[DATABASE]["interval"])
                    passes[name][h].append(report["pass"])
                    coverage[name][h].append([report[axis] for axis in report["gate_axes"]])
        entry = {"cell": cell, "recipe": recipe, **{name: {h: {
            "interval_pass_share": float(np.mean(passes[name][h])),
            "interval_robust": bool(np.mean(passes[name][h]) >= ROBUST_SHARE),
            "mean_gate_coverage": np.mean(coverage[name][h], axis=0).round(4).tolist(),
        } for h in "AB"} for name in CANDIDATES}}
        print(json.dumps({"cell": cell, **{n: [entry[n][h]["interval_pass_share"] for h in "AB"] for n in CANDIDATES}}), flush=True)
        cells.append(entry)
    return cells


def robust_count(cells: list[dict], name: str, side: str) -> int:
    return sum(c[name][side]["interval_robust"] for c in cells)


def choose(cells: list[dict], side: str) -> str:
    return max(CANDIDATES, key=lambda name: (robust_count(cells, name, side), -CANDIDATES.index(name)))


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = sorted((r for r in development["cells"] if r["task_type"] != "binary_event_probability"),
                       key=lambda r: (int(r["horizon_hours"]), r["target"]))
    if args.cells:
        contracts = [r for r in contracts if r["cell"] in set(args.cells.split(","))]
    groups = [[r for r in contracts if int(r["horizon_hours"]) == h] for h in sorted({int(r["horizon_hours"]) for r in contracts})]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cells = [cell for group in pool.map(audit_group, [args] * len(groups), groups) for cell in group]
    folds = {}
    for select_on, score_on in (("A", "B"), ("B", "A")):
        chosen = choose(cells, select_on)
        folds[f"select_{select_on}_score_{score_on}"] = {
            "chosen": chosen, "held_out_robust": robust_count(cells, chosen, score_on),
            "current_robust": robust_count(cells, "current", score_on)}
    held_out = sum(f["held_out_robust"] for f in folds.values())
    baseline = sum(f["current_robust"] for f in folds.values())
    adopted = held_out > baseline
    totals = {name: {side: robust_count(cells, name, side) for side in "AB"} for name in CANDIDATES}
    output = {
        "schema": "site_adaptive_network_dev.v2",
        "database": DATABASE,
        "candidates": list(CANDIDATES),
        "robust_counts_by_half": totals,
        "folds": folds,
        "held_out_total": held_out,
        "current_total": baseline,
        "adopted": bool(adopted),
        "selected": max(CANDIDATES, key=lambda n: (sum(totals[n].values()), -CANDIDATES.index(n))) if adopted else "current",
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
    parser.add_argument("--point-dev", type=Path, default=Path("site_adaptive_point_dev_20261016.json"))
    parser.add_argument("--dtaci-dev", type=Path, default=Path("site_adaptive_dtaci_dev_20261017.json"))
    parser.add_argument("--dtaci-prior", type=Path, default=Path("site_adaptive_dtaci_prior_dev_20261019.json"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cells", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({k: result[k] for k in ("robust_counts_by_half", "folds", "held_out_total", "current_total", "adopted", "selected")}, indent=1))
