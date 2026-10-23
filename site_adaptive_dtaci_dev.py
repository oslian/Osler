"""MIMIC-only selection between the frozen fixed-step interval tracker and DtACI.

Deployment form under test: frozen MIMIC recipe + online point recalibration
(``affine_0.005``) + day-one weighting + an online interval tracker.  Interval
candidates: the frozen fixed-step tracker (site 0.02, care unit 0.01) and DtACI
over step-size experts {0.005, 0.01, 0.02, 0.05, 0.1, 0.2} with meta learning
rate in {1, 5, 20} and fixed share 0.005.

Streams: the MIMIC S1 (time) and S2 (cardiac units) deployments on the shared
deployment clock, each in two variants: ``long`` (the whole stream is one site)
and ``short`` (the stays are split into random pseudo-sites of about 300 rows,
each a fresh deployment on its own clock with its own warm-up, day-one weights
and point correction).  Warm-up stays are not scored.  300 rows matches the
typical external hospital stream length (stream metadata only; no external
labels).

Predeclared selection rule (written before any candidate was scored): maximize
passing streams (patient-axis point gate against persistence and the corrected
comparator, plus patient- and care-unit-equal coverage in [0.87, 0.93]) over
47 cells x 2 scenarios x 2 variants; ties -> narrower median width relative to
the frozen tracker; DtACI is adopted only with strictly more passing streams
than the frozen tracker.  No eICU or NWICU data is read.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import group_equal_coverage, normalize_cohort, paired_comparison, stable_seed
from osler_jepa.site_adaptive_conformal import DtaciConfig, adaptive_stream_widths, dtaci_stream_widths
from osler_jepa.site_adaptive_point import online_point_correction
from site_adaptive_external_audit import FROZEN_CONFIG
from site_adaptive_interval_dev import BOUNDS, pseudo_sites, replay
from site_adaptive_point_dev import SCENARIOS, candidates as point_candidates, deployment_view, scenario_streams


POINT_CONFIG = point_candidates()["affine_0.005"]
SHORT_ROWS = 300
CANDIDATES = {"frozen_fixed_step": None, **{f"dtaci_{rate}": DtaciConfig(meta_rate=rate) for rate in (1.0, 5.0, 20.0)}}


def run_variant(stream: dict, rows_per_site: int | None) -> dict:
    frame, target = stream["frame"], stream["target"]
    if rows_per_site is None:
        sites = [stream]
    else:
        parts = pseudo_sites(frame, stream["test_rows"], rows_per_site, stable_seed(stream["seed"], "pseudo_sites"))
        sites = [deployment_view(stream, replay(frame, part, target, stable_seed(stream["seed"], "pseudo_site", k)))
                 for k, part in enumerate(parts)]
    pieces = {key: [] for key in ("rows", "observed", "now", "corrected", "comparator", *CANDIDATES)}
    for site in sites:
        scored, observed, now = site["scored"], site["observed"], site["now"]
        corrected = online_point_correction(current=now, prediction=site["candidate"], observed=observed,
                                            scale=site["point_scale"], config=POINT_CONFIG, **site["common"])
        comparator = now if site["persistence_comparator"] else online_point_correction(
            current=now, prediction=site["comparator"], observed=observed, scale=site["point_scale"], config=POINT_CONFIG,
            **site["common"])
        inputs = dict(calibration_scores=site["cal_scores"], calibration_weights=site["weights"], scales=site["scales"],
                      predictions=corrected, observed=observed, **site["common"])
        for label, config in CANDIDATES.items():
            result = (adaptive_stream_widths(config=FROZEN_CONFIG, **inputs) if config is None
                      else dtaci_stream_widths(config=config, **inputs))
            pieces[label].append(result.widths[scored])
        for key, value in (("rows", site["rows"]), ("observed", observed), ("now", now), ("corrected", corrected),
                           ("comparator", comparator)):
            pieces[key].append(value[scored])
    rows, observed, now, corrected, comparator = (np.concatenate(pieces[k]) for k in ("rows", "observed", "now", "corrected", "comparator"))
    widths = {label: np.concatenate(pieces[label]) for label in CANDIDATES}
    loss = np.abs(corrected - observed)
    point = all(
        paired_comparison(frame, rows, loss, other, seed=stream["seed"])["patient"]["pass"]
        for other in (np.abs(now - observed), np.abs(comparator - observed))
    )
    output = {}
    for label, width in widths.items():
        covered = loss <= width
        patient = group_equal_coverage(frame, rows, covered, "subject_id")
        unit = group_equal_coverage(frame, rows, covered, "careunit")
        output[label] = {
            "point_pass": bool(point),
            "patient_equal_coverage": patient,
            "careunit_equal_coverage": unit,
            "full_pass": bool(point and BOUNDS[0] <= patient <= BOUNDS[1] and BOUNDS[0] <= unit <= BOUNDS[1]),
            "median_width_ratio": float(np.median(width / widths["frozen_fixed_step"])),
        }
    return output


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = [row for row in development["cells"] if row["task_type"] != "binary_event_probability"]
    patients = pd.read_csv(args.patients, usecols=["subject_id", "anchor_year_group"])
    years = pd.Series(patients["anchor_year_group"].to_numpy(), index="MIMIC:" + patients["subject_id"].astype(str))
    by_horizon: dict[int, list[dict]] = {}
    for row in contracts:
        by_horizon.setdefault(int(row["horizon_hours"]), []).append(row)
    cells = []
    for horizon in sorted(by_horizon):
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        for contract in sorted(by_horizon[horizon], key=lambda row: row["target"]):
            cell, target = contract["cell"], contract["target"]
            print(f"dtaci dev {cell}", flush=True)
            full, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
            full = full.assign(first_careunit=full["careunit"])
            entry = {"cell": cell}
            for name, stream in scenario_streams(full, years, cell=cell, target=target, horizon=horizon,
                                                 source=contract["source_model"]):
                entry[name] = {variant: run_variant(stream, size) for variant, size in (("long", None), ("short", SHORT_ROWS))}
            cells.append(entry)
        del raw
    summary = {}
    for label in CANDIDATES:
        streams = [cell[s][v][label] for cell in cells for s in SCENARIOS for v in ("long", "short")]
        summary[label] = {
            "passing_streams": int(sum(item["full_pass"] for item in streams)),
            "passing_short_streams": int(sum(cell[s]["short"][label]["full_pass"] for cell in cells for s in SCENARIOS)),
            "streams": len(streams),
            "median_width_ratio": float(np.median([item["median_width_ratio"] for item in streams])),
        }
    ranked = sorted(CANDIDATES, key=lambda label: (-summary[label]["passing_streams"], summary[label]["median_width_ratio"]))
    winner = ranked[0]
    adopted = winner if winner != "frozen_fixed_step" and summary[winner]["passing_streams"] > summary["frozen_fixed_step"]["passing_streams"] else "frozen_fixed_step"
    output = {
        "schema": "site_adaptive_dtaci_dev.v2",
        "database": "MIMIC-IV only",
        "selection_rule": "most passing streams; ties -> narrower width; DtACI adopted only with strictly more passes than the frozen tracker",
        "summary": summary,
        "ranking": ranked,
        "selected": adopted,
        "cells": cells,
        "external_data_read": False,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", type=Path, default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"))
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--patients", type=Path, default=Path.home() / "Downloads/mimic-iv-3.1/hosp/patients.csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"selected": result["selected"], "summary": result["summary"]}, indent=1))
