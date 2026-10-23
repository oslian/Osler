"""MIMIC-only development of self-calibrating intervals under simulated deployment shift.

Two shift scenarios inside MIMIC-IV, each treated as deployment to a new site:

- S1 time: fit and calibrate on patients with anchor_year_group 2008-2013;
  deploy to patients from 2017-2022;
- S2 care unit: fit and calibrate outside cardiac units; deploy to CVICU + CCU.

The deployment stream is the target patients replayed on the shared deployment
clock (``replay``): stays arrive as a Poisson process and overlap, and an
outcome resolves once the clock passes anchor + horizon + |label offset|.
Day-one weights use only warm-up inputs received before the first scored stay
arrives (no labels); warm-up stays are not scored.

Candidates: static v2; day-one weighting only; and for tracking step
eta in {0.02, 0.05, 0.1}: site; weighting + site; weighting + site + value
group; weighting + site + care unit; weighting + site + care unit + value
group; weighting + site + care unit + value group + patient.  Levels below the
site use eta / 2.  The care-unit level was added after a one-cell MIMIC smoke
test (potassium@6h) showed a persistent single-unit shortfall that a site term
cannot correct; no external data had been read.

Predeclared selection rule (written before any candidate was scored):
maximize the number of (cell, scenario) streams whose patient-equal and
care-unit-equal coverage both lie in [0.87, 0.93]; ties -> smaller median width
relative to static v2.  The winner is adopted only if it beats static v2.
No eICU or NWICU data is read.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import (
    cap_by_subject,
    fit_continuous_recipe,
    group_equal_coverage,
    normalize_cohort,
    stable_seed,
)
from osler_jepa.deployment_clock import deployment_clock
from osler_jepa.interval_scale import fit_interval_scale
from osler_jepa.patient_split import selection_patients
from osler_jepa.site_adaptive_conformal import AdaptiveConfig, adaptive_stream_widths, covariate_shift_weights
from target_specific_domain_conformal_audit import _safe_float


COVARIATES = (
    "heart_rate", "map", "respiratory_rate", "o2sat", "temperature", "creatinine", "bun", "potassium",
    "sodium", "glucose", "hemoglobin", "hematocrit", "platelets", "wbc", "bicarbonate", "chloride",
)
STREAM_ROWS = 30000
BOUNDS = (0.87, 0.93)
SCENARIOS = {
    "S1_time": ("anchor_year_group", ("2008 - 2010", "2011 - 2013"), ("2017 - 2019", "2020 - 2022")),
    "S2_cardiac_units": ("first_careunit", None, ("Cardiac Vascular Intensive Care Unit (CVICU)", "Coronary Care Unit (CCU)")),
}


def candidates() -> dict[str, tuple[bool, AdaptiveConfig]]:
    output = {"static_v2": (False, AdaptiveConfig(etas={})), "weighting_only": (True, AdaptiveConfig(etas={}))}
    for eta in (0.02, 0.05, 0.1):
        low = eta / 2
        output[f"site_{eta}"] = (False, AdaptiveConfig(etas={"site": eta}))
        output[f"weight_site_{eta}"] = (True, AdaptiveConfig(etas={"site": eta}))
        output[f"weight_site_value_{eta}"] = (True, AdaptiveConfig(etas={"site": eta, "value_group": low}))
        output[f"weight_site_unit_{eta}"] = (True, AdaptiveConfig(etas={"site": eta, "care_unit": low}))
        output[f"weight_site_unit_value_{eta}"] = (
            True, AdaptiveConfig(etas={"site": eta, "care_unit": low, "value_group": low})
        )
        output[f"weight_site_unit_value_patient_{eta}"] = (
            True, AdaptiveConfig(etas={"site": eta, "care_unit": low, "value_group": low, "patient": low})
        )
    return output


def covariate_matrix(frame: pd.DataFrame, rows: np.ndarray, target: str, reference: pd.DataFrame) -> np.ndarray:
    columns = [f"{target}_t", f"{target}_age_hr"] + [f"{var}_t" for var in COVARIATES if var != target]
    output = []
    for column in columns:
        values = _safe_float(frame, column)[rows]
        ref = _safe_float(reference, column)
        ref = ref[np.isfinite(ref)]
        center = float(np.median(ref)) if len(ref) else 0.0
        spread = float(np.subtract(*np.percentile(ref, [75, 25]))) if len(ref) else 1.0
        spread = spread if spread > 0 else 1.0
        observed = np.isfinite(values)
        output.append(np.where(observed, (values - center) / spread, 0.0))
        output.append(observed.astype(float))
    return np.column_stack(output)


@dataclass(frozen=True)
class Replay:
    """One deployment's rows on the shared deployment clock, in clock order."""

    rows: np.ndarray     # frame positions
    clock: np.ndarray    # anchor time on the deployment clock (hours)
    resolve: np.ndarray  # clock time at which the row's outcome becomes available
    warm: np.ndarray     # rows of warm-up stays: never scored

    @property
    def day_one(self) -> np.ndarray:
        """Warm-up rows received before the first scored stay arrives: the only inputs day-one weights may use."""

        cutoff = self.clock[~self.warm].min() if np.any(~self.warm) else np.inf
        return self.rows[self.warm & (self.clock < cutoff)]

    def common(self, frame: pd.DataFrame, levels: dict[str, str]) -> dict:
        """Keyword arguments shared by the online point and interval trackers."""

        return {"stays": frame.iloc[self.rows]["stay_id"].astype(str).to_numpy(),
                "levels": {name: frame.iloc[self.rows][column].astype(str).to_numpy() for name, column in levels.items()},
                "anchor_hours": self.clock, "resolve_hours": self.resolve}


def replay(frame: pd.DataFrame, rows: np.ndarray, target: str, seed: int) -> Replay:
    """Replay ``rows`` as one live deployment (``osler_jepa.deployment_clock``)."""

    availability = _safe_float(frame, f"future_{target}_available_hr")[rows]
    if not np.all(np.isfinite(availability)):
        raise ValueError(f"label availability times missing for {target}: rebuild transitions with label offsets")
    clocked = deployment_clock(frame.iloc[rows]["stay_id"].to_numpy(), _safe_float(frame, "hours_since_onset")[rows], seed=seed)
    order = clocked.order
    clock = clocked.clock_hours[order]
    return Replay(rows=np.asarray(rows)[order], clock=clock, resolve=clock + availability[order], warm=clocked.warmup[order])


def network_replay(frame: pd.DataFrame, sites: list[Replay]) -> Replay:
    """Several deployments that all start at clock 0, merged into one stream on one clock."""

    rows = np.concatenate([site.rows for site in sites])
    clock = np.concatenate([site.clock for site in sites])
    order = np.lexsort((_safe_float(frame, "hours_since_onset")[rows], clock))
    return Replay(rows=rows[order], clock=clock[order], resolve=np.concatenate([site.resolve for site in sites])[order],
                  warm=np.concatenate([site.warm for site in sites])[order])


def pseudo_sites(frame: pd.DataFrame, rows: np.ndarray, rows_per_site: int, seed: int) -> list[np.ndarray]:
    """Split ``rows`` into fresh deployments of about ``rows_per_site`` rows each, never splitting a stay."""

    stays = frame.iloc[rows]["stay_id"].astype(str)
    sizes = stays.value_counts()
    order = np.random.default_rng(seed).permutation(np.sort(sizes.index.to_numpy()))
    site_of, site, count = {}, 0, 0
    for stay in order:
        if count >= rows_per_site:
            site, count = site + 1, 0
        site_of[stay] = site
        count += int(sizes[stay])
    labels = stays.map(site_of).to_numpy()
    return [np.asarray(rows)[labels == label] for label in range(site + 1)]


def evaluate(frame, rows, covered, widths, reference_width) -> dict:
    patient = group_equal_coverage(frame, rows, covered, "subject_id")
    careunit = group_equal_coverage(frame, rows, covered, "careunit")
    return {
        "patient_equal_coverage": patient,
        "careunit_equal_coverage": careunit,
        "pass": bool(BOUNDS[0] <= patient <= BOUNDS[1] and BOUNDS[0] <= careunit <= BOUNDS[1]),
        "median_width_ratio": float(np.median(widths / reference_width)),
    }


def run_cell(full: pd.DataFrame, years: pd.Series, *, cell: str, target: str, horizon: int, source: str) -> dict:
    results = {}
    full = full.assign(anchor_year_group=full["subject_id"].map(years).fillna("unknown"))
    for name, (column, source_values, target_values) in SCENARIOS.items():
        in_target = full[column].isin(target_values).to_numpy()
        in_source = full[column].isin(source_values).to_numpy() if source_values else ~in_target
        source_frame = cap_by_subject(full[in_source], 15000, stable_seed(cell, name, "source"))
        target_frame = cap_by_subject(full[in_target], STREAM_ROWS, stable_seed(cell, name, "target"))
        source_frame = source_frame.assign(subject_id="SRC:" + source_frame["subject_id"].astype(str))
        target_frame = target_frame.assign(subject_id="TGT:" + target_frame["subject_id"].astype(str))
        frame = pd.concat([source_frame, target_frame], ignore_index=True, sort=False)
        frame.sort_values(["subject_id", "anchor_time"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        subject = frame["subject_id"].astype(str)
        train = np.flatnonzero(subject.str.startswith("SRC:"))
        test = np.flatnonzero(subject.str.startswith("TGT:"))
        fitted = fit_continuous_recipe(frame, train, test, target=target, source=source, seed=stable_seed(cell, "mimic_to_eicu"))
        future, current = _safe_float(frame, f"future_{target}"), _safe_float(frame, f"{target}_t")
        held_out = fitted["calibration_rows"]
        residual = np.abs(fitted["calibration_prediction"] - future[held_out])
        choosing = selection_patients(frame.iloc[held_out]["subject_id"].to_numpy())
        sel, cal = held_out[choosing], held_out[~choosing]       # scale on selection patients, scores on conformal patients
        scale = fit_interval_scale("quintile", current[sel], np.full(len(sel), np.nan), residual[choosing],
                                   frame.iloc[sel]["subject_id"].to_numpy())
        cal_scores = residual[~choosing] / scale(current[cal], np.full(len(cal), np.nan))

        prediction = pd.Series(fitted["test_prediction"], index=fitted["test_rows"])
        deployment = replay(frame, fitted["test_rows"], target, stable_seed(cell, name, "clock"))
        rows = deployment.rows
        bins = np.digitize(current[rows], scale.edges)
        scales = scale(current[rows], np.full(len(rows), np.nan))
        weights = covariate_shift_weights(
            covariate_matrix(frame, cal, target, frame.iloc[cal]),
            covariate_matrix(frame, deployment.day_one, target, frame.iloc[cal]),
        )
        stream = {
            **deployment.common(frame, {"care_unit": "careunit", "patient": "subject_id"}),
            "scales": scales,
            "predictions": prediction[rows].to_numpy(),
            "observed": future[rows],
        }
        stream["levels"]["value_group"] = bins
        scored = ~deployment.warm
        errors = np.abs(stream["predictions"] - stream["observed"])
        reference = None
        scenario = {}
        for label, (weighted, config) in candidates().items():
            calibration_weights = weights if weighted else np.ones(len(cal_scores))
            result = adaptive_stream_widths(
                calibration_scores=cal_scores, calibration_weights=calibration_weights, config=config, **stream,
            )
            if reference is None:
                reference = result.widths
            scenario[label] = evaluate(frame, rows[scored], (errors <= result.widths)[scored], result.widths[scored], reference[scored])
        results[name] = {"stream_rows": int(len(rows)), "scored_rows": int(scored.sum()),
                         "stream_patients": int(len(set(stream["levels"]["patient"]))), "candidates": scenario}
    return results


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
            print(f"site-adaptive dev {cell}", flush=True)
            full, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
            full = full.assign(first_careunit=full["careunit"])
            cells.append({"cell": cell, **run_cell(full, years, cell=cell, target=target, horizon=horizon,
                                                     source=contract["source_model"])})
        del raw
    names = list(candidates())
    summary = {}
    for label in names:
        streams = [cell[s]["candidates"][label] for cell in cells for s in SCENARIOS]
        summary[label] = {
            "passing_streams": int(sum(item["pass"] for item in streams)),
            "streams": len(streams),
            "median_patient_coverage": float(np.median([item["patient_equal_coverage"] for item in streams])),
            "median_width_ratio_vs_static": float(np.median([item["median_width_ratio"] for item in streams])),
        }
    ranked = sorted(names, key=lambda label: (-summary[label]["passing_streams"], summary[label]["median_width_ratio_vs_static"]))
    winner = ranked[0]
    adopted = winner if summary[winner]["passing_streams"] > summary["static_v2"]["passing_streams"] else None
    output = {
        "schema": "site_adaptive_interval_dev.v2",
        "database": "MIMIC-IV only",
        "mimic_template": args.mimic_template,
        "selection_rule": "most streams with patient- and care-unit-equal coverage in [0.87, 0.93]; ties -> smaller width; must beat static v2",
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
    print(json.dumps({"selected": result["selected"], "top": result["ranking"][:5],
                      "summary": {k: result["summary"][k] for k in result["ranking"][:5] + ["static_v2"]}}, indent=1))
