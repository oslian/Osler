"""Simulated deployment of the frozen self-calibrating interval at external sites.

Method frozen from ``site_adaptive_interval_dev_20261016.json`` (MIMIC only):
``weight_site_unit_0.02`` = day-one covariate-shift weighting + online log-scale
quantile tracking at site level (eta 0.02) and care-unit level (eta 0.01), on
top of the v2 value-bin normalization and the frozen MIMIC point recipe.  Point
recipes are the external-development selections (``--recipes``, default
``deployment_recipes_20261020.json``: the 47 continuous catalog cells and the
approved point-of-care glucose extension), including persistence fallback cells.

Every external hospital is an independent deployment that starts from MIMIC
calibration: day-one weights use only the inputs of the site's first
min(200, 25%) arriving stays; afterwards the interval adapts as outcomes
resolve.  Stays arrive in a fixed random order (eICU and NWICU carry no usable
cross-patient calendar order), chronological within a stay; an outcome resolves
at anchor + horizon.  Streams use up to 60,000 external rows per cell.

Gates are those of the external audits: point gate (patient axis, plus the
hospital axis on eICU) and interval coverage over the whole stream, patient-
equal plus hospital-equal (eICU) or care-unit-equal (NWICU), in [0.87, 0.93].
Each cell is refit on 5 MIMIC samples; robust = full pass in >= 80% of samples.
Three configurations run on the same streams: static v2; adaptive interval
only; and online point recalibration (frozen from
``site_adaptive_point_dev_20261016.json``, applied to the candidate and to the
recipe's comparator alike) followed by the adaptive interval; and the same
point recalibration followed by DtACI (meta rate frozen from
``site_adaptive_dtaci_dev_20261017.json``; a new site starts from exp(theta0)
times the calibration quantile, theta0 frozen by leave-one-database-out in
``site_adaptive_dtaci_prior_dev_20261019.json``).

When a deployment has more than one hospital (a network, e.g. eICU), the
DtACI interval is network-shared: whole stays from all hospitals arrive
interleaved at random (each hospital's own order kept), the DtACI site level is
learned from every hospital and the unit level is keyed by hospital (step ratio
frozen in ``site_adaptive_network_dev_20261020.json``), so a small hospital
borrows the network's adaptation until it has its own outcomes.  A
single-hospital deployment is unchanged.

Cells whose recipe is ``persistence`` (the selection's safety fallback) output
the current value without point correction and make no point claim; their
full pass is the interval gate alone.

Both external databases were already used: non-independent.  Research-only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from osler_jepa.recipe_availability import deployable_recipe

from external_value_conditional_interval_audit import (
    verify_protocol,
    EXTERNAL_PREFIX,
    GATE_AXES,
    evaluability,
    external_cohort,
    interval_report,
)
from mimic_to_eicu_49_cell_external_audit import fit_continuous_recipe, normalize_cohort, paired_comparison, stable_seed
from multi_sample_external_audit import mimic_sample
from osler_jepa.site_adaptive_conformal import (
    AdaptiveConfig,
    DtaciConfig,
    adaptive_stream_widths,
    covariate_shift_weights,
    dtaci_stream_widths,
)
from osler_jepa.interval_scale import fit_interval_scale
from osler_jepa.patient_split import selection_patients
from osler_jepa.site_adaptive_point import PointCorrectionConfig, online_point_correction
from site_adaptive_interval_dev import covariate_matrix, network_replay, replay
from target_specific_domain_conformal_audit import _safe_float


FROZEN_METHOD = "weight_site_unit_0.02"
FROZEN_CONFIG = AdaptiveConfig(etas={"site": 0.02, "care_unit": 0.01})
STATIC_CONFIG = AdaptiveConfig(etas={})
ROBUST_SHARE = 0.8
POINT_METHOD = "point_and_interval_adaptive"
DTACI_METHOD = "point_and_dtaci_adaptive"
LABELS = ("static_v2", FROZEN_METHOD, POINT_METHOD, DTACI_METHOD)
SCALE_SCHEME = "quintile"


def frozen_dtaci_config(path: Path, prior: Path | None = None) -> tuple[str, DtaciConfig]:
    """DtACI meta rate from ``path``; the new-site starting multiplier from ``prior`` when given."""

    selected = json.loads(path.read_text(encoding="utf-8"))["selected"]
    if not selected.startswith("dtaci_"):
        raise ValueError(f"DtACI was not adopted in {path}: {selected}")
    theta = float(json.loads(prior.read_text(encoding="utf-8"))["selected_theta0"]) if prior else 0.0
    return f"{selected}_theta0_{theta:.2f}", DtaciConfig(meta_rate=float(selected.removeprefix("dtaci_")),
                                                         initial_log_multiplier=theta)


def frozen_network_config(path: Path, dtaci_config: DtaciConfig) -> DtaciConfig | None:
    """Network-shared DtACI config from the network development report, or None when not adopted."""

    report = json.loads(path.read_text(encoding="utf-8"))
    if not report["adopted"]:
        return None
    return replace(dtaci_config, unit_ratio=float(report["selected"].removeprefix("network_")))


def frozen_point_config(path: Path) -> tuple[str, PointCorrectionConfig]:
    from site_adaptive_point_dev import candidates

    selected = json.loads(path.read_text(encoding="utf-8"))["selected"]
    return selected, candidates()[selected]


def simulate_sites(frame, rows, fitted, target, horizon, cell, sample, point_config, dtaci_config,
                   dtaci_variants=None, scale_scheme=SCALE_SCHEME, labels=LABELS,
                   network_config=None) -> dict[str, dict[str, np.ndarray]]:
    """Replay each external hospital as a live, concurrent deployment; return predictions, comparator, widths and the scored mask.

    Held-out MIMIC patients are split (``osler_jepa.patient_split``): selection
    patients fit the point and interval scales, conformal patients give the
    calibration scores.  Stays arrive on a shared deployment clock (``osler_jepa.deployment_clock``),
    overlap in time, and an outcome updates the online point correction and the
    interval trackers only once the clock passes anchor + horizon + |label offset|
    (``future_<target>_available_hr`` from ``normalize_cohort``).  Day-one weights
    use only warm-up inputs received before the first scored stay arrives; warm-up
    stays are not scored (``scored`` = False).  ``dtaci_variants`` adds DtACI runs on
    the same corrected predictions; ``network_config`` (multi-hospital deployments
    only) replaces the DtACI widths with one network-shared stream on the same clock.
    """

    dtaci_variants = dict(dtaci_variants or {})
    labels = tuple(labels) + tuple(dtaci_variants)

    future, current = _safe_float(frame, f"future_{target}"), _safe_float(frame, f"{target}_t")
    change = np.abs(current - _safe_float(frame, f"lag1__{target}_t"))
    held_out = fitted["calibration_rows"]
    held_out_residual = np.abs(fitted["calibration_prediction"] - future[held_out])
    choosing = selection_patients(frame.iloc[held_out]["subject_id"].to_numpy())
    sel, cal = held_out[choosing], held_out[~choosing]          # scales are fitted on selection patients only
    point_scale = float(np.median(held_out_residual[choosing]))
    interval_scale = fit_interval_scale(scale_scheme, current[sel], change[sel], held_out_residual[choosing],
                                        frame.iloc[sel]["subject_id"].to_numpy())
    cal_scores = held_out_residual[~choosing] / interval_scale(current[cal], change[cal])
    cal_covariates = covariate_matrix(frame, cal, target, frame.iloc[cal])

    def day_one_weights(deployment):
        return covariate_shift_weights(cal_covariates, covariate_matrix(frame, deployment.day_one, target, frame.iloc[cal]))

    raw_prediction = pd.Series(fitted["test_prediction"], index=fitted["test_rows"])
    raw_comparator = pd.Series(np.asarray(fitted["comparator"], dtype=float), index=fitted["test_rows"])
    persistence_comparator = fitted["comparator_name"] == "persistence_matched_baseline"
    output = {label: {key: pd.Series(np.nan, index=rows) for key in ("prediction", "comparator", "width")} for label in labels}
    warm_of = pd.Series(False, index=rows)
    hospitals = frame.iloc[rows]["hospitalid"].astype(str).to_numpy()
    sites = []
    for hospital in np.unique(hospitals):
        deployment = replay(frame, rows[hospitals == hospital], target, stable_seed(cell, sample, hospital, "clock"))
        sites.append(deployment)
        site_rows = deployment.rows
        warm_of[site_rows] = deployment.warm
        weights = day_one_weights(deployment)
        common = deployment.common(frame, {"care_unit": "careunit"})
        scales = interval_scale(current[site_rows], change[site_rows])
        observed, now = future[site_rows], current[site_rows]
        candidate, comparator = raw_prediction[site_rows].to_numpy(), raw_comparator[site_rows].to_numpy()
        corrected = online_point_correction(current=now, prediction=candidate, observed=observed,
                                            scale=point_scale, config=point_config, **common)
        corrected_comparator = now if persistence_comparator else online_point_correction(
            current=now, prediction=comparator, observed=observed, scale=point_scale, config=point_config, **common)
        settings = (
            ("static_v2", STATIC_CONFIG, np.ones(len(cal_scores)), candidate, comparator),
            (FROZEN_METHOD, FROZEN_CONFIG, weights, candidate, comparator),
            (POINT_METHOD, FROZEN_CONFIG, weights, corrected, corrected_comparator),
        )
        settings = settings + ((DTACI_METHOD, dtaci_config, weights, corrected, corrected_comparator),) + tuple(
            (label, config, weights, corrected, corrected_comparator) for label, config in dtaci_variants.items())
        for label, config, calibration_weights, prediction, comparison in settings:
            if label not in labels:
                continue
            tracker = dtaci_stream_widths if isinstance(config, DtaciConfig) else adaptive_stream_widths
            result = tracker(calibration_scores=cal_scores, calibration_weights=calibration_weights,
                             scales=scales, predictions=prediction, observed=observed, config=config, **common)
            output[label]["prediction"][site_rows] = prediction
            output[label]["comparator"][site_rows] = comparison
            output[label]["width"][site_rows] = result.widths
    if network_config is not None and DTACI_METHOD in labels and len(sites) > 1:
        network = network_replay(frame, sites)                 # every hospital starts deploying at clock 0
        result = dtaci_stream_widths(
            calibration_scores=cal_scores, calibration_weights=day_one_weights(network),
            **network.common(frame, {"care_unit": "hospitalid"}),
            scales=interval_scale(current[network.rows], change[network.rows]),
            predictions=output[DTACI_METHOD]["prediction"][network.rows].to_numpy(),
            observed=future[network.rows], config=network_config,
        )
        output[DTACI_METHOD]["width"][network.rows] = result.widths
    scored = ~warm_of[rows].to_numpy()
    return {label: {**{key: series[rows].to_numpy() for key, series in values.items()}, "scored": scored}
            for label, values in output.items()}


def scored_only(rows: np.ndarray, values: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Drop warm-up rows from one ``simulate_sites`` label's output: warm-up stays are never scored."""

    keep = values["scored"]
    return rows[keep], {key: value[keep] for key, value in values.items() if key != "scored"}


LOCKBOX_DATABASES = frozenset({"SICdb", "HiRID"})  # Zigong (2026-10-18) and Emory (2026-10-20) lockboxes were scored once; now development data


def run(args: argparse.Namespace) -> dict:
    if args.database in LOCKBOX_DATABASES:
        if args.protocol is None:
            raise ValueError(f"{args.database} is an untouched lockbox: --protocol is required")
        verify_protocol(args.protocol)
    catalog = [row for path in args.development_report for row in json.loads(path.read_text(encoding="utf-8"))["cells"]]
    contracts = sorted(
        (row for row in catalog if row["task_type"] != "binary_event_probability"),
        key=lambda row: (int(row["horizon_hours"]), row["target"]),
    )
    if args.cells:
        contracts = [row for row in contracts if row["cell"] in set(args.cells.split(","))]
    axes = GATE_AXES[args.database]
    point_name, point_config = frozen_point_config(args.point_dev)
    dtaci_name, dtaci_config = frozen_dtaci_config(args.dtaci_dev, args.dtaci_prior)
    network_config = frozen_network_config(args.network_dev, dtaci_config)
    recipes = json.loads(args.recipes.read_text(encoding="utf-8"))["recipes"]
    cells, current_horizon, raw = [], None, None
    for contract in contracts:
        cell, target, horizon = contract["cell"], contract["target"], int(contract["horizon_hours"])
        source = recipes[cell]
        if horizon != current_horizon:
            raw, current_horizon = pd.read_parquet(args.mimic_template.format(horizon=horizon)), horizon
        external, _ = external_cohort(args, target, horizon)
        source, recipe_reason = deployable_recipe(source, external, target)
        reason = evaluability(args.database, external)
        if reason:
            cells.append({"cell": cell, "evaluable": False, "reason": reason})
            continue
        print(f"{args.database} site-adaptive {cell}", flush=True)
        full, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
        samples = []
        for sample in range(args.samples):
            mimic = mimic_sample(full, target=target, horizon=horizon, sample=sample, rows=15000)
            frame = pd.concat([mimic, external], ignore_index=True, sort=False)
            frame.sort_values(["subject_id", "anchor_time"], inplace=True)
            frame.reset_index(drop=True, inplace=True)
            subject = frame["subject_id"].astype(str)
            train = np.flatnonzero(subject.str.startswith("MIMIC:"))
            test = np.flatnonzero(subject.str.startswith(EXTERNAL_PREFIX[args.database]))
            seed = stable_seed(cell, "mimic_to_eicu")
            fitted = fit_continuous_recipe(frame, train, test, target=target, source=source, seed=seed)
            rows = fitted["test_rows"]
            observed = _safe_float(frame, f"future_{target}")[rows]
            current = _safe_float(frame, f"{target}_t")[rows]
            # A persistence-fallback cell outputs the current value unchanged: no online point correction.
            cell_point_config = PointCorrectionConfig() if source == "persistence" else point_config
            simulated = simulate_sites(frame, rows, fitted, target, horizon, cell, sample, cell_point_config, dtaci_config,
                                       network_config=network_config)
            entry = {"sample": sample}
            for label, values in simulated.items():
                keep = values["scored"]            # warm-up stays only fit day-one weights; they are never scored
                scored_rows = rows[keep]
                loss = np.abs(values["prediction"] - observed)[keep]
                persistence_loss = np.abs(current - observed)[keep]
                comparisons = {
                    "persistence": paired_comparison(frame, scored_rows, loss, persistence_loss, seed=seed),
                    fitted["comparator_name"]: paired_comparison(frame, scored_rows, loss, np.abs(values["comparator"] - observed)[keep], seed=seed + 2),
                }
                # Persistence fallback makes no point claim; only its interval is gated.
                point = None if source == "persistence" else bool(
                    all(report[axis]["pass"] for report in comparisons.values() for axis in axes["point"]))
                report = interval_report(frame, scored_rows, loss <= values["width"][keep], values["width"][keep], axes["interval"])
                entry[label] = {
                    "point_pass": point,
                    "point_ci": {name: {axis: [report[axis]["bootstrap_low"], report[axis]["bootstrap_high"]] for axis in axes["point"]}
                                 for name, report in comparisons.items()},
                    "mae": float(loss.mean()),
                    "persistence_mae": float(persistence_loss.mean()),
                    "scored_rows": int(keep.sum()),
                    "patient_equal_coverage": report["patient_equal_coverage"],
                    "group_equal_coverage": report["hospital_equal_coverage" if args.database == "eICU" else "careunit_equal_coverage"],
                    "interval_pass": report["pass"],
                    "full_pass": bool(point is not False and report["pass"]),
                    "median_half_width": report["median_half_width"],
                }
            samples.append(entry)
        summary = {
            label: {
                "full_pass_share": float(np.mean([s[label]["full_pass"] for s in samples])),
                "point_pass_share": None if source == "persistence" else float(np.mean([s[label]["point_pass"] for s in samples])),
                "interval_pass_share": float(np.mean([s[label]["interval_pass"] for s in samples])),
                "patient_coverage_range": [min(s[label]["patient_equal_coverage"] for s in samples),
                                           max(s[label]["patient_equal_coverage"] for s in samples)],
                "group_coverage_range": [min(s[label]["group_equal_coverage"] for s in samples),
                                         max(s[label]["group_equal_coverage"] for s in samples)],
                "robust": bool(np.mean([s[label]["full_pass"] for s in samples]) >= ROBUST_SHARE),
            }
            for label in LABELS
        }
        cells.append({"cell": cell, "evaluable": True, "recipe": source, "rows": int(len(rows)),
                      "hospitals": int(frame.iloc[rows]["hospitalid"].nunique()),
                      "methods": summary, "per_sample": samples})
        print(json.dumps({"cell": cell, **{label: summary[label]["full_pass_share"] for label in summary}}), flush=True)
    evaluable = [cell for cell in cells if cell["evaluable"]]
    output = {
        "schema": "site_adaptive_external_audit.v3",
        "external_database": args.database,
        "frozen_method": FROZEN_METHOD,
        "mimic_template": args.mimic_template,
        "samples": args.samples,
        "external_rows_per_cell": args.max_rows_per_database,
        "frozen_point_method": point_name,
        "frozen_dtaci_method": dtaci_name,
        "recipe_source": str(args.recipes),
        "robust_counts": {label: sum(cell["methods"][label]["robust"] for cell in evaluable) for label in LABELS},
        "robust_point_counts": {label: sum((cell["methods"][label]["point_pass_share"] or 0.0) >= ROBUST_SHARE for cell in evaluable) for label in LABELS},
        "persistence_fallback_cells": sorted(cell["cell"] for cell in evaluable if cell["recipe"] == "persistence"),
        "evaluable_continuous_cells": len(evaluable),
        "independent_validation": False,
        "cells": cells,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=sorted(GATE_AXES), required=True)
    parser.add_argument("--development-report", type=Path, nargs="+",
                        default=[Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"),
                                 Path("glucose_poc_catalog_extension_20261020.json")],
                        help="cell catalog files: the 49-cell catalog plus approved extensions")
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_harmonized_20261022"))
    parser.add_argument("--external-template", default=None, help="parquet path template with {horizon} for single-site databases")
    parser.add_argument("--max-rows-per-database", type=int, default=60000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--point-dev", type=Path, default=Path("site_adaptive_point_dev_20261016.json"))
    parser.add_argument("--dtaci-dev", type=Path, default=Path("site_adaptive_dtaci_dev_20261017.json"))
    parser.add_argument("--dtaci-prior", type=Path, default=Path("site_adaptive_dtaci_prior_dev_20261019.json"))
    parser.add_argument("--network-dev", type=Path, default=Path("site_adaptive_network_dev_20261020.json"))
    parser.add_argument("--recipes", type=Path, default=Path("deployment_recipes_20261020.json"),
                        help="JSON with a 'recipes' map cell -> recipe (default: the frozen external-development selections)")
    parser.add_argument("--cells", default="")
    parser.add_argument("--protocol", type=Path, default=None, help="frozen lockbox protocol; required for lockbox databases (SICdb, HiRID, Emory)")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args())["robust_counts"]))
