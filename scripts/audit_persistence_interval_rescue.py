"""Test one predeclared conditional interval for point-qualified fallback cells.

No new point-head fitting, external recalibration, nominal-level search, or
external runner-up selection. Patient prediction caches remain outside Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_normalized_intervals import _fit_scale_model, _predict_scale, _serialize_scale_model
from whole_body_jepa.multisource_intervals import SEEDS, coverage, domain_coverage, fit_radius, representative_rows, sha256
from whole_body_jepa.source_selection import load_selector, patient_roles, residual_frame, validate_selected


def scale_features(frame: pd.DataFrame, source: str) -> np.ndarray:
    current = frame["current"].to_numpy(dtype=float)
    prediction = frame[source].to_numpy(dtype=float)
    return np.column_stack([current, np.abs(prediction - current)])


def audit_conditional_interval(development, external, selection, cell):
    baseline = validate_selected(development, external, selection, cell)
    if not all(baseline.get(db, {}).get("point", {}).get("pass", False) for db in ("mimic_test", "nwicu")):
        return {"pass": False, "reason": "point_gate_not_passed", "baseline": baseline}
    roles = patient_roles(development)
    source = selection["selected_source"]
    fit = development.loc[roles["selection"]]
    fit = fit.iloc[representative_rows(fit, SEEDS[0])]
    errors = (fit["truth"] - fit[source]).abs().to_numpy()
    if not np.isfinite(errors).all() or np.sum(errors > 0) < 2:
        return {"pass": False, "reason": "insufficient_positive_scale_errors", "baseline": baseline}
    model = _fit_scale_model(scale_features(fit, source), errors, ridge=10.)
    cal = development.loc[roles["calibration"]]
    normalized = residual_frame(cal, source)
    normalized["residual"] /= _predict_scale(model, scale_features(cal, source))
    quantile = fit_radius(normalized, SEEDS[0])
    results = {}
    for name, frame in (("mimic_test", development.loc[roles["test"]]), ("nwicu", external)):
        scales = _predict_scale(model, scale_features(frame, source))
        residuals = residual_frame(frame, source)
        residuals["residual"] /= scales
        time = pd.to_datetime(frame["t"])
        results[name] = {
            "point": baseline[name]["point"],
            "coverage": coverage(residuals, quantile),
            "careunit_coverage": domain_coverage(residuals, quantile),
            "late_coverage": coverage(residuals.loc[time >= time.quantile(.8)], quantile),
            "mean_width": float(np.mean(2 * quantile * scales)),
            "median_width": float(np.median(2 * quantile * scales)),
        }
    reasons = [f"{db}:{gate}" for db, result in results.items()
               for gate in ("point", "coverage", "careunit_coverage", "late_coverage") if not result[gate]["pass"]]
    return {"pass": not reasons, "failure_reasons": reasons,
            "method": "normalized_current_predicted_delta", "feature_names": ["current", "absolute_predicted_delta"],
            "normalized_radius": quantile, "scale_model": _serialize_scale_model(model),
            "fit_patients": len(fit), "calibration_patients": int(cal["subject_id"].nunique()),
            "nominal_level": .9, "baseline": baseline, **results}


def run(args):
    selector = load_selector(args.selector, args.model_dir)
    rows, not_attempted = {}, {}
    for cell, row in selector["cells"].items():
        if row["can_move"]:
            continue
        validation = row["validation"]
        qualified = all(validation.get(db, {}).get("point", {}).get("pass", False) for db in ("mimic_test", "nwicu"))
        if not qualified:
            not_attempted[cell] = validation["failure_reasons"]
            continue
        stem = cell.replace("@", "_")
        meta = json.loads((args.cache_dir / f"{stem}.json").read_text())
        if meta["key"]["binding"] != selector["release_binding"]:
            raise ValueError(f"stale prediction cache: {cell}")
        development, external = (pd.read_parquet(args.cache_dir / f"{stem}_{db}.parquet") for db in ("mimic", "nwicu"))
        result = audit_conditional_interval(development, external, row["selection"], cell)
        rows[cell] = {"selected_source": row["selected_source"], **result}
        print(json.dumps({"cell": cell, "pass": result["pass"], "failures": result.get("failure_reasons", result.get("reason")),
                          "mimic_coverage": result.get("mimic_test", {}).get("coverage"),
                          "nwicu_coverage": result.get("nwicu", {}).get("coverage")}), flush=True)
    payload = {"schema": "whole_body_jepa.persistence_interval_rescue.v1",
               "parent_selector_sha256": sha256(args.selector), "release_binding": selector["release_binding"],
               "attempted_cells": len(rows), "passed_cells": [c for c, r in rows.items() if r["pass"]],
               "not_attempted": not_attempted, "cells": rows,
               "point_weights_changed": False, "external_labels_used_to_fit_intervals": False,
               "previous_test_results_inspected": True, "new_untouched_lockbox_claim": False,
               "causal_claim_allowed": False, "clinical_promotion_allowed": False}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Persistence interval rescue", "",
             "All 14 fallback cells were inspected. Only the four whose frozen development-selected",
             "source already passed both databases' point gates are eligible for this interval-only test.",
             "The other ten are not relabelled as successful or tested against external runner-ups.", "",
             "One fixed log-residual ridge scale model (penalty 10) uses only current target value and",
             "absolute predicted change. One anchor per selection patient fits the scale; separate",
             "calibration patients determine the exact nominal 90% conformal quantile. Point weights,",
             "source choices, nominal level and coverage bounds [0.87,0.93] stay fixed. External residuals",
             "are never used for fitting. These are reused research validation sets, not a fresh lockbox.", "",
             f"Attempted: {len(rows)}; passed: {len(payload['passed_cells'])}.", "",
             "| Cell | MIMIC row coverage | NWICU row coverage | Pass | Failures |",
             "|---|---:|---:|---|---|"]
    for c, r in rows.items():
        lines.append(f"| {c} | {r.get('mimic_test', {}).get('coverage', {}).get('row_coverage')} | "
                     f"{r.get('nwicu', {}).get('coverage', {}).get('row_coverage')} | {r['pass']} | "
                     f"{', '.join(r.get('failure_reasons', [])) or r.get('reason', '-')} |")
    lines += ["", "## Other fallbacks", ""]
    lines.extend(f"- `{c}`: {', '.join(reason)}" for c, reason in not_attempted.items())
    lines += ["", "Passing this test concerns factual point/interval behavior, not latent identifiability",
              "or clinical safety. Coverage is empirical and not a guarantee for each patient.", ""]
    args.findings.write_text("\n".join(lines))
    print(json.dumps({"attempted": len(rows), "passed": payload["passed_cells"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, default=Path("models/unified_source_selector.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/unified_source_predictions_20260902"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/persistence_interval_rescue_20260902.json"))
    parser.add_argument("--findings", type=Path, default=Path("PERSISTENCE_INTERVAL_RESCUE_FINDINGS_20260902.md"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
