#!/usr/bin/env python3
"""Route kidney forecasts by predicted distributional risk, not realized benefit labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_release_masked_cells import donor_indices
from scripts.audit_state_conditioned_kidney import (
    BOUNDS,
    TARGETS,
    comparison_gate,
    coverage_gate,
    patient_bootstrap,
    positions,
    prepare_external,
    prepare_source,
)
from whole_body_jepa.multisource_intervals import conformal_radius, sha256
from whole_body_jepa.runtime import load_release
from whole_body_jepa.state_conditioned import (
    DYNAMIC_THRESHOLDS,
    distribution_risk_samples,
    empirical_crps,
    fit_distribution_expert,
    predict_distribution_risk_conditioned,
)


MARGIN_QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)


def target_delta(frame: pd.DataFrame, rows: np.ndarray, target: str) -> np.ndarray:
    selected = frame.iloc[rows]
    return (
        selected[f"future_{target}"].to_numpy(dtype=float)
        - selected[f"{target}_t"].to_numpy(dtype=float)
    )


def _patient_mae(frame: pd.DataFrame, error: np.ndarray) -> float:
    return float(pd.Series(error, index=frame["subject_id"]).groupby(level=0).mean().mean())


def tune(
    frame: pd.DataFrame,
    rows: np.ndarray,
    features: np.ndarray,
    target: str,
    head: dict,
) -> tuple[float, float, list[dict]]:
    selected = frame.iloc[rows].reset_index(drop=True)
    current = selected[f"{target}_t"].to_numpy(dtype=float)
    truth = selected[f"future_{target}"].to_numpy(dtype=float)
    delta = truth - current
    base = predict_distribution_risk_conditioned(head, features, risk_margin=-np.inf)
    margins = np.unique(np.quantile(base["relative_risk_advantage"], MARGIN_QUANTILES))
    dynamic = np.abs(delta) >= DYNAMIC_THRESHOLDS[target]
    choices = []
    trace = []
    for margin in margins:
        prediction = predict_distribution_risk_conditioned(head, features, risk_margin=float(margin))
        error = np.abs(delta - prediction["point_delta"])
        patient_mae = _patient_mae(selected, error)
        persistence_mae = _patient_mae(selected, np.abs(delta))
        stable_harm = float(error[~dynamic].mean() - np.abs(delta[~dynamic]).mean())
        eligible = patient_mae < persistence_mae and stable_harm <= 0.01 * DYNAMIC_THRESHOLDS[target]
        row = {
            "margin": float(margin),
            "patient_mae": patient_mae,
            "persistence_patient_mae": persistence_mae,
            "stable_harm": stable_harm,
            "move_fraction": float(prediction["move"].mean()),
            "eligible": bool(eligible),
        }
        trace.append(row)
        choices.append((not eligible, patient_mae, -row["move_fraction"], float(margin)))
    margin = min(choices)[3]
    prediction = predict_distribution_risk_conditioned(head, features, risk_margin=margin)
    samples = distribution_risk_samples(prediction, head["persistence_delta_quantiles"])
    center = current + np.quantile(samples, 0.50, axis=1)
    low = current + np.quantile(samples, 0.05, axis=1)
    high = current + np.quantile(samples, 0.95, axis=1)
    lower_width = np.maximum(center - low, 1e-6)
    upper_width = np.maximum(high - center, 1e-6)
    nonconformity = np.maximum((center - truth) / lower_width, (truth - center) / upper_width)
    return margin, conformal_radius(np.maximum(nonconformity, 0.0)), trace


def evaluate(
    frame: pd.DataFrame,
    rows: np.ndarray,
    features: np.ndarray,
    target: str,
    head: dict,
    margin: float,
    interval_scale: float,
    role: str,
    seed: int,
) -> dict:
    selected = frame.iloc[rows].reset_index(drop=True)
    current = selected[f"{target}_t"].to_numpy(dtype=float)
    truth = selected[f"future_{target}"].to_numpy(dtype=float)
    delta = truth - current
    prediction = predict_distribution_risk_conditioned(head, features, risk_margin=margin)
    point = current + prediction["point_delta"]
    plain = current + prediction["expert_delta"]

    donor = donor_indices(selected["subject_id"], seed)
    placebo_move = prediction["relative_risk_advantage"][donor] > margin
    placebo_point = current + np.where(placebo_move, prediction["expert_delta"], 0.0)

    samples = distribution_risk_samples(prediction, head["persistence_delta_quantiles"])
    persistence_samples = np.tile(head["persistence_delta_quantiles"], (len(rows), 1))
    candidate_crps = empirical_crps(samples, delta)
    persistence_crps = empirical_crps(persistence_samples, delta)
    center = current + np.quantile(samples, 0.50, axis=1)
    base_lower = current + np.quantile(samples, 0.05, axis=1)
    base_upper = current + np.quantile(samples, 0.95, axis=1)
    lower = center - interval_scale * (center - base_lower)
    upper = center + interval_scale * (base_upper - center)

    realized_benefit = np.abs(delta) - np.abs(delta - prediction["expert_delta"])
    beneficial = realized_benefit > 0.0
    moved = prediction["move"]
    dynamic = np.abs(delta) >= DYNAMIC_THRESHOLDS[target]
    comparisons = {
        "persistence": comparison_gate(selected, point, current, truth, f"risk:{target}:{role}:persistence"),
        "plain_equal_input_head": comparison_gate(selected, point, plain, truth, f"risk:{target}:{role}:plain"),
        "shuffled_risk_routing": comparison_gate(
            selected, point, placebo_point, truth, f"risk:{target}:{role}:placebo"
        ),
    }
    crps_gate = patient_bootstrap(
        selected, candidate_crps - persistence_crps, f"risk:{target}:{role}:crps"
    )
    coverage = coverage_gate(selected, truth, lower, upper)
    result = {
        "rows": int(len(selected)),
        "patients": int(selected["subject_id"].nunique()),
        "move_fraction": float(moved.mean()),
        "move_precision": float(beneficial[moved].mean()) if moved.any() else None,
        "stable_false_move_fraction": float(moved[~dynamic].mean()) if (~dynamic).any() else None,
        "mean_predicted_risk_advantage": float(prediction["risk_advantage"].mean()),
        "mean_relative_risk_advantage": float(prediction["relative_risk_advantage"].mean()),
        "mae": {
            "distribution_risk": float(np.abs(point - truth).mean()),
            "persistence": float(np.abs(current - truth).mean()),
            "plain_equal_input_head": float(np.abs(plain - truth).mean()),
            "shuffled_risk_routing": float(np.abs(placebo_point - truth).mean()),
            "stable_distribution_risk": float(np.abs(point[~dynamic] - truth[~dynamic]).mean()),
            "stable_persistence": float(np.abs(current[~dynamic] - truth[~dynamic]).mean()),
            "dynamic_distribution_risk": float(np.abs(point[dynamic] - truth[dynamic]).mean()),
            "dynamic_persistence": float(np.abs(current[dynamic] - truth[dynamic]).mean()),
        },
        "crps": {
            "distribution_risk": float(candidate_crps.mean()),
            "empirical_persistence": float(persistence_crps.mean()),
            "paired": crps_gate,
        },
        "comparisons": comparisons,
        "coverage": coverage,
    }
    result["full_gate_pass"] = bool(
        all(item["pass"] for item in comparisons.values())
        and crps_gate["pass"]
        and coverage["pass"]
    )
    return result


def write_findings(report: dict, path: Path) -> None:
    lines = [
        "# Distribution-Risk Kidney Router Audit",
        "",
        "## Result",
        "",
        "This audit does not learn a noisy realized-benefit label. The all-outcome target",
        "expert predicts q05/q50/q95 of future delta. Persistence and the expert are then",
        "compared by expected absolute loss under that conditional distribution.",
        "All margins and interval scales are selected on eICU only.",
        "",
        "| Target | Risk margin | eICU patient | eICU hospital | MIMIC-IV | Promotion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target, item in report["targets"].items():
        evaluation = item["evaluation"]
        lines.append(
            f"| {target}@{report['horizon_hours']}h | {item['risk_margin']:.6g} | "
            f"{evaluation['eicu_patient']['full_gate_pass']} | "
            f"{evaluation['eicu_hospital']['full_gate_pass']} | "
            f"{evaluation['mimic_external']['full_gate_pass']} | {item['promotion_allowed']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "This is a factual candidate audit. MIMIC-IV has been inspected and is not an",
        "untouched lockbox. Failure leaves the release and persistence fallback unchanged.",
        "No causal or clinical-decision claim is allowed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    report = {
        "schema": "whole_body_jepa.distribution_risk_kidney_audit.v1",
        "horizon_hours": args.horizon,
        "source_database": "eICU",
        "external_database": "MIMIC-IV",
        "external_is_untouched_lockbox": False,
        "release_binding": {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()},
        "routing_target": "relative_predicted_expected_absolute_loss",
        "coverage_bounds": list(BOUNDS),
        "targets": {},
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
    }
    print("prepare eICU source features", flush=True)
    source_frame, source_rows, source_features, source_lookup = prepare_source(args, assets)
    bundles = {}
    for target in TARGETS:
        print(f"fit distribution expert: {target}", flush=True)
        rows = source_rows[target]
        train_pos = positions(rows["train"], source_lookup)
        tune_pos = positions(rows["tune"], source_lookup)
        head = fit_distribution_expert(
            source_features[target][train_pos],
            target_delta(source_frame, rows["train"], target),
            seed=args.seed,
        )
        margin, interval_scale, tuning_trace = tune(
            source_frame, rows["tune"], source_features[target][tune_pos], target, head
        )
        bundles[target] = {"head": head, "margin": margin, "interval_scale": interval_scale}
        item = {
            "risk_margin": margin,
            "normalized_conformal_interval_scale": interval_scale,
            "training_rows": int(len(rows["train"])),
            "tuning_rows": int(len(rows["tune"])),
            "tuning_trace": tuning_trace,
            "evaluation": {},
        }
        for role, label in (("test", "eicu_patient"), ("hospital", "eicu_hospital")):
            pos = positions(rows[role], source_lookup)
            item["evaluation"][label] = evaluate(
                source_frame, rows[role], source_features[target][pos], target,
                head, margin, interval_scale, label, args.seed,
            )
        report["targets"][target] = item

    del source_features
    print("prepare MIMIC-IV external features", flush=True)
    external_frame, external_rows, external_features, external_lookup = prepare_external(args, assets)
    for target in TARGETS:
        rows = external_rows[target]
        pos = positions(rows, external_lookup)
        bundle = bundles[target]
        result = evaluate(
            external_frame, rows, external_features[target][pos], target,
            bundle["head"], bundle["margin"], bundle["interval_scale"],
            "mimic_external", args.seed,
        )
        item = report["targets"][target]
        item["evaluation"]["mimic_external"] = result
        item["promotion_allowed"] = all(value["full_gate_pass"] for value in item["evaluation"].values())
        print(json.dumps({
            "target": target,
            "margin": bundle["margin"],
            "mimic_gate": result["full_gate_pass"],
            "mimic_move_fraction": result["move_fraction"],
            "mimic_move_precision": result["move_precision"],
            "mimic_mae": result["mae"],
            "mimic_crps": result["crps"],
            "mimic_coverage": result["coverage"],
        }, allow_nan=False), flush=True)
    report["summary"] = {
        "evaluated_targets": len(TARGETS),
        "promoted_targets": [target for target, item in report["targets"].items() if item["promotion_allowed"]],
        "mimic_point_better_than_persistence": [
            target for target, item in report["targets"].items()
            if item["evaluation"]["mimic_external"]["mae"]["distribution_risk"]
            < item["evaluation"]["mimic_external"]["mae"]["persistence"]
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(
        {
            "schema": "whole_body_jepa.distribution_risk_kidney_heads.v1",
            "status": "candidate_only",
            "release_binding": report["release_binding"],
            "heads": bundles,
        },
        args.output_dir / "heads.joblib",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_findings(report, args.findings)
    print(json.dumps(report["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--eicu", type=Path,
        default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_future_label_repaired/eicu_aki_transitions_6h.parquet"),
    )
    parser.add_argument(
        "--mimic", type=Path,
        default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/mimiciv_observation_transitions_6h_full_v31_treatment_unit_safe.parquet"),
    )
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/distribution_risk_kidney_relative_20260904"))
    parser.add_argument("--findings", type=Path, default=Path("DISTRIBUTION_RISK_KIDNEY_RELATIVE_FINDINGS_20260904.md"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
