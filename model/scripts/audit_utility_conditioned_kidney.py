#!/usr/bin/env python3
"""Test whether shared body state can route between persistence and a target expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from osler_jepa.shared_release_features import FEATURE_SLICES
from scripts.audit_release_masked_cells import donor_indices
from scripts.audit_state_conditioned_kidney import (
    BOUNDS,
    ROLE_CAPS,
    TARGETS,
    comparison_gate,
    coverage_gate,
    encode_features,
    feature_index,
    patient_bootstrap,
    positions,
    prepare_external,
    prepare_source,
)
from whole_body_jepa.multisource_intervals import conformal_radius, sha256, stable_hash
from whole_body_jepa.runtime import load_release
from whole_body_jepa.state_conditioned import (
    DYNAMIC_THRESHOLDS,
    empirical_crps,
    fit_utility_conditioned_head,
    fit_utility_router,
    predict_utility_conditioned,
    utility_mixture_samples,
)


ROUTER_STATE_STOP = FEATURE_SLICES["patient_belief"].stop
CUTOFFS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)


def expert_router_split(frame: pd.DataFrame, rows: np.ndarray, target: str) -> tuple[np.ndarray, np.ndarray]:
    buckets = np.asarray(
        [stable_hash("utility_role", target, patient) % 5 for patient in frame.iloc[rows]["subject_id"]]
    )
    return rows[buckets < 3], rows[buckets >= 3]


def fit_head(
    frame: pd.DataFrame,
    train_rows: np.ndarray,
    train_features: np.ndarray,
    train_lookup: dict[int, int],
    target: str,
    seed: int,
) -> tuple[dict, dict, dict]:
    expert_rows, router_rows = expert_router_split(frame, train_rows, target)
    expert_features = train_features[positions(expert_rows, train_lookup)]
    router_features = train_features[positions(router_rows, train_lookup)]
    expert_frame = frame.iloc[expert_rows]
    router_frame = frame.iloc[router_rows]
    expert_delta = (
        expert_frame[f"future_{target}"].to_numpy(dtype=float)
        - expert_frame[f"{target}_t"].to_numpy(dtype=float)
    )
    router_delta = (
        router_frame[f"future_{target}"].to_numpy(dtype=float)
        - router_frame[f"{target}_t"].to_numpy(dtype=float)
    )
    head = fit_utility_conditioned_head(
        expert_features,
        expert_delta,
        router_features,
        router_features[:, :ROUTER_STATE_STOP],
        router_delta,
        seed=seed,
    )
    expert_prediction = head["point_head"].predict(router_features)
    realized_benefit = np.abs(router_delta) - np.abs(router_delta - expert_prediction)
    donor = donor_indices(router_frame["subject_id"], seed)
    placebo_router = fit_utility_router(
        router_features[donor, :ROUTER_STATE_STOP], realized_benefit, seed=seed + 101
    )
    placebo = {**head, **placebo_router}
    support = {
        "expert_fit_rows": int(len(expert_rows)),
        "router_fit_rows": int(len(router_rows)),
        "router_beneficial_fraction": float((realized_benefit > 0.0).mean()),
    }
    return head, placebo, support


def _point_predictions(head: dict, features: np.ndarray, cutoff: float) -> tuple[dict, np.ndarray]:
    prediction = predict_utility_conditioned(
        head,
        features,
        features[:, :ROUTER_STATE_STOP],
        probability_cutoff=cutoff,
    )
    return prediction, prediction["point_delta"]


def tune(frame: pd.DataFrame, rows: np.ndarray, features: np.ndarray, target: str, head: dict) -> tuple[float, float]:
    selected = frame.iloc[rows].reset_index(drop=True)
    current = selected[f"{target}_t"].to_numpy(dtype=float)
    truth = selected[f"future_{target}"].to_numpy(dtype=float)
    delta = truth - current
    dynamic = np.abs(delta) >= DYNAMIC_THRESHOLDS[target]
    choices = []
    for cutoff in CUTOFFS:
        prediction, point_delta = _point_predictions(head, features, cutoff)
        error = np.abs(current + point_delta - truth)
        patient_mae = pd.Series(error, index=selected["subject_id"]).groupby(level=0).mean().mean()
        persistence_mae = pd.Series(np.abs(delta), index=selected["subject_id"]).groupby(level=0).mean().mean()
        stable_harm = float(error[~dynamic].mean() - np.abs(delta[~dynamic]).mean())
        move_fraction = float(prediction["move"].mean())
        eligible = patient_mae < persistence_mae and stable_harm <= 0.01 * DYNAMIC_THRESHOLDS[target]
        choices.append((not eligible, float(patient_mae), -move_fraction, float(cutoff)))
    cutoff = min(choices)[3]
    prediction, _ = _point_predictions(head, features, cutoff)
    samples = utility_mixture_samples(prediction, head["persistence_delta_quantiles"])
    center = current + np.quantile(samples, 0.50, axis=1)
    low = current + np.quantile(samples, 0.05, axis=1)
    high = current + np.quantile(samples, 0.95, axis=1)
    lower_width = np.maximum(center - low, 1e-6)
    upper_width = np.maximum(high - center, 1e-6)
    nonconformity = np.maximum((center - truth) / lower_width, (truth - center) / upper_width)
    return cutoff, conformal_radius(np.maximum(nonconformity, 0.0))


def evaluate(
    frame: pd.DataFrame,
    rows: np.ndarray,
    features: np.ndarray,
    target: str,
    head: dict,
    placebo: dict,
    cutoff: float,
    interval_scale: float,
    role: str,
) -> dict:
    selected = frame.iloc[rows].reset_index(drop=True)
    current = selected[f"{target}_t"].to_numpy(dtype=float)
    truth = selected[f"future_{target}"].to_numpy(dtype=float)
    delta = truth - current
    actual, point_delta = _point_predictions(head, features, cutoff)
    placebo_prediction, placebo_delta = _point_predictions(placebo, features, cutoff)
    point = current + point_delta
    plain = current + actual["expert_delta"]
    placebo_point = current + placebo_delta
    realized_benefit = np.abs(delta) - np.abs(delta - actual["expert_delta"])
    beneficial = realized_benefit > 0.0
    benefit_auc = None
    if beneficial.any() and (~beneficial).any():
        benefit_auc = float(roc_auc_score(beneficial, actual["benefit_probability"]))
    moved = actual["move"]
    samples = utility_mixture_samples(actual, head["persistence_delta_quantiles"])
    persistence_samples = np.tile(head["persistence_delta_quantiles"], (len(rows), 1))
    candidate_crps = empirical_crps(samples, delta)
    persistence_crps = empirical_crps(persistence_samples, delta)
    center = current + np.quantile(samples, 0.50, axis=1)
    base_lower = current + np.quantile(samples, 0.05, axis=1)
    base_upper = current + np.quantile(samples, 0.95, axis=1)
    lower = center - interval_scale * (center - base_lower)
    upper = center + interval_scale * (base_upper - center)
    dynamic = np.abs(delta) >= DYNAMIC_THRESHOLDS[target]
    comparisons = {
        "persistence": comparison_gate(selected, point, current, truth, f"utility:{target}:{role}:persistence"),
        "plain_equal_input_head": comparison_gate(selected, point, plain, truth, f"utility:{target}:{role}:plain"),
        "capacity_matched_placebo": comparison_gate(
            selected, point, placebo_point, truth, f"utility:{target}:{role}:placebo"
        ),
    }
    crps_gate = patient_bootstrap(
        selected, candidate_crps - persistence_crps, f"utility:{target}:{role}:crps"
    )
    coverage = coverage_gate(selected, truth, lower, upper)
    result = {
        "rows": int(len(selected)),
        "patients": int(selected["subject_id"].nunique()),
        "benefit_auc": benefit_auc,
        "move_fraction": float(moved.mean()),
        "move_precision": float(beneficial[moved].mean()) if moved.any() else None,
        "stable_false_move_fraction": float(moved[~dynamic].mean()) if (~dynamic).any() else None,
        "mae": {
            "utility_conditioned": float(np.abs(point - truth).mean()),
            "persistence": float(np.abs(current - truth).mean()),
            "plain_equal_input_head": float(np.abs(plain - truth).mean()),
            "capacity_matched_placebo": float(np.abs(placebo_point - truth).mean()),
            "stable_utility_conditioned": float(np.abs(point[~dynamic] - truth[~dynamic]).mean()),
            "stable_persistence": float(np.abs(current[~dynamic] - truth[~dynamic]).mean()),
            "dynamic_utility_conditioned": float(np.abs(point[dynamic] - truth[dynamic]).mean()),
            "dynamic_persistence": float(np.abs(current[dynamic] - truth[dynamic]).mean()),
        },
        "crps": {
            "utility_conditioned": float(candidate_crps.mean()),
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
        "# Utility-Conditioned Kidney Router Audit", "", "## Result", "",
        "The router is supervised by out-of-sample realized benefit rather than future-change magnitude.",
        "A target expert is fit on one patient partition; the router is fit on disjoint patients.",
        "MIMIC-IV labels do not select the router, cutoff or interval scale.", "",
        "| Target | Cutoff | eICU patient | eICU hospital | MIMIC-IV | Promotion |", "|---|---:|---:|---:|---:|---:|",
    ]
    for target, item in report["targets"].items():
        evaluation = item["evaluation"]
        lines.append(
            f"| {target}@{report['horizon_hours']}h | {item['probability_cutoff']:.3f} | "
            f"{evaluation['eicu_patient']['full_gate_pass']} | "
            f"{evaluation['eicu_hospital']['full_gate_pass']} | "
            f"{evaluation['mimic_external']['full_gate_pass']} | {item['promotion_allowed']} |"
        )
    lines += [
        "", "## Boundary", "",
        "This is a factual candidate audit. Failure leaves the validated release and persistence fallback unchanged.",
        "MIMIC-IV has been inspected previously and is not described as an untouched lockbox.",
        "No causal or clinical-decision claim is allowed.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    report = {
        "schema": "whole_body_jepa.utility_conditioned_kidney_audit.v1",
        "horizon_hours": args.horizon,
        "source_database": "eICU",
        "external_database": "MIMIC-IV",
        "external_is_untouched_lockbox": False,
        "release_binding": {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()},
        "routing_target": "out_of_sample_realized_mae_benefit",
        "router_inputs": "shared_context_and_patient_belief_only",
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
        print(f"fit utility-conditioned head: {target}", flush=True)
        rows = source_rows[target]
        head, placebo, support = fit_head(
            source_frame, rows["train"], source_features[target], source_lookup, target, args.seed
        )
        tune_pos = positions(rows["tune"], source_lookup)
        cutoff, interval_scale = tune(
            source_frame, rows["tune"], source_features[target][tune_pos], target, head
        )
        bundles[target] = {"head": head, "placebo": placebo, "cutoff": cutoff, "interval_scale": interval_scale}
        item = {
            "probability_cutoff": cutoff,
            "normalized_conformal_interval_scale": interval_scale,
            "support": support,
            "evaluation": {},
        }
        for role, label in (("test", "eicu_patient"), ("hospital", "eicu_hospital")):
            pos = positions(rows[role], source_lookup)
            item["evaluation"][label] = evaluate(
                source_frame, rows[role], source_features[target][pos], target,
                head, placebo, cutoff, interval_scale, label,
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
            bundle["head"], bundle["placebo"], bundle["cutoff"], bundle["interval_scale"],
            "mimic_external",
        )
        item = report["targets"][target]
        item["evaluation"]["mimic_external"] = result
        item["promotion_allowed"] = all(value["full_gate_pass"] for value in item["evaluation"].values())
        print(json.dumps({
            "target": target,
            "cutoff": bundle["cutoff"],
            "mimic_gate": result["full_gate_pass"],
            "mimic_benefit_auc": result["benefit_auc"],
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
            if item["evaluation"]["mimic_external"]["mae"]["utility_conditioned"]
            < item["evaluation"]["mimic_external"]["mae"]["persistence"]
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(
        {
            "schema": "whole_body_jepa.utility_conditioned_kidney_heads.v1",
            "status": "candidate_only",
            "release_binding": report["release_binding"],
            "heads": bundles,
        },
        args.output_dir / "heads.joblib",
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/utility_conditioned_kidney_20260904"))
    parser.add_argument("--findings", type=Path, default=Path("UTILITY_CONDITIONED_KIDNEY_FINDINGS_20260904.md"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
