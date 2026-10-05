"""Matched repair: expose expert outputs to the shared-state persistence selector."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from scripts.audit_expected_error_routing import (
    assert_disjoint, calibrating_split, predictions as old_predictions, interval_parts,
)
from scripts.audit_preanchor_context_routing import (
    ROUTER_STATE_STOP, _router_features, _eicu_context, _mimic_context,
)
from scripts.audit_release_masked_cells import donor_indices
from scripts.audit_state_conditioned_kidney import (
    TARGETS, positions, prepare_source, prepare_external, comparison_gate,
    coverage_gate, patient_bootstrap,
)
from scripts.audit_utility_conditioned_kidney import expert_router_split
from whole_body_jepa.expected_error_routing import (
    FORECAST_EVIDENCE_NAMES, forecast_evidence, fit_expected_error_router,
    fit_cost_router, cost_advantage,
)
from whole_body_jepa.multisource_intervals import conformal_radius, sha256
from whole_body_jepa.runtime import load_release
from whole_body_jepa.state_conditioned import DYNAMIC_THRESHOLDS, empirical_crps


NEW_METHODS = ("aware_mean", "aware_cost", "shuffled_mean", "shuffled_cost",
               "no_context_mean", "no_context_cost", "blind_cost")


def inputs(base, context, evidence, subjects, seed):
    shared = _router_features(base, context)
    shuffled = evidence[donor_indices(subjects, seed)]
    return {"aware": np.column_stack((shared, evidence)),
            "shuffled": np.column_stack((shared, shuffled)),
            "no_context": np.column_stack((base[:, :ROUTER_STATE_STOP], evidence)),
            "blind": shared}


def fixed_predictions(old_bundle, frame, base, context, target, seed):
    pred = old_predictions(old_bundle["model"], base, context, frame.subject_id, seed)
    head = old_bundle["model"]["conservative"]["context"]
    quantiles = np.column_stack([head["quantile_heads"][str(q)].predict(base)
                                 for q in (.05, .5, .95)])
    pred["evidence"] = forecast_evidence(
        frame[f"{target}_t"].to_numpy(float), frame[f"{target}_age_hr"].to_numpy(float),
        pred["expert"], quantiles, scale=DYNAMIC_THRESHOLDS[target],
    )
    return pred


def fit_heads(base, context, pred, frame, target, seed):
    x = inputs(base, context, pred["evidence"], frame.subject_id, seed + 101)
    delta = frame[f"future_{target}"].to_numpy(float) - frame[f"{target}_t"].to_numpy(float)
    heads = {}
    for variant in ("aware", "shuffled", "no_context"):
        heads[f"{variant}_mean"] = fit_expected_error_router(
            x[variant], delta, pred["expert"], seed=seed + 51)["benefit"]
        heads[f"{variant}_cost"] = fit_cost_router(
            x[variant], delta, pred["expert"], seed=seed + 51)
    heads["blind_cost"] = fit_cost_router(x["blind"], delta, pred["expert"], seed=seed + 51)
    return heads


def add_scores(heads, pred, base, context, subjects, seed):
    x = inputs(base, context, pred["evidence"], subjects, seed + 301)
    pred["new_scores"] = {}
    for method in NEW_METHODS:
        variant, objective = method.rsplit("_", 1)
        pred["new_scores"][method] = (heads[method].predict(x[variant]) if objective == "mean"
                                      else cost_advantage(heads[method], x[variant]))
    return pred


def choose_thresholds(pred, truth_delta, target):
    thresholds, losses = {}, {}
    for method in NEW_METHODS:
        margins = (tuple(DYNAMIC_THRESHOLDS[target] * x for x in (0, .02, .05, .1, .2))
                   if method.endswith("mean") else (0, .1, .2, .4, .6))
        choices = []
        for margin in margins:
            move = pred["new_scores"][method] > margin
            error = np.abs(truth_delta - np.where(move, pred["expert"], 0))
            choices.append((float(error.mean()), float(move.mean()), float(margin)))
        losses[method], _, thresholds[method] = min(choices)
    primary = min(("aware_mean", "aware_cost"), key=lambda m: losses[m])
    return thresholds, losses, primary


def choices(pred, thresholds, old):
    move = {name: pred["new_scores"][name] > value for name, value in thresholds.items()}
    for name, original in (("old_primary", old["primary"]), ("old_q25", "old_q25"),
                           ("blind_mean", "mean_benefit")):
        move[name] = pred["scores"][original] > old["thresholds"][original]
        if original == "old_q25":
            move[name] &= pred["q25"] > 0
    move["persistence"] = np.zeros(len(pred["expert"]), dtype=bool)
    move["always_expert"] = np.ones(len(pred["expert"]), dtype=bool)
    return move


def calibrate(pred, delta, thresholds, old):
    scales = {}
    for method, move in choices(pred, thresholds, old).items():
        _, center, left, right = interval_parts(pred, move)
        scores = np.maximum.reduce(((center - delta) / left, (delta - center) / right,
                                    np.zeros(len(delta))))
        scales[method] = conformal_radius(scores)
    return scales


def evaluate(frame, pred, thresholds, scales, old, target, role, primary):
    current = frame[f"{target}_t"].to_numpy(float)
    truth = frame[f"future_{target}"].to_numpy(float)
    delta = truth - current
    selected = choices(pred, thresholds, old)
    points = {name: current + np.where(move, pred["expert"], 0) for name, move in selected.items()}
    beneficial = np.abs(delta) > np.abs(delta - pred["expert"])
    dynamic = np.abs(delta) >= DYNAMIC_THRESHOLDS[target]
    wrong_direction = pred["expert"] * delta < 0
    overshoot = (~wrong_direction) & (np.abs(pred["expert"]) > 2 * np.abs(delta))
    result = {"patients": int(frame.subject_id.nunique()), "rows": len(frame),
              "dynamic_fraction": float(dynamic.mean()), "expert_winner_fraction": float(beneficial.mean()),
              "expert_wrong_direction_fraction": float(wrong_direction.mean()),
              "expert_overshoot_fraction": float(overshoot.mean()), "methods": {}}
    reference_samples = np.tile(pred["persistence_samples"], (len(frame), 1))
    reference_crps = empirical_crps(reference_samples, delta)
    for method, move in selected.items():
        errors = np.abs(points[method] - truth)
        samples, center, left, right = interval_parts(pred, move)
        low = current + center - scales[method] * left
        high = current + center + scales[method] * right
        hit = (truth >= low) & (truth <= high)
        score = pred["new_scores"].get(method)
        item = {"mae": float(errors.mean()), "move_fraction": float(move.mean()),
                "stable_mae": float(errors[~dynamic].mean()) if (~dynamic).any() else None,
                "dynamic_mae": float(errors[dynamic].mean()) if dynamic.any() else None,
                "move_precision": float(beneficial[move].mean()) if move.any() else None,
                "selected_wrong_direction_fraction": float(wrong_direction[move].mean()) if move.any() else None,
                "selected_overshoot_fraction": float(overshoot[move].mean()) if move.any() else None,
                "benefit_auc": float(roc_auc_score(beneficial, score))
                    if score is not None and len(np.unique(beneficial)) == 2 else None,
                "coverage": coverage_gate(frame, truth, low, high),
                "coverage_by_careunit": {str(u): {"n": int(mask.sum()), "coverage": float(hit[mask].mean())}
                    for u in frame.careunit.unique() for mask in [frame.careunit.eq(u).to_numpy()]},
                "crps": float(empirical_crps(samples, delta).mean())}
        if method == primary:
            objective = primary.rsplit("_", 1)[1]
            references = ("persistence", "always_expert", "old_primary", f"blind_{objective}",
                          f"shuffled_{objective}", f"no_context_{objective}")
            item["comparisons"] = {name: comparison_gate(frame, points[method], points[name], truth,
                f"forecast_aware:{target}:{role}:{name}") for name in references}
            item["crps_gate"] = patient_bootstrap(frame, empirical_crps(samples, delta) - reference_crps,
                                                   f"forecast_aware:{target}:{role}:crps")
            item["full_gate_pass"] = bool(item["coverage"]["pass"] and item["crps_gate"]["pass"]
                and all(v["pass"] and v["all"]["pass"] for v in item["comparisons"].values()))
        result["methods"][method] = item
    bins = np.digitize(np.abs(pred["expert"]) / DYNAMIC_THRESHOLDS[target], [.5, 1, 2])
    result["proposal_magnitude_bins"] = {}
    for index, label in enumerate(("lt_0.5", "0.5_to_1", "1_to_2", "ge_2")):
        mask = bins == index
        if mask.any():
            result["proposal_magnitude_bins"][label] = {"n": int(mask.sum()),
                "realized_mean_benefit": float((np.abs(delta) - np.abs(delta - pred["expert"]))[mask].mean()),
                "primary_move_fraction": float(selected[primary][mask].mean())}
    return result


def verify_fixed_expert(result, previous):
    for name in ("persistence", "always_expert"):
        if not np.isclose(result["methods"][name]["mae"], previous["reference_mae"][name], atol=1e-8, rtol=0):
            raise ValueError("expert or evaluation cohort changed since the paired reference run")


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    previous = json.loads((args.previous / "report.json").read_text())
    for filename, digest in previous["release_binding"].items():
        if sha256(args.model_dir / filename) != digest:
            raise ValueError(f"release changed: {filename}")
    old_bundles = joblib.load(args.previous / "candidate_heads.joblib")
    assets = load_release(args.model_dir)
    report = {"schema": "forecast_aware_routing.v1", "targets": {},
              "previous_report_sha256": sha256(args.previous / "report.json"),
              "expert_bundle_sha256": sha256(args.previous / "candidate_heads.joblib"),
              "protocol_sha256": sha256(Path("FORECAST_AWARE_ROUTING_PROTOCOL_20260905.md")),
              "release_binding": previous["release_binding"],
              "added_feature_names": list(FORECAST_EVIDENCE_NAMES),
              "promotion_allowed": False, "runtime_modified": False, "causal_claim_allowed": False}
    print("prepare eICU frozen representations", flush=True)
    frame, rows, features, lookup = prepare_source(args, assets)
    for target, role in rows.items():
        selection, calibration = calibrating_split(frame, role["tune"])
        expert_fit, router_fit = expert_router_split(frame, role["train"], target)
        assert_disjoint(frame, {"expert": expert_fit, "router": router_fit, "selection": selection,
                               "calibration": calibration, "test": role["test"], "hospital": role["hospital"]})
    print("extract same pre-anchor eICU context", flush=True)
    context, names = _eicu_context(frame, lookup, args.eicu_root)
    if list(names) != previous["context_features"]:
        raise ValueError("context feature contract changed")
    bundles = {}
    for target, role in rows.items():
        print(f"fit forecast-aware selector: {target}", flush=True)
        old = old_bundles[target]
        selection, calibration = calibrating_split(frame, role["tune"])
        _, router_rows = expert_router_split(frame, role["train"], target)
        def fixed(which):
            pos = positions(which, lookup)
            sub = frame.iloc[which].reset_index(drop=True)
            return fixed_predictions(old, sub, features[target][pos], context[pos], target, args.seed), sub, pos
        pred, sub, pos = fixed(router_rows)
        heads = fit_heads(features[target][pos], context[pos], pred, sub, target, args.seed)
        def infer(which):
            pred, sub, pos = fixed(which)
            return add_scores(heads, pred, features[target][pos], context[pos], sub.subject_id, args.seed), sub
        pred, sub = infer(selection)
        thresholds, losses, primary = choose_thresholds(pred,
            sub[f"future_{target}"].to_numpy(float) - sub[f"{target}_t"].to_numpy(float), target)
        pred, sub = infer(calibration)
        scales = calibrate(pred, sub[f"future_{target}"].to_numpy(float) - sub[f"{target}_t"].to_numpy(float),
                           thresholds, old)
        bundles[target] = {"heads": heads, "thresholds": thresholds, "primary": primary, "scales": scales}
        item = {"primary": primary, "selection_mae": losses, "thresholds": thresholds,
                "interval_scales": scales, "selector_patients": len(router_rows),
                "selection_patients": len(selection), "calibration_patients": len(calibration), "evaluation": {}}
        for label in ("test", "hospital"):
            pred, sub = infer(role[label])
            result = evaluate(sub, pred, thresholds, scales, old, target, label, primary)
            verify_fixed_expert(result, previous["targets"][target]["evaluation"][label])
            item["evaluation"][label] = result
        report["targets"][target] = item
        print(json.dumps({"target": target, "primary": primary, "selection_mae": losses}), flush=True)
    joblib.dump(bundles, args.output_dir / "selector_heads.joblib")
    (args.output_dir / "development.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    del frame, features, context
    print("prepare MIMIC frozen representations", flush=True)
    frame, rows, features, lookup = prepare_external(args, assets)
    print("extract same pre-anchor MIMIC context", flush=True)
    context, external_names = _mimic_context(frame, lookup, args.mimic_root)
    if names != external_names:
        raise ValueError("context feature order mismatch")
    for target, which in rows.items():
        b, old = bundles[target], old_bundles[target]
        sub, pos = frame.iloc[which].reset_index(drop=True), positions(which, lookup)
        pred = fixed_predictions(old, sub, features[target][pos], context[pos], target, args.seed)
        pred = add_scores(b["heads"], pred, features[target][pos], context[pos], sub.subject_id, args.seed)
        result = evaluate(sub, pred, b["thresholds"], b["scales"], old, target, "mimic", b["primary"])
        verify_fixed_expert(result, previous["targets"][target]["evaluation"]["mimic"])
        item = report["targets"][target]
        item["evaluation"]["mimic"] = result
        item["all_role_gate_pass"] = all(e["methods"][b["primary"]]["full_gate_pass"]
                                         for e in item["evaluation"].values())
        print(json.dumps({"target": target, "primary": b["primary"],
            "mae": {m: v["mae"] for m, v in result["methods"].items()},
            "all_role_gate_pass": item["all_role_gate_pass"]}), flush=True)
    report["gate_pass_targets"] = [t for t, v in report["targets"].items() if v["all_role_gate_pass"]]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("FINISHED " + json.dumps(report["gate_pass_targets"]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, default=Path("artifacts/expected_error_routing_20260905_v2"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/forecast_aware_routing_20260905"))
    p.add_argument("--eicu", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_future_label_repaired/eicu_aki_transitions_6h.parquet"))
    p.add_argument("--mimic", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/mimiciv_observation_transitions_6h_full_v31_treatment_unit_safe.parquet"))
    p.add_argument("--eicu-root", type=Path, default=Path("/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0"))
    p.add_argument("--mimic-root", type=Path, default=Path("/Users/chunyouchang/Downloads/mimic-iv-3.1"))
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--threads", type=int, default=6)
    run(p.parse_args())


if __name__ == "__main__":
    main()
