"""Paired bounded-amplitude repair on frozen JEPA expert predictions."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.audit_expected_error_routing import assert_disjoint, calibrating_split
from scripts.audit_forecast_aware_routing import fixed_predictions, add_scores, choices, inputs
from scripts.audit_preanchor_context_routing import _eicu_context, _mimic_context
from scripts.audit_release_masked_cells import donor_indices
from scripts.audit_state_conditioned_kidney import (
    positions, prepare_source, prepare_external, comparison_gate, coverage_gate, patient_bootstrap,
)
from scripts.audit_utility_conditioned_kidney import expert_router_split
from whole_body_jepa.amplitude_calibration import fit_amplitude, adoption
from whole_body_jepa.multisource_intervals import conformal_radius, sha256
from whole_body_jepa.runtime import load_release
from whole_body_jepa.state_conditioned import DYNAMIC_THRESHOLDS, empirical_crps


LEARNED = ("conditional", "shuffled", "global", "evidence_only", "no_context")


def feature_blocks(base, context, pred, subjects, seed):
    x = inputs(base, context, pred["evidence"], subjects, seed)
    return {"conditional": x["aware"],
            "shuffled": x["aware"][donor_indices(subjects, seed + 701)],
            "global": x["aware"], "evidence_only": pred["evidence"],
            "no_context": x["no_context"]}


def truth_delta(frame, target):
    return frame[f"future_{target}"].to_numpy(float) - frame[f"{target}_t"].to_numpy(float)


def fit_heads(base, context, pred, frame, target, seed):
    blocks = feature_blocks(base, context, pred, frame.subject_id, seed)
    return {m: fit_amplitude(blocks[m], truth_delta(frame, target), pred["expert"],
                            conditional=m != "global", seed=seed) for m in LEARNED}


def predict(heads, old, binary, base, context, pred, frame, seed):
    blocks = feature_blocks(base, context, pred, frame.subject_id, seed + 301)
    alpha = {m: adoption(heads[m], blocks[m]) for m in LEARNED}
    scored = add_scores(binary["heads"], pred, base, context, frame.subject_id, seed)
    move = choices(scored, binary["thresholds"], old)[binary["primary"]]
    alpha.update(binary=move.astype(float), persistence=np.zeros(len(frame)),
                 always_expert=np.ones(len(frame)))
    return alpha


def centered_shape(pred, method):
    raw = (np.tile(pred["persistence_samples"], (len(pred["expert"]), 1))
           if method == "persistence" else pred["expert_samples"])
    shape = raw - np.median(raw, axis=1)[:, None]
    # Use the same small spread floor in calibration and evaluation.
    left = np.quantile(shape, .05, axis=1)
    right = np.quantile(shape, .95, axis=1)
    flat = (left > -1e-6) | (right < 1e-6)
    if flat.any():
        shape = shape.copy()
        floor = np.linspace(-1e-6 / .9, 1e-6 / .9, shape.shape[1])
        shape[flat] += floor
    return shape


def calibrate(pred, alpha, delta):
    scales = {}
    for method, weight in alpha.items():
        point = weight * pred["expert"]
        shape = centered_shape(pred, method)
        left = np.maximum(-np.quantile(shape, .05, axis=1), 1e-6)
        right = np.maximum(np.quantile(shape, .95, axis=1), 1e-6)
        score = np.maximum.reduce(((point - delta) / left, (delta - point) / right,
                                   np.zeros(len(delta))))
        scales[method] = conformal_radius(score)
    return scales


def distribution(pred, alpha, method, scale):
    point = alpha[method] * pred["expert"]
    shape = centered_shape(pred, method)
    return point[:, None] + float(scale) * shape


def evaluate(frame, pred, alpha, scales, target, role):
    current, delta = frame[f"{target}_t"].to_numpy(float), truth_delta(frame, target)
    truth = current + delta
    points = {m: current + a * pred["expert"] for m, a in alpha.items()}
    dynamic = np.abs(delta) >= DYNAMIC_THRESHOLDS[target]
    result = {"patients": int(frame.subject_id.nunique()), "rows": len(frame),
              "dynamic_fraction": float(dynamic.mean()), "methods": {}}
    crps = {}
    for method, weight in alpha.items():
        error = np.abs(points[method] - truth)
        samples = distribution(pred, alpha, method, scales[method])
        low, high = current + np.quantile(samples, .05, axis=1), current + np.quantile(samples, .95, axis=1)
        hit = (truth >= low) & (truth <= high)
        crps[method] = empirical_crps(samples, delta)
        result["methods"][method] = {
            "mae": float(error.mean()), "mean_alpha": float(weight.mean()),
            "alpha_quantiles": np.quantile(weight, [0, .25, .5, .75, 1]).tolist(),
            "move_fraction": float((np.abs(points[method] - current) > 1e-12).mean()),
            "stable_mae": float(error[~dynamic].mean()) if (~dynamic).any() else None,
            "dynamic_mae": float(error[dynamic].mean()) if dynamic.any() else None,
            "coverage": coverage_gate(frame, truth, low, high),
            "coverage_by_careunit": {str(u): {"n": int(mask.sum()), "coverage": float(hit[mask].mean())}
                for u in frame.careunit.unique() for mask in [frame.careunit.eq(u).to_numpy()]},
            "crps": float(crps[method].mean()),
        }
    primary = result["methods"]["conditional"]
    primary["comparisons"] = {m: comparison_gate(frame, points["conditional"], points[m], truth,
        f"amplitude:{target}:{role}:{m}") for m in alpha if m != "conditional"}
    primary["crps_gate"] = patient_bootstrap(frame, crps["conditional"] - crps["persistence"],
                                             f"amplitude:{target}:{role}:crps")
    primary["full_gate_pass"] = bool(primary["coverage"]["pass"] and primary["crps_gate"]["pass"]
        and all(v["pass"] and v["all"]["pass"] for v in primary["comparisons"].values()))
    return result


def verify_reference(result, previous, primary):
    for ours, theirs in (("binary", primary), ("persistence", "persistence"),
                         ("always_expert", "always_expert")):
        if not np.isclose(result["methods"][ours]["mae"], previous["methods"][theirs]["mae"], atol=1e-8, rtol=0):
            raise ValueError(f"fixed expert, binary selector or cohort changed: {ours}")


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    previous = json.loads((args.previous / "report.json").read_text())
    original = json.loads((args.experts / "report.json").read_text())
    for filename, digest in previous["release_binding"].items():
        if sha256(args.model_dir / filename) != digest:
            raise ValueError(f"release changed: {filename}")
    if sha256(args.experts / "candidate_heads.joblib") != previous["expert_bundle_sha256"]:
        raise ValueError("expert bundle changed")
    old_bundles = joblib.load(args.experts / "candidate_heads.joblib")
    binaries = joblib.load(args.previous / "selector_heads.joblib")
    assets = load_release(args.model_dir)
    report = {"schema": "bounded_amplitude.v1", "targets": {}, "primary": "conditional",
              "protocol_sha256": sha256(Path("AMPLITUDE_CALIBRATION_PROTOCOL_20260905.md")),
              "previous_report_sha256": sha256(args.previous / "report.json"),
              "binary_bundle_sha256": sha256(args.previous / "selector_heads.joblib"),
              "expert_bundle_sha256": previous["expert_bundle_sha256"],
              "release_binding": previous["release_binding"],
              "promotion_allowed": False, "runtime_modified": False, "causal_claim_allowed": False}
    print("prepare eICU frozen representations", flush=True)
    frame, rows, features, lookup = prepare_source(args, assets)
    for target, role in rows.items():
        selection, calibration = calibrating_split(frame, role["tune"])
        expert, fit = expert_router_split(frame, role["train"], target)
        assert_disjoint(frame, {"expert": expert, "fit": fit, "unused_selection": selection,
                               "calibration": calibration, "test": role["test"], "hospital": role["hospital"]})
    print("extract same eICU pre-anchor context", flush=True)
    context, names = _eicu_context(frame, lookup, args.eicu_root)
    if list(names) != original["context_features"]:
        raise ValueError("context feature contract changed")
    bundles = {}
    for target, role in rows.items():
        print(f"fit amplitude readouts: {target}", flush=True)
        old, binary = old_bundles[target], binaries[target]
        def fixed(which):
            pos = positions(which, lookup)
            sub = frame.iloc[which].reset_index(drop=True)
            return fixed_predictions(old, sub, features[target][pos], context[pos], target, args.seed), sub, pos
        _, fit = expert_router_split(frame, role["train"], target)
        pred, sub, pos = fixed(fit)
        heads = fit_heads(features[target][pos], context[pos], pred, sub, target, args.seed)
        def infer(which):
            pred, sub, pos = fixed(which)
            alpha = predict(heads, old, binary, features[target][pos], context[pos], pred, sub, args.seed)
            return pred, sub, alpha
        _, calibration = calibrating_split(frame, role["tune"])
        pred, sub, alpha = infer(calibration)
        scales = calibrate(pred, alpha, truth_delta(sub, target))
        bundles[target] = {"heads": heads, "scales": scales}
        item = {"fit_patients": len(fit), "calibration_patients": len(calibration),
                "scales": scales, "global_alpha": heads["global"]["constant"], "evaluation": {}}
        for label in ("test", "hospital"):
            pred, sub, alpha = infer(role[label])
            result = evaluate(sub, pred, alpha, scales, target, label)
            verify_reference(result, previous["targets"][target]["evaluation"][label], binary["primary"])
            item["evaluation"][label] = result
        report["targets"][target] = item
        print(json.dumps({"target": target, "global_alpha": item["global_alpha"],
                          "source_mae": {m: v["mae"] for m, v in item["evaluation"]["test"]["methods"].items()}}), flush=True)
    joblib.dump(bundles, args.output_dir / "amplitude_heads.joblib")
    (args.output_dir / "development.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    del frame, features, context
    print("prepare MIMIC frozen representations", flush=True)
    frame, rows, features, lookup = prepare_external(args, assets)
    print("extract same MIMIC pre-anchor context", flush=True)
    context, external_names = _mimic_context(frame, lookup, args.mimic_root)
    if names != external_names:
        raise ValueError("context feature order mismatch")
    for target, which in rows.items():
        b, old, binary = bundles[target], old_bundles[target], binaries[target]
        sub, pos = frame.iloc[which].reset_index(drop=True), positions(which, lookup)
        base, ctx = features[target][pos], context[pos]
        pred = fixed_predictions(old, sub, base, ctx, target, args.seed)
        alpha = predict(b["heads"], old, binary, base, ctx, pred, sub, args.seed)
        result = evaluate(sub, pred, alpha, b["scales"], target, "mimic")
        verify_reference(result, previous["targets"][target]["evaluation"]["mimic"], binary["primary"])
        item = report["targets"][target]
        item["evaluation"]["mimic"] = result
        item["all_role_gate_pass"] = all(e["methods"]["conditional"]["full_gate_pass"]
                                         for e in item["evaluation"].values())
        print(json.dumps({"target": target, "mae": {m: v["mae"] for m, v in result["methods"].items()},
                          "all_role_gate_pass": item["all_role_gate_pass"]}), flush=True)
    report["gate_pass_targets"] = [t for t, v in report["targets"].items() if v["all_role_gate_pass"]]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("FINISHED " + json.dumps(report["gate_pass_targets"]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, default=Path("artifacts/forecast_aware_routing_20260905"))
    p.add_argument("--experts", type=Path, default=Path("artifacts/expected_error_routing_20260905_v2"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/amplitude_calibration_20260905"))
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
