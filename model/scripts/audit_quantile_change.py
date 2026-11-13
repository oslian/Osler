"""Frozen-JEPA distribution-risk and pooled-CQR follow-up; never updates runtime."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.audit_prospective_change import (
    TARGETS, METHODS, role_masks, select_role_rows, prepare, valid_rows,
    predict_raw, evaluate_predictions,
)
from scripts.audit_release_masked_cells import read_examples, input_frame
from whole_body_jepa.runtime import load_release, _build_inputs
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS, estimate, interval_score
from whole_body_jepa.multisource_intervals import sha256, stable_hash, representative_rows, subject_digest
from whole_body_jepa.prospective_change import calibrate_risk, alarm_cutoff
from whole_body_jepa.quantile_change import (
    fit_distribution, predict_distribution, tail_probabilities, pinball_score,
    fit_pooled_cqr, pooled_bounds, paired_ranking_recall,
)


POLICIES = ("pooled", "normalized")


def raw_predictions(data, bundle, old, target):
    current = data["frame"][f"{target}_t"].to_numpy(float)
    previous = predict_raw(data, old["heads"], target)
    raw = {}
    for method in METHODS:
        if method in bundle["heads"]:
            delta_q = predict_distribution(bundle["heads"][method], data["features"][method])
            tails = tail_probabilities(delta_q, CHANGE_THRESHOLDS[target])
            risk = np.clip(tails[:, 0] + tails[:, 2], 1e-6, 1 - 1e-6)
            q = delta_q + current[:, None]
            raw[method] = {"quantiles": q, "point": q[:, 9], "score": np.log(risk / (1 - risk)),
                           "uncalibrated_change_probability": risk}
        else:
            p = previous[method]
            raw[method] = {"quantiles": p["quantiles"], "point": p["quantiles"][:, 1], "score": p["risk_logit"]}
    p = previous["direct"]
    raw["previous_head"] = {"quantiles": p["quantiles"], "point": p["quantiles"][:, 1], "score": p["risk_logit"]}
    return raw


def calibration_bundle(data, heads, old, target):
    val, cal = data["validation"], data["calibration"]
    bundle = {"heads": heads, "risk_boundary": old["risk_boundary"]}
    raw = raw_predictions(val, bundle, old, target)
    f = val["frame"]
    labels = np.abs(f[f"future_{target}"] - f[f"{target}_t"]).to_numpy() >= CHANGE_THRESHOLDS[target]
    bundle["probability"] = {m: calibrate_risk(v["score"], labels, f.subject_id) for m, v in raw.items()}
    bundle["cutoffs"] = {m: alarm_cutoff(bundle["probability"][m].predict(v["score"]), labels) for m, v in raw.items()}
    cf = cal["frame"]
    cr = raw_predictions(cal, bundle, old, target)
    bundle["radii"] = {policy: {m: fit_pooled_cqr(cf[f"future_{target}"].to_numpy(float),
        v["quantiles"], cf.subject_id, CHANGE_THRESHOLDS[target], normalized=policy == "normalized")
        for m, v in cr.items() if m != "release"} for policy in POLICIES}
    return bundle


def evaluate(data, bundle, old, target, bootstrap):
    sub = data["frame"]
    raw = raw_predictions(data, bundle, old, target)
    risks = {m: bundle["probability"][m].predict(v["score"]) for m, v in raw.items()}
    high = old["probability"]["direct"].predict(raw["previous_head"]["score"]) > old["risk_boundary"]
    primary = representative_rows(sub, stable_hash("prospective_primary", target))
    f = sub.iloc[primary].reset_index(drop=True)
    truth = f[f"future_{target}"].to_numpy(float)
    changed = np.abs(truth - f[f"{target}_t"].to_numpy(float)) >= CHANGE_THRESHOLDS[target]
    result = {"policies": {}, "paired_recall_at_5pct_fpr": {}, "distribution_pinball": {}}
    for m in raw:
        if m != "direct":
            result["paired_recall_at_5pct_fpr"][m] = paired_ranking_recall(changed, risks["direct"][primary], risks[m][primary], repetitions=bootstrap)
        if m in bundle["heads"]:
            score = pinball_score(truth, raw[m]["quantiles"][primary])
            result["distribution_pinball"][m] = estimate(f, score, repetitions=bootstrap)
    for policy in POLICIES:
        predictions = {}
        for m, v in raw.items():
            q = v["quantiles"]
            lo, hi = (q[:, 0], q[:, -1]) if m == "release" else pooled_bounds(
                q, CHANGE_THRESHOLDS[target], bundle["radii"][policy][m], normalized=policy == "normalized")
            predictions[m] = {"point": v["point"], "lower": lo, "upper": hi,
                              "risk": risks[m], "alert": risks[m] > bundle["cutoffs"][m]}
        scored = evaluate_predictions(data, predictions, high, bundle["cutoffs"], target, bootstrap)
        scored["risk_group_definition"] = "previous_head_source_validation_median_preanchor_risk"
        direct = predictions["direct"]
        scores = interval_score(truth, direct["lower"][primary], direct["upper"][primary])
        scored["paired_interval_score"] = {}
        for m, p in predictions.items():
            if m != "direct":
                score = interval_score(truth, p["lower"][primary], p["upper"][primary])
                scored["paired_interval_score"][m] = estimate(f, scores - score, repetitions=bootstrap)
        # The inherited diagnostic is only a screen. This additional audit does
        # not authorize promotion, even if every displayed check happens to pass.
        direct_metrics = scored["methods"]["direct"]
        unit_pass = len(direct_metrics["careunits"]) >= 2 and all(
            u["coverage_status"] == "within_nominal_band" for u in direct_metrics["careunits"].values())
        recall_pass = all(v["ci95"] is not None and v["ci95"][0] > 0
            for m, v in result["paired_recall_at_5pct_fpr"].items() if m in ("release", "history_only", "shuffled_history"))
        scored["careunit_interval_screen_pass"] = unit_pass
        scored["prospective_recall_screen_pass"] = recall_pass
        scored["combined_diagnostic_pass"] = scored["diagnostic_screen_pass"] and unit_pass and recall_pass
        result["policies"][policy] = scored
    return result


def write(report, output):
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run(args):
    torch.set_num_threads(4)
    assets = load_release(args.model_dir)
    bindings = {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}
    previous = joblib.load(args.previous)
    if previous["release_binding"] != bindings:
        raise ValueError("previous candidate belongs to another release")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "quantile_change.v1", "complete": False, "promotion_allowed": False,
        "runtime_modified": False, "clinical_claim_allowed": False, "causal_claim_allowed": False,
        "next_measurement_time_modeled": False, "untouched_lockbox": False,
        "release_binding": bindings, "previous_sha256": sha256(args.previous),
        "protocol_sha256": sha256(Path("QUANTILE_CHANGE_PROTOCOL_20260907.md")),
        "code_sha256": {p: sha256(Path(p)) for p in ("scripts/audit_quantile_change.py",
            "whole_body_jepa/quantile_change.py", "scripts/audit_prospective_change.py")},
        "training_seeds": [20260906], "inputs": {"eicu": sha256(args.eicu)}, "targets": {}}
    print("Read eICU; preserve all prior patient roles and anchor selections", flush=True)
    frame = read_examples(args.eicu, assets, "eicu", 3, grid=True)
    masks = role_masks(frame)
    sets = {r: set(frame.loc[m, "subject_id"]) for r, m in masks.items()}
    if any(sets[a] & sets[b] for a in sets for b in sets if a < b):
        raise ValueError("overlapping patient roles")
    report["role_subject_digests"] = {r: subject_digest(s) for r, s in sets.items()}
    arrays = _build_inputs(input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours), assets, 3)
    bundles = {}
    for target in TARGETS:
        data = {}
        old = previous["bundles"][target]
        for role, mask in masks.items():
            rows = select_role_rows(frame, valid_rows(frame, target), mask,
                patients=5000 if role in ("fit", "validation", "calibration") else 500,
                anchors=6 if role == "fit" else 1 if role in ("validation", "calibration") else 0,
                salt=f"{target}:{role}")
            print(target, role, len(rows), flush=True)
            data[role] = prepare(frame, assets, arrays, rows, target)
        f = data["fit"]["frame"]
        delta = (f[f"future_{target}"] - f[f"{target}_t"]).to_numpy(float)
        heads = {}
        for method in ("direct", "shuffled_history", "history_only"):
            print("Fit 19 quantiles", target, method, flush=True)
            heads[method] = fit_distribution(data["fit"]["features"][method], delta, f.subject_id)
        parity = {m: {str(q): float(np.max(np.abs(heads[m][index].predict(data["validation"]["features"][m]) -
            old["heads"][m]["quantiles"][q].predict(data["validation"]["features"][m]))))
            for q, index in ((.05, 0), (.5, 9), (.95, 18))} for m in heads}
        bundle = calibration_bundle(data, heads, old, target)
        bundles[target] = bundle
        item = {"support": {r: {"rows": len(d["frame"]), "patients": int(d["frame"].subject_id.nunique())} for r, d in data.items()},
                "q05_q50_q95_parity_max_abs": parity, "radii": bundle["radii"], "evaluation": {}}
        report["targets"][target] = item
        for role in ("patient_test", "hospital_test"):
            print("Evaluate", target, role, flush=True)
            item["evaluation"][role] = evaluate(data[role], bundle, old, target, args.bootstrap)
        write(report, args.output)
        del data
    candidate = args.output / "candidate_heads.joblib"
    joblib.dump({"status": "candidate_only", "release_binding": bindings,
        "previous_sha256": report["previous_sha256"], "bundles": bundles}, candidate)
    report["candidate_sha256_before_external"] = sha256(candidate)
    (args.output / "development.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    del frame, arrays
    print("Candidate frozen; now read NWICU", flush=True)
    report["inputs"]["nwicu"] = sha256(args.nwicu)
    frame = read_examples(args.nwicu, assets, "nwicu", 3)
    arrays = _build_inputs(input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours), assets, 3)
    for target in TARGETS:
        rows = select_role_rows(frame, valid_rows(frame, target), np.ones(len(frame), dtype=bool),
                                patients=500, anchors=0, salt=f"{target}:nwicu")
        print("NWICU", target, len(rows), flush=True)
        data = prepare(frame, assets, arrays, rows, target)
        result = evaluate(data, bundles[target], previous["bundles"][target], target, args.bootstrap)
        report["targets"][target]["evaluation"]["nwicu"] = result
        for p, s in result["policies"].items():
            m = s["methods"]["direct"]
            print(json.dumps({"target": target, "policy": p, "pass": s["combined_diagnostic_pass"],
                "mae": m["change"]["all"]["mae"]["mean"],
                "coverage": m["change"]["all"]["coverage_when_interval_available"]["mean"],
                "recall5": m["risk"]["recall_at_matched_5pct_fpr_ranking_only"]}), flush=True)
        write(report, args.output)
    if bindings != {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}:
        raise ValueError("release changed during audit")
    if sha256(candidate) != report["candidate_sha256_before_external"] or sha256(args.previous) != report["previous_sha256"]:
        raise ValueError("candidate changed during external evaluation")
    report["complete"] = True
    write(report, args.output)
    print("FINISHED; release unchanged", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--previous", type=Path, default=Path("artifacts/prospective_change_20260906/candidate_heads.joblib"))
    p.add_argument("--eicu", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet"))
    p.add_argument("--nwicu", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("artifacts/quantile_change_20260907"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
