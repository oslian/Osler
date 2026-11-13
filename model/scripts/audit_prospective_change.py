"""Fixed JEPA signed-change readouts, matched controls and source-only CQR."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_change_behavior import frozen_outputs, label_mask
from scripts.audit_release_masked_cells import read_examples, input_frame
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS, estimate, evaluate_change, interval_score
from whole_body_jepa.multisource_intervals import representative_rows, stable_hash, sha256, patient_split, SEEDS, subject_digest
from whole_body_jepa.prospective_change import (
    trend_features, fit_head, raw_prediction, calibrate_risk, alarm_cutoff,
    recall_at_fpr, fit_risk_cqr, cqr_bounds,
)
from whole_body_jepa.runtime import load_release, _build_inputs


TARGETS = ("glucose", "map")
METHODS = ("direct", "shuffled_history", "history_only", "linear_trend", "release", "persistence")


def role_masks(frame):
    if "hospitalid" not in frame or frame.hospitalid.nunique() < 2:
        raise ValueError("source requires hospital provenance")
    hospital = np.asarray([stable_hash("prospective_change_hospital", h) % 5 == 0 for h in frame.hospitalid])
    excluded = frame.subject_id.isin(set(frame.loc[hospital, "subject_id"])).to_numpy()
    bucket = np.asarray([stable_hash("prospective_change_patient", p) % 10 for p in frame.subject_id])
    return {"fit": ~excluded & (bucket < 5), "validation": ~excluded & (bucket >= 5) & (bucket < 7),
            "calibration": ~excluded & (bucket == 7), "patient_test": ~excluded & (bucket >= 8), "hospital_test": hospital}


def select_role_rows(frame, valid, mask, *, patients, anchors, salt):
    eligible = np.flatnonzero(valid & mask)
    subset = frame.iloc[eligible]
    names = sorted(set(subset.subject_id), key=lambda p: stable_hash("prospective_cap", salt, p))[:patients]
    eligible = eligible[subset.subject_id.isin(names)]
    if anchors:
        selected = frame.iloc[eligible][["subject_id", "stay_id", "t"]].copy()
        selected["row"] = eligible
        selected["key"] = [stable_hash(salt, *v) for v in selected[["subject_id", "stay_id", "t"]].itertuples(index=False, name=None)]
        eligible = selected.sort_values("key").groupby("subject_id", sort=False).head(anchors).row.to_numpy()
    return np.sort(eligible)


def shuffled_donors(subjects, seed=20260906):
    subjects = np.asarray(subjects).astype(str)
    names = np.unique(subjects)
    if len(names) < 2:
        raise ValueError("shuffled history requires two distinct patients")
    rng = np.random.default_rng(seed)
    names = rng.permutation(names)
    mapping = dict(zip(names, np.roll(names, 1)))
    first = {p: np.flatnonzero(subjects == p)[0] for p in names}
    return np.asarray([first[mapping[p]] for p in subjects], dtype=int)


def encode(assets, arrays, frame, rows, target, donors=None):
    idx = assets.checkpoint["variables"].index(target)
    organ = int(assets.model.variable_module_ids[idx])
    features = []
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        past = batch if donors is None else rows[donors[start:start + 256]]
        history = arrays["input"][arrays["history"][past]].copy()
        if donors is not None:
            history[:, -1] = arrays["input"][batch]  # preserve own current (lag zero)
        with torch.inference_mode():
            output = assets.model(torch.from_numpy(history).float(),
                torch.from_numpy(arrays["current"][batch]).float(), torch.from_numpy(arrays["mask"][batch]).float(),
                torch.from_numpy(arrays["ages"][batch]).float(), torch.from_numpy(arrays["presence"][batch]).float(),
                torch.from_numpy(arrays["horizon"][batch]).float())
        out = {k: v.cpu().numpy() for k, v in output.items() if isinstance(v, torch.Tensor)}
        features.append(np.concatenate([out["context_latent"], out["belief_state"],
            out["predicted_target_latent"][:, idx], out["organ_latents"][:, organ], out["value"][:, idx, None],
            arrays["current"][batch], arrays["current"][batch, idx, None]], axis=1))
    return np.concatenate(features)


def prepare(frame, assets, arrays, rows, target):
    if len(rows) < 100:
        raise ValueError(f"insufficient {target} role rows: {len(rows)}")
    sub = frame.iloc[rows].reset_index(drop=True)
    history = trend_features(frame, rows, target)
    donors = shuffled_donors(sub.subject_id)
    shuffled = history[donors].copy()
    shuffled[:, [0, 3]] = history[:, [0, 3]]
    full = encode(assets, arrays, frame, rows, target)
    placebo = encode(assets, arrays, frame, rows, target, donors)
    return {"frame": sub, "features": {"direct": np.column_stack([full, history]),
            "shuffled_history": np.column_stack([placebo, shuffled]), "history_only": history},
            "history": history, "release": frozen_outputs(assets, arrays, frame, rows, f"{target}@3h")}


def predict_raw(data, heads, target):
    current = data["frame"][f"{target}_t"].to_numpy(float)
    result = {}
    for method in METHODS:
        if method in heads:
            result[method] = raw_prediction(heads[method], data["features"][method])
            result[method]["quantiles"] = result[method].pop("delta_quantiles") + current[:, None]
        else:
            point = (data["release"].point.to_numpy() if method == "release" else
                     current + 3 * data["history"][:, 1] if method == "linear_trend" else current.copy())
            q = (data["release"][["lower", "point", "upper"]].to_numpy() if method == "release"
                 else np.tile(point[:, None], (1, 3)))
            result[method] = {"quantiles": q, "risk_logit": np.abs(point - current)}
    return result


def calibrated_predictions(data, bundle, target):
    raw = predict_raw(data, bundle["heads"], target)
    risk = {m: bundle["probability"][m].predict(raw[m]["risk_logit"]) for m in METHODS}
    high = risk["direct"] > bundle["risk_boundary"]
    result = {}
    for m in METHODS:
        q = raw[m]["quantiles"]
        lo, hi = (q[:, 0], q[:, 2]) if m == "release" else cqr_bounds(q, high, bundle["radii"][m])
        result[m] = {"point": q[:, 1], "lower": lo, "upper": hi, "risk": risk[m],
                     "alert": risk[m] > bundle["cutoffs"][m]}
    return result, high


def evaluate(data, bundle, target, bootstrap):
    predictions, high = calibrated_predictions(data, bundle, target)
    return evaluate_predictions(data, predictions, high, bundle["cutoffs"], target, bootstrap)


def evaluate_predictions(data, predictions, high, cutoffs, target, bootstrap):
    """Score supplied frozen predictions without fitting or choosing a method."""
    sub = data["frame"]
    primary = representative_rows(sub, stable_hash("prospective_primary", target))
    f = sub.iloc[primary].reset_index(drop=True)
    current, truth = (f[c].to_numpy(float) for c in (f"{target}_t", f"future_{target}"))
    change = np.abs(truth - current) >= CHANGE_THRESHOLDS[target]
    result = {"primary_patients": len(f), "replay_rows": len(sub), "methods": {},
              "risk_group_definition": "source_validation_median_of_preanchor_direct_risk"}
    for m, full in predictions.items():
        pred = {k: v[primary] for k, v in full.items()}
        scored = pd.DataFrame({"subject_id": f.subject_id, "current": current, "truth": truth,
                              **{k: pred[k] for k in ("point", "lower", "upper")}})
        metrics = evaluate_change(scored, CHANGE_THRESHOLDS[target], repetitions=bootstrap)
        risk = {"brier": estimate(f, (pred["risk"] - change) ** 2, repetitions=bootstrap),
                "source_cutoff": cutoffs[m],
                "fpr_at_source_cutoff": estimate(f, np.where(~change, pred["alert"], np.nan), repetitions=bootstrap),
                "recall_at_source_cutoff": estimate(f, np.where(change, pred["alert"], np.nan), repetitions=bootstrap),
                "recall_at_matched_5pct_fpr_ranking_only": recall_at_fpr(change, pred["risk"], f.subject_id)}
        bins = {str(int(h)): evaluate_change(scored.loc[high[primary] == h], CHANGE_THRESHOLDS[target], repetitions=bootstrap)["all"]
                for h in (False, True)}
        units = {str(u): evaluate_change(scored.iloc[ix], CHANGE_THRESHOLDS[target], repetitions=bootstrap)["all"]
                 for u, ix in f.groupby("careunit").indices.items() if len(ix) >= 30 and str(u).lower() not in ("unknown", "nan", "none")}
        future = pd.to_datetime(sub[f"future_event_time_{target}"])
        lead = (future - sub.t).dt.total_seconds() / 3600
        changed = np.abs(sub[f"future_{target}"].to_numpy(float) - sub[f"{target}_t"].to_numpy(float)) >= CHANGE_THRESHOLDS[target]
        events = pd.DataFrame({"patient": sub.subject_id, "stay": sub.stay_id, "future": future,
            "changed": changed, "alert": changed & full["alert"],
            "lead": np.where(changed & full["alert"], lead, np.nan)})
        events = events.groupby(["patient", "stay", "future"]).agg(changed=("changed", "max"), alert=("alert", "max"), lead=("lead", "max"))
        positive = events.loc[events.changed]
        last = sub.groupby("stay_id")["_time_hours"].transform("max")
        late = sub["_time_hours"] >= .8 * last
        replay_scored = pd.DataFrame({"subject_id": sub.subject_id,
            "current": sub[f"{target}_t"], "truth": sub[f"future_{target}"],
            **{k: full[k] for k in ("point", "lower", "upper")}})
        result["methods"][m] = {"change": metrics, "risk": risk, "predicted_risk_bins": bins,
            "careunits": units, "replay": {
                "eligible_recorded_change_events": len(positive),
                "event_recall": float(positive.alert.mean()) if len(positive) else None,
                "median_lead_hours_to_measurement_not_onset": float(positive.lead.median()) if positive.lead.notna().any() else None,
                "late_stay": evaluate_change(replay_scored.loc[late], CHANGE_THRESHOLDS[target], repetitions=bootstrap)["all"],
                "onset_detection_validated": False}}
    direct_error = np.abs(predictions["direct"]["point"][primary] - truth)
    comparisons = {}
    for m in predictions:
        if m == "direct":
            continue
        delta = direct_error - np.abs(predictions[m]["point"][primary] - truth)
        folds = []
        for seed in SEEDS:
            _, mask = patient_split(f.subject_id, seed)
            folds.append(estimate(f.loc[mask].reset_index(drop=True), delta[mask], seed=seed, repetitions=bootstrap))
        comparisons[m] = {"paired_mae": estimate(f, delta, repetitions=bootstrap), "seven_evaluation_resamples": folds,
            "paired_recall_at_frozen_cutoffs": estimate(f, np.where(change,
                predictions["direct"]["alert"][primary].astype(float) - predictions[m]["alert"][primary], np.nan), repetitions=bootstrap)}
    result["direct_comparisons"] = comparisons
    d = result["methods"]["direct"]
    ci_pass = all(c["paired_mae"]["ci95"] is not None and c["paired_mae"]["ci95"][1] < 0
                  and all(s["ci95"] is not None and s["ci95"][1] < 0 for s in c["seven_evaluation_resamples"]) for c in comparisons.values())
    stable = d["change"]["stable"]["mae_delta_vs_persistence"]
    result["diagnostic_screen_pass"] = bool(ci_pass and stable["ci95"] is not None
        and stable["ci95"][1] <= .05 * CHANGE_THRESHOLDS[target]
        and d["change"]["all"]["coverage_status"] == "within_nominal_band"
        and all(v["coverage_status"] == "within_nominal_band" for v in d["predicted_risk_bins"].values())
        and d["risk"]["fpr_at_source_cutoff"]["mean"] is not None and d["risk"]["fpr_at_source_cutoff"]["mean"] <= .05)
    return result


def valid_rows(frame, target):
    tolerance = .5 if target == "map" else 1.
    dt = pd.to_numeric(frame[f"future_delta_t_hr_{target}"], errors="coerce")
    return label_mask(frame, target) & frame[f"{target}_age_hr"].between(0, 1).to_numpy() & (np.abs(dt - 3) <= tolerance).to_numpy()


def write(report, output):
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run(args):
    torch.set_num_threads(4)
    assets = load_release(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    bindings = {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}
    report = {"schema": "prospective_change.v1", "release_binding": bindings,
        "protocol_sha256": sha256(Path("PROSPECTIVE_CHANGE_PROTOCOL_20260906.md")),
        "code_sha256": {p: sha256(Path(p)) for p in ("scripts/audit_prospective_change.py", "whole_body_jepa/prospective_change.py")},
        "inputs": {"eicu": sha256(args.eicu)}, "complete": False, "promotion_allowed": False,
        "clinical_claim_allowed": False, "causal_claim_allowed": False, "runtime_modified": False,
        "untouched_lockbox": False, "training_seeds": [20260906], "targets": {}}
    print("read eICU; fit roles are patient/hospital-disjoint", flush=True)
    frame = read_examples(args.eicu, assets, "eicu", 3, grid=True)
    masks = role_masks(frame)
    patient_sets = {r: set(frame.loc[m, "subject_id"]) for r, m in masks.items()}
    if any(patient_sets[a] & patient_sets[b] for a in masks for b in masks if a < b):
        raise ValueError("patient roles overlap")
    report["role_subject_digests"] = {r: subject_digest(s) for r, s in patient_sets.items()}
    arrays = _build_inputs(input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours), assets, 3)
    bundles = {}
    for target in TARGETS:
        print(f"prepare {target} source roles", flush=True)
        valid = valid_rows(frame, target)
        data = {}
        for role, mask in masks.items():
            rows = select_role_rows(frame, valid, mask, patients=5000 if role in ("fit", "validation", "calibration") else args.max_test_patients,
                                    anchors=6 if role == "fit" else 1 if role in ("validation", "calibration") else 0,
                                    salt=f"{target}:{role}")
            print(json.dumps({"target": target, "role": role, "rows": len(rows), "patients": int(frame.iloc[rows].subject_id.nunique())}), flush=True)
            data[role] = prepare(frame, assets, arrays, rows, target)
        fit, val, cal = (data[r] for r in ("fit", "validation", "calibration"))
        delta = (fit["frame"][f"future_{target}"] - fit["frame"][f"{target}_t"]).to_numpy(float)
        print(f"fit {target} direct / shuffled / history-only quantiles and risk", flush=True)
        heads = {m: fit_head(fit["features"][m], delta, fit["frame"].subject_id, CHANGE_THRESHOLDS[target])
                 for m in ("direct", "shuffled_history", "history_only")}
        vr = predict_raw(val, heads, target)
        labels = np.abs(val["frame"][f"future_{target}"] - val["frame"][f"{target}_t"]).to_numpy() >= CHANGE_THRESHOLDS[target]
        probability = {m: calibrate_risk(vr[m]["risk_logit"], labels, val["frame"].subject_id) for m in METHODS}
        risks = {m: probability[m].predict(vr[m]["risk_logit"]) for m in METHODS}
        boundary = float(np.median(risks["direct"]))
        cutoffs = {m: alarm_cutoff(risks[m], labels) for m in METHODS}
        cr = predict_raw(cal, heads, target)
        high = probability["direct"].predict(cr["direct"]["risk_logit"]) > boundary
        radii = {m: fit_risk_cqr(cal["frame"][f"future_{target}"].to_numpy(float), cr[m]["quantiles"], high)
                 for m in METHODS if m != "release"}
        bundle = {"heads": heads, "probability": probability, "risk_boundary": boundary, "cutoffs": cutoffs, "radii": radii}
        bundles[target] = bundle
        item = {"support": {r: {"rows": len(d["frame"]), "patients": int(d["frame"].subject_id.nunique())} for r, d in data.items()},
                "risk_boundary": boundary, "cutoffs": cutoffs, "radii": radii, "evaluation": {}}
        report["targets"][target] = item
        for role in ("patient_test", "hospital_test"):
            item["evaluation"][role] = evaluate(data[role], bundle, target, args.bootstrap)
        write(report, args.output)
        del data
    joblib.dump({"status": "candidate_only", "release_binding": bindings, "bundles": bundles}, args.output / "candidate_heads.joblib")
    (args.output / "development.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["candidate_sha256_before_external"] = sha256(args.output / "candidate_heads.joblib")
    del frame, arrays
    print("frozen artifacts saved; now read NWICU", flush=True)
    report["inputs"]["nwicu"] = sha256(args.nwicu)
    frame = read_examples(args.nwicu, assets, "nwicu", 3)
    arrays = _build_inputs(input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours), assets, 3)
    for target in TARGETS:
        rows = select_role_rows(frame, valid_rows(frame, target), np.ones(len(frame), dtype=bool), patients=args.max_test_patients, anchors=0, salt=f"{target}:nwicu")
        print(json.dumps({"target": target, "external_rows": len(rows), "patients": int(frame.iloc[rows].subject_id.nunique())}), flush=True)
        data = prepare(frame, assets, arrays, rows, target)
        result = evaluate(data, bundles[target], target, args.bootstrap)
        report["targets"][target]["evaluation"]["nwicu"] = result
        print(json.dumps({"target": target, "screen_pass": result["diagnostic_screen_pass"],
            "direct_mae": result["methods"]["direct"]["change"]["all"]["mae"]["mean"],
            "release_mae": result["methods"]["release"]["change"]["all"]["mae"]["mean"]}), flush=True)
        write(report, args.output)
    if bindings != {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}:
        raise ValueError("release changed")
    if report["candidate_sha256_before_external"] != sha256(args.output / "candidate_heads.joblib"):
        raise ValueError("candidate changed after external outcomes")
    report["complete"] = True
    write(report, args.output)
    print("FINISHED; runtime unchanged", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--eicu", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet"))
    p.add_argument("--nwicu", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    p.add_argument("--max-test-patients", type=int, default=500)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("artifacts/prospective_change_20260906"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
