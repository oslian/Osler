"""Test observation semantics, representation loss, and separate capability gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.audit_change_behavior import frozen_outputs
from scripts.audit_prospective_change import (
    encode,
    role_masks,
    select_role_rows,
    shuffled_donors,
    valid_rows,
)
from scripts.audit_release_masked_cells import input_frame, read_examples
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS
from whole_body_jepa.multisource_intervals import conformal_radius, representative_rows, sha256, stable_hash, subject_digest
from whole_body_jepa.probability_calibration import fit_probability_calibrator
from whole_body_jepa.prospective_change import alarm_cutoff, fit_head, raw_prediction, trend_features
from whole_body_jepa.representation_repair import (
    interval_gate, observation_feature_blocks, point_gate, risk_gate,
)
from whole_body_jepa.runtime import _build_inputs, load_release


TARGETS = ("glucose", "map")
LEARNED = ("history", "raw", "raw_placebo", "aware_raw", "aware_raw_placebo",
           "latent", "latent_placebo", "aware_latent", "aware_latent_placebo")
CANDIDATES = ("raw", "aware_raw", "latent", "aware_latent")
PLACEBO = {"raw": "raw_placebo", "aware_raw": "aware_raw_placebo",
           "latent": "latent_placebo", "aware_latent": "aware_latent_placebo"}


def observation_arrays(frame, assets):
    filtered = input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours)
    return assets.scaler.normalize_current(filtered)


def prepare(frame, assets, arrays, observed_arrays, rows, target):
    if len(rows) < 100:
        raise ValueError(f"insufficient {target} role rows: {len(rows)}")
    sub = frame.iloc[rows].reset_index(drop=True)
    own_history = trend_features(frame, rows, target)
    donors = shuffled_donors(sub.subject_id)
    current, mask, ages = observed_arrays
    raw_all, aware_all = observation_feature_blocks(
        current, mask, ages, assets.checkpoint["variables"], target
    )
    raw = raw_all[rows]
    aware = aware_all[rows]
    latent = encode(assets, arrays, frame, rows, target)
    features = {
        "history": own_history,
        "raw": np.column_stack([raw, own_history]),
        "raw_placebo": np.column_stack([raw[donors], own_history]),
        "aware_raw": np.column_stack([aware, own_history]),
        "aware_raw_placebo": np.column_stack([aware[donors], own_history]),
        "latent": np.column_stack([latent, own_history]),
        "latent_placebo": np.column_stack([latent[donors], own_history]),
        "aware_latent": np.column_stack([latent, aware, own_history]),
        "aware_latent_placebo": np.column_stack([latent[donors], aware[donors], own_history]),
    }
    return {
        "frame": sub,
        "features": features,
        "release": frozen_outputs(assets, arrays, frame, rows, f"{target}@3h"),
    }


def fit_bundle(data, target):
    fit, validation, calibration = (data[name] for name in ("fit", "validation", "calibration"))
    delta = (fit["frame"][f"future_{target}"] - fit["frame"][f"{target}_t"]).to_numpy(float)
    threshold = CHANGE_THRESHOLDS[target]
    heads = {}
    probabilities = {}
    cutoffs = {}
    radii = {}
    validation_labels = np.abs(
        validation["frame"][f"future_{target}"].to_numpy(float)
        - validation["frame"][f"{target}_t"].to_numpy(float)
    ) >= threshold
    for method in LEARNED:
        print(f"fit {target} {method}", flush=True)
        head = fit_head(fit["features"][method], delta, fit["frame"].subject_id, threshold)
        heads[method] = head
        validation_raw = raw_prediction(head, validation["features"][method])
        calibrator = fit_probability_calibrator(
            "platt", validation_raw["risk_logit"], validation_labels, validation["frame"].subject_id
        )
        probabilities[method] = calibrator
        validation_risk = calibrator.predict(validation_raw["risk_logit"])
        cutoffs[method] = alarm_cutoff(validation_risk, validation_labels)
        calibration_raw = raw_prediction(head, calibration["features"][method])
        current = calibration["frame"][f"{target}_t"].to_numpy(float)
        quantiles = calibration_raw["delta_quantiles"] + current[:, None]
        truth = calibration["frame"][f"future_{target}"].to_numpy(float)
        errors = np.maximum.reduce([quantiles[:, 0] - truth, truth - quantiles[:, 2], np.zeros(len(truth))])
        radii[method] = float(conformal_radius(errors))
    return {"heads": heads, "probabilities": probabilities, "cutoffs": cutoffs, "radii": radii}


def predict(data, bundle, target):
    current = data["frame"][f"{target}_t"].to_numpy(float)
    output = {}
    for method in LEARNED:
        raw = raw_prediction(bundle["heads"][method], data["features"][method])
        q = raw["delta_quantiles"] + current[:, None]
        radius = bundle["radii"][method]
        output[method] = {
            "point": q[:, 1],
            "lower": q[:, 0] - radius,
            "upper": q[:, 2] + radius,
            "risk": bundle["probabilities"][method].predict(raw["risk_logit"]),
        }
    release = data["release"]
    output["release"] = {
        "point": release.point.to_numpy(float),
        "lower": release.lower.to_numpy(float),
        "upper": release.upper.to_numpy(float),
        "risk": np.abs(release.point.to_numpy(float) - current),
    }
    return output


def evaluate(data, bundle, target, repetitions):
    selected = representative_rows(data["frame"], stable_hash("representation_repair", target))
    frame = data["frame"].iloc[selected].reset_index(drop=True)
    predictions = {
        name: {key: value[selected] for key, value in payload.items()}
        for name, payload in predict(data, bundle, target).items()
    }
    truth = frame[f"future_{target}"].to_numpy(float)
    current = frame[f"{target}_t"].to_numpy(float)
    labels = np.abs(truth - current) >= CHANGE_THRESHOLDS[target]
    methods = {}
    for candidate in CANDIDATES:
        references = ("history", "release", PLACEBO[candidate])
        methods[candidate] = {
            "mae": float(np.mean(np.abs(predictions[candidate]["point"] - truth))),
            "point": point_gate(frame, predictions, truth, current, candidate, references,
                                CHANGE_THRESHOLDS[target], repetitions=repetitions),
            "risk": risk_gate(frame, predictions, labels, candidate, references,
                              bundle["cutoffs"][candidate], repetitions=repetitions),
            "interval": interval_gate(frame, predictions, truth, candidate, references,
                                      CHANGE_THRESHOLDS[target], repetitions=repetitions),
        }
    baselines = {name: float(np.mean(np.abs(predictions[name]["point"] - truth)))
                 for name in ("history", "release", *PLACEBO.values())}
    return {"patients": len(frame), "methods": methods, "baseline_mae": baselines}


def write(report, path):
    (path / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    bindings = {path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()}
    report = {
        "schema": "representation_repair.v1",
        "complete": False,
        "status": "candidate_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "untouched_lockbox": False,
        "release_binding": bindings,
        "protocol_sha256": sha256(Path("REPRESENTATION_REPAIR_PROTOCOL_20260907.md")),
        "inputs": {"eicu": sha256(args.eicu)},
        "targets": {},
    }
    print("read eICU", flush=True)
    frame = read_examples(args.eicu, assets, "eicu", 3, grid=True)
    roles = role_masks(frame)
    subjects = {name: set(frame.loc[mask, "subject_id"]) for name, mask in roles.items()}
    if any(subjects[a] & subjects[b] for a in subjects for b in subjects if a < b):
        raise ValueError("patient roles overlap")
    report["role_subject_digests"] = {name: subject_digest(values) for name, values in subjects.items()}
    arrays = _build_inputs(input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours), assets, 3)
    observed = observation_arrays(frame, assets)
    bundles = {}
    for target in TARGETS:
        role_data = {}
        for role, mask in roles.items():
            rows = select_role_rows(
                frame, valid_rows(frame, target), mask,
                patients=5000 if role in {"fit", "validation", "calibration"} else args.max_test_patients,
                anchors=6 if role == "fit" else 1 if role in {"validation", "calibration"} else 0,
                salt=f"representation:{target}:{role}",
            )
            print(json.dumps({"target": target, "role": role, "rows": len(rows),
                              "patients": int(frame.iloc[rows].subject_id.nunique())}), flush=True)
            role_data[role] = prepare(frame, assets, arrays, observed, rows, target)
        bundles[target] = fit_bundle(role_data, target)
        report["targets"][target] = {"evaluation": {}}
        for role in ("patient_test", "hospital_test"):
            report["targets"][target]["evaluation"][role] = evaluate(
                role_data[role], bundles[target], target, args.bootstrap
            )
        write(report, args.output)
    candidate = args.output / "candidate_heads.joblib"
    joblib.dump({"status": "candidate_only", "release_binding": bindings, "bundles": bundles}, candidate)
    report["candidate_sha256_before_external"] = sha256(candidate)
    (args.output / "development.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    del frame, arrays, observed

    print("candidate frozen; read NWICU", flush=True)
    report["inputs"]["nwicu"] = sha256(args.nwicu)
    frame = read_examples(args.nwicu, assets, "nwicu", 3)
    arrays = _build_inputs(input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours), assets, 3)
    observed = observation_arrays(frame, assets)
    for target in TARGETS:
        rows = select_role_rows(frame, valid_rows(frame, target), np.ones(len(frame), dtype=bool),
                                patients=args.max_test_patients, anchors=0, salt=f"representation:{target}:nwicu")
        data = prepare(frame, assets, arrays, observed, rows, target)
        result = evaluate(data, bundles[target], target, args.bootstrap)
        report["targets"][target]["evaluation"]["nwicu"] = result
        print(json.dumps({"target": target, "patients": result["patients"],
            "methods": {name: {"mae": item["mae"], "point": item["point"]["pass"],
                "risk": item["risk"]["pass"], "interval": item["interval"]["pass"]}
                for name, item in result["methods"].items()}}), flush=True)
        write(report, args.output)
    if bindings != {path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()}:
        raise ValueError("release changed during audit")
    if report["candidate_sha256_before_external"] != sha256(candidate):
        raise ValueError("candidate changed after external evaluation")
    report["complete"] = True
    write(report, args.output)
    print("FINISHED; release unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--eicu", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet"))
    parser.add_argument("--nwicu", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/representation_repair_20260907"))
    parser.add_argument("--max-test-patients", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
