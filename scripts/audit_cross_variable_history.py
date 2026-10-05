"""Test whether raw non-target history adds future signal beyond own history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.audit_change_behavior import frozen_outputs
from scripts.audit_prospective_change import (
    role_masks,
    select_role_rows,
    shuffled_donors,
    valid_rows,
)
from scripts.audit_release_masked_cells import input_frame, read_examples
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS
from whole_body_jepa.cross_variable_history import cross_variable_history_features
from whole_body_jepa.multisource_intervals import (
    conformal_radius,
    representative_rows,
    sha256,
    stable_hash,
    subject_digest,
)
from whole_body_jepa.probability_calibration import fit_probability_calibrator
from whole_body_jepa.prospective_change import (
    alarm_cutoff,
    fit_head,
    raw_prediction,
    trend_features,
)
from whole_body_jepa.representation_repair import interval_gate, point_gate, risk_gate
from whole_body_jepa.runtime import HISTORY_LAGS_HOURS, _build_inputs, load_release


TARGETS = ("glucose", "map")
LEARNED = ("history", "current_cross", "cross_history", "cross_history_placebo")
REFERENCES = ("history", "current_cross", "release", "cross_history_placebo")


def observed_arrays(frame, assets):
    filtered = input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours)
    return assets.scaler.normalize_current(filtered)


def prepare(frame, assets, arrays, observations, rows, target):
    if len(rows) < 100:
        raise ValueError(f"insufficient {target} role rows: {len(rows)}")
    sub = frame.iloc[rows].reset_index(drop=True)
    own_history = trend_features(frame, rows, target)
    current_cross, cross_history = cross_variable_history_features(
        frame,
        rows,
        variables=assets.checkpoint["variables"],
        target=target,
        normalized=observations[0],
        observed=observations[1],
        ages=observations[2],
        history_indices=arrays["history"],
        history_lags_hours=HISTORY_LAGS_HOURS,
    )
    donors = shuffled_donors(sub.subject_id)
    features = {
        "history": own_history,
        "current_cross": np.column_stack([own_history, current_cross]),
        "cross_history": np.column_stack([own_history, cross_history]),
        "cross_history_placebo": np.column_stack([own_history, cross_history[donors]]),
    }
    return {
        "frame": sub,
        "features": features,
        "release": frozen_outputs(assets, arrays, frame, rows, f"{target}@3h"),
    }


def fit_bundle(data, target):
    fit, validation, calibration = (data[name] for name in ("fit", "validation", "calibration"))
    delta = (
        fit["frame"][f"future_{target}"] - fit["frame"][f"{target}_t"]
    ).to_numpy(float)
    threshold = CHANGE_THRESHOLDS[target]
    validation_delta = (
        validation["frame"][f"future_{target}"] - validation["frame"][f"{target}_t"]
    ).to_numpy(float)
    validation_labels = np.abs(validation_delta) >= threshold
    heads, probabilities, cutoffs, radii = {}, {}, {}, {}
    for method in LEARNED:
        print(f"fit {target} {method}", flush=True)
        head = fit_head(
            fit["features"][method], delta, fit["frame"].subject_id, threshold
        )
        heads[method] = head
        validation_raw = raw_prediction(head, validation["features"][method])
        calibrator = fit_probability_calibrator(
            "platt",
            validation_raw["risk_logit"],
            validation_labels,
            validation["frame"].subject_id,
        )
        probabilities[method] = calibrator
        validation_risk = calibrator.predict(validation_raw["risk_logit"])
        cutoffs[method] = alarm_cutoff(validation_risk, validation_labels)
        calibration_raw = raw_prediction(head, calibration["features"][method])
        current = calibration["frame"][f"{target}_t"].to_numpy(float)
        quantiles = calibration_raw["delta_quantiles"] + current[:, None]
        truth = calibration["frame"][f"future_{target}"].to_numpy(float)
        errors = np.maximum.reduce(
            [quantiles[:, 0] - truth, truth - quantiles[:, 2], np.zeros(len(truth))]
        )
        radii[method] = float(conformal_radius(errors))
    return {"heads": heads, "probabilities": probabilities, "cutoffs": cutoffs, "radii": radii}


def predict(data, bundle, target):
    current = data["frame"][f"{target}_t"].to_numpy(float)
    output = {}
    for method in LEARNED:
        raw = raw_prediction(bundle["heads"][method], data["features"][method])
        quantiles = raw["delta_quantiles"] + current[:, None]
        radius = bundle["radii"][method]
        output[method] = {
            "point": quantiles[:, 1],
            "lower": quantiles[:, 0] - radius,
            "upper": quantiles[:, 2] + radius,
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
    selected = representative_rows(data["frame"], stable_hash("cross_variable_history", target))
    frame = data["frame"].iloc[selected].reset_index(drop=True)
    predictions = {
        name: {key: value[selected] for key, value in payload.items()}
        for name, payload in predict(data, bundle, target).items()
    }
    truth = frame[f"future_{target}"].to_numpy(float)
    current = frame[f"{target}_t"].to_numpy(float)
    labels = np.abs(truth - current) >= CHANGE_THRESHOLDS[target]
    candidate = "cross_history"
    result = {
        "patients": len(frame),
        "mae": {
            name: float(np.mean(np.abs(payload["point"] - truth)))
            for name, payload in predictions.items()
        },
        "point": point_gate(
            frame,
            predictions,
            truth,
            current,
            candidate,
            REFERENCES,
            CHANGE_THRESHOLDS[target],
            repetitions=repetitions,
        ),
        "risk": risk_gate(
            frame,
            predictions,
            labels,
            candidate,
            REFERENCES,
            bundle["cutoffs"][candidate],
            repetitions=repetitions,
        ),
        "interval": interval_gate(
            frame,
            predictions,
            truth,
            candidate,
            REFERENCES,
            CHANGE_THRESHOLDS[target],
            repetitions=repetitions,
        ),
    }
    result["all_capability_gates_pass"] = bool(
        result["point"]["pass"] and result["risk"]["pass"] and result["interval"]["pass"]
    )
    return result


def write(report, output):
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    bindings = {path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()}
    report = {
        "schema": "cross_variable_history.v1",
        "complete": False,
        "status": "candidate_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "untouched_lockbox": False,
        "release_binding": bindings,
        "protocol_sha256": sha256(Path("CROSS_VARIABLE_HISTORY_PROTOCOL_20260907.md")),
        "inputs": {"eicu": sha256(args.eicu)},
        "history_lags_hours": [24.0, 12.0, 6.0, 3.0, 1.0],
        "targets": {},
    }

    print("read eICU; fit roles are patient/hospital-disjoint", flush=True)
    frame = read_examples(args.eicu, assets, "eicu", 3, grid=True)
    roles = role_masks(frame)
    subjects = {name: set(frame.loc[mask, "subject_id"]) for name, mask in roles.items()}
    if any(subjects[a] & subjects[b] for a in subjects for b in subjects if a < b):
        raise ValueError("patient roles overlap")
    report["role_subject_digests"] = {
        name: subject_digest(values) for name, values in subjects.items()
    }
    filtered = input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours)
    arrays = _build_inputs(filtered, assets, 3)
    observations = observed_arrays(frame, assets)
    bundles = {}
    for target in TARGETS:
        role_data = {}
        for role, mask in roles.items():
            rows = select_role_rows(
                frame,
                valid_rows(frame, target),
                mask,
                patients=5000 if role in {"fit", "validation", "calibration"} else args.max_test_patients,
                anchors=6 if role == "fit" else 1 if role in {"validation", "calibration"} else 0,
                salt=f"cross-history:{target}:{role}",
            )
            print(
                json.dumps(
                    {
                        "target": target,
                        "role": role,
                        "rows": len(rows),
                        "patients": int(frame.iloc[rows].subject_id.nunique()),
                    }
                ),
                flush=True,
            )
            role_data[role] = prepare(frame, assets, arrays, observations, rows, target)
        bundles[target] = fit_bundle(role_data, target)
        evaluations = {
            role: evaluate(role_data[role], bundles[target], target, args.bootstrap)
            for role in ("patient_test", "hospital_test")
        }
        report["targets"][target] = {
            "feature_widths": {
                name: int(role_data["fit"]["features"][name].shape[1]) for name in LEARNED
            },
            "evaluation": evaluations,
            "source_point_gate_pass": bool(
                evaluations["patient_test"]["point"]["pass"]
                and evaluations["hospital_test"]["point"]["pass"]
            ),
        }
        write(report, args.output)

    candidate = args.output / "candidate_heads.joblib"
    joblib.dump(
        {"status": "candidate_only", "release_binding": bindings, "bundles": bundles},
        candidate,
    )
    report["candidate_sha256_before_external"] = sha256(candidate)
    (args.output / "development.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    del frame, arrays, observations

    print("candidate frozen; read NWICU", flush=True)
    report["inputs"]["nwicu"] = sha256(args.nwicu)
    frame = read_examples(args.nwicu, assets, "nwicu", 3)
    filtered = input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours)
    arrays = _build_inputs(filtered, assets, 3)
    observations = observed_arrays(frame, assets)
    for target in TARGETS:
        rows = select_role_rows(
            frame,
            valid_rows(frame, target),
            np.ones(len(frame), dtype=bool),
            patients=args.max_test_patients,
            anchors=0,
            salt=f"cross-history:{target}:nwicu",
        )
        data = prepare(frame, assets, arrays, observations, rows, target)
        result = evaluate(data, bundles[target], target, args.bootstrap)
        report["targets"][target]["evaluation"]["nwicu"] = result
        print(
            json.dumps(
                {
                    "target": target,
                    "patients": result["patients"],
                    "mae": result["mae"],
                    "point": result["point"]["pass"],
                    "risk": result["risk"]["pass"],
                    "interval": result["interval"]["pass"],
                }
            ),
            flush=True,
        )
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
    parser.add_argument(
        "--eicu",
        type=Path,
        default=Path(
            "/Users/chunyouchang/Desktop/llm_project/medical_jepa/"
            "eicu_canonical_event_examples_full_5000.parquet"
        ),
    )
    parser.add_argument(
        "--nwicu", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cross_variable_history_20260907")
    )
    parser.add_argument("--max-test-patients", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

