"""Compare own history, raw cross-history, and frozen JEPA on aligned ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from scripts.audit_prospective_change import encode, select_role_rows
from scripts.audit_release_masked_cells import input_frame, read_examples
from scripts.screen_cross_variable_history_catalog import (
    comparison,
    fit_regressor,
    target_role_masks,
    valid_rows,
)
from whole_body_jepa.cross_variable_history import (
    cross_variable_history_features,
    cross_variable_history_views,
    matched_wrong_patient_rows,
)
from whole_body_jepa.multisource_intervals import (
    representative_rows,
    sha256,
    stable_hash,
    subject_digest,
)
from whole_body_jepa.prospective_change import trend_features
from whole_body_jepa.runtime import HISTORY_LAGS_HOURS, _build_inputs, load_release


TARGET = "creatinine"
HORIZON = 6
METHODS = (
    "own_history",
    "current_cross",
    "current_cross_placebo",
    "raw_cross_history",
    "temporal_placebo",
    "full_cross_placebo",
    "latent_only",
    "latent_only_placebo",
    "release_features",
    "release_features_placebo",
)
COMPARISONS = {
    "current_cross": ("own_history", "current_cross_placebo", "persistence"),
    "raw_cross_history": ("own_history", "current_cross", "temporal_placebo", "full_cross_placebo", "persistence"),
    "latent_only": ("own_history", "latent_only_placebo", "persistence"),
    "release_features": ("own_history", "release_features_placebo", "persistence"),
}


def feature_views(own, current, full, bundle, donors, contract):
    raw = cross_variable_history_views(own, current, full, donors)
    if bundle.shape[1] != contract["feature_width"]:
        raise ValueError("release feature contract width mismatch")
    names = ("shared_context", "patient_belief", "target_future", "target_organ")
    blocks = {b["name"]: b for b in contract["blocks"]}
    latent = np.concatenate([bundle[:, blocks[n]["start"]:blocks[n]["stop"]] for n in names], axis=1)
    views = {
        "own_history": own,
        "current_cross": raw["current_cross"],
        "current_cross_placebo": raw["current_cross_placebo"],
        "raw_cross_history": raw["cross_history"],
        "temporal_placebo": raw["cross_history_placebo"],
        "full_cross_placebo": np.column_stack([own, full[donors]]),
        "latent_only": np.column_stack([own, latent]),
        "latent_only_placebo": np.column_stack([own, latent[donors]]),
        "release_features": np.column_stack([own, bundle]),
        "release_features_placebo": np.column_stack([own, bundle[donors]]),
    }
    return {k: v.astype(np.float32) for k, v in views.items()}


def window_mask(frame, bounds):
    delta = pd.to_numeric(frame[f"future_delta_t_hr_{TARGET}"], errors="coerce").to_numpy(float)
    event = pd.to_datetime(frame[f"future_event_time_{TARGET}"], errors="coerce", utc=True)
    anchor = pd.to_datetime(frame["t"], errors="coerce", utc=True)
    actual = (event - anchor).dt.total_seconds().to_numpy() / 3600.0
    return np.isfinite(delta) & np.isfinite(actual) & np.isclose(delta, actual, atol=1e-6) & (delta >= bounds[0]) & (delta <= bounds[1])


def prepare(frame, rows, assets, arrays, observations, contract):
    selected = frame.iloc[rows].reset_index(drop=True)
    own = trend_features(frame, rows, TARGET)
    current, full = cross_variable_history_features(
        frame,
        rows,
        variables=assets.checkpoint["variables"],
        target=TARGET,
        normalized=observations[0],
        observed=observations[1],
        ages=observations[2],
        history_indices=arrays["history"],
        history_lags_hours=HISTORY_LAGS_HOURS,
    )
    donors = matched_wrong_patient_rows(
        selected,
        seed=stable_hash("anchor_aligned_signal_donor", TARGET),
    )
    bundle = encode(assets, arrays, frame, rows, TARGET)
    return selected, feature_views(own, current, full, bundle, donors, contract)


def predict(features, models, current):
    return {
        method: current + models[method].predict(features[method])
        for method in METHODS
    } | {"persistence": current.copy()}


def pair_result(frame, predictions, truth, candidate, reference, role, bootstrap):
    return comparison(
        frame,
        predictions,
        truth,
        candidate,
        reference,
        salt=f"anchor-aligned:{TARGET}:{role}:{candidate}:{reference}",
        repetitions=bootstrap,
    )


def careunit_results(frame, predictions, truth, candidate, reference, bootstrap):
    results = {}
    for unit, positions in frame.groupby("careunit", sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        sub = frame.iloc[positions].reset_index(drop=True)
        if str(unit).lower() in {"", "nan", "none", "unknown"} or sub.subject_id.nunique() < 30:
            continue
        result = pair_result(
            sub,
            {name: values[positions] for name, values in predictions.items()},
            truth[positions],
            candidate,
            reference,
            f"careunit:{unit}",
            bootstrap,
        )
        results[str(unit)] = result["paired_patient"]
    return results


def evaluate(frame, features, models, role, bootstrap):
    selected = representative_rows(
        frame, stable_hash("anchor_aligned_signal_representative", TARGET, role)
    )
    work = frame.iloc[selected].reset_index(drop=True)
    selected_features = {name: values[selected] for name, values in features.items()}
    current = work[f"{TARGET}_t"].to_numpy(float)
    truth = work[f"future_{TARGET}"].to_numpy(float)
    predictions = predict(selected_features, models, current)
    comparisons = {}
    for candidate, references in COMPARISONS.items():
        candidate_results = {
            reference: pair_result(
                work, predictions, truth, candidate, reference, role, bootstrap
            )
            for reference in references
        }
        careunits = {
            reference: careunit_results(
                work, predictions, truth, candidate, reference, bootstrap
            )
            for reference in references
        }
        comparisons[candidate] = {
            "comparisons": candidate_results,
            "overall_pass": bool(all(item["pass"] for item in candidate_results.values())),
            "careunits": careunits,
            "all_supported_careunits_pass": bool(
                careunits["own_history"]
                and all(
                    item["pass"]
                    for reference in careunits.values()
                    for item in reference.values()
                )
            ),
        }
    actual_delta = truth - current
    return {
        "patients": int(work.subject_id.nunique()),
        "timing": {
            col: {str(q): float(work[col].quantile(q)) for q in (0, .1, .5, .9, 1)}
            for col in (f"future_delta_t_hr_{TARGET}", f"{TARGET}_age_hr")
        },
        "mae": {
            name: float(np.mean(np.abs(values - truth)))
            for name, values in predictions.items()
        },
        "actual_delta": {
            "median": float(np.median(actual_delta)),
            "median_absolute": float(np.median(np.abs(actual_delta))),
            "p90_absolute": float(np.quantile(np.abs(actual_delta), 0.90)),
        },
        "methods": comparisons,
    }


def external_interpretation(result):
    return {
        "method_diagnostic_pass": {
            name: bool(item["overall_pass"] and item["all_supported_careunits_pass"])
            for name, item in result["methods"].items()
        },
        "conclusion": "bounded_readout_evidence_only_not_information_absence_or_encoder_loss_proof",
        "promotion_allowed": False,
    }


def load_dataset(path, database, assets):
    frame = read_examples(path, assets, database, HORIZON, grid=True)
    filtered = input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours)
    arrays = _build_inputs(filtered, assets, HORIZON)
    observations = assets.scaler.normalize_current(filtered)
    return frame, arrays, observations


def write_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    if args.output.exists() or args.fitted_output.exists():
        raise ValueError("refusing to overwrite existing experiment artifacts")
    protocol = Path("ANCHOR_ALIGNED_SIGNAL_CORRECTION_PROTOCOL_20260908.md")
    contract_path = args.model_dir / "FEATURE_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    bounds = (5.0, 7.0) if args.window == "narrow" else (2.0, 10.0)
    report = {
        "schema": "whole_body_jepa.anchor_aligned_signal.v2",
        "complete": False,
        "status": "source_attribution_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "target": TARGET,
        "horizon_hours": HORIZON,
        "protocol_sha256": sha256(protocol),
        "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": sha256(args.model_dir / "whole_body_joint_jepa.pt"),
        "feature_contract_sha256": sha256(contract_path),
        "actual_future_window_hours": bounds,
        "window": args.window,
        "split_note": "same broad-support hospital/patient roles for both windows; seven resamples, not seven independent fits",
        "encoder_holdout_exclusion_verified": False,
        "inputs": {"eicu": sha256(args.eicu), "mimic": sha256(args.mimic)},
        "data_contract": {
            "lookback_hours": 6,
            "anchor_step_hours": 3,
            "maximum_episode_hours": 96,
            "strict_real_future_measurement_only": True,
            "treatments_excluded": True,
        },
        "evaluation": {},
    }

    print("read aligned eICU", flush=True)
    frame, arrays, observations = load_dataset(args.eicu, "eicu", assets)
    eligible = valid_rows(frame, TARGET, assets.max_age_hours)
    roles, hospital_role = target_role_masks(frame, eligible, f"{TARGET}@{HORIZON}h")
    eligible &= window_mask(frame, bounds)
    role_data = {}
    for role in ("fit", "patient_test", "hospital_test"):
        rows = select_role_rows(
            frame,
            eligible,
            roles[role],
            patients=args.max_fit_patients if role == "fit" else args.max_test_patients,
            anchors=args.fit_anchors if role == "fit" else 0,
            salt=f"anchor-aligned:{TARGET}:{role}",
        )
        if len(rows) < args.min_rows or frame.iloc[rows].subject_id.nunique() < args.min_patients:
            raise ValueError(f"insufficient eICU {role} support")
        role_data[role] = prepare(frame, rows, assets, arrays, observations, contract)
        print(json.dumps({"role": role, "rows": len(rows), "patients": int(role_data[role][0].subject_id.nunique())}), flush=True)
    fit_frame, fit_features = role_data["fit"]
    subject_sets = {r: set(data[0].subject_id) for r, data in role_data.items()}
    if any(subject_sets[a] & subject_sets[b] for a, b in (("fit", "patient_test"), ("fit", "hospital_test"), ("patient_test", "hospital_test"))):
        raise ValueError("patient roles overlap")
    delta = (fit_frame[f"future_{TARGET}"] - fit_frame[f"{TARGET}_t"]).to_numpy(float)
    models = {
        method: fit_regressor(fit_features[method], delta, fit_frame.subject_id)
        for method in METHODS
    }
    args.fitted_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "bindings": {k: report[k] for k in ("protocol_sha256", "script_sha256", "checkpoint_sha256", "feature_contract_sha256", "actual_future_window_hours")}}, args.fitted_output)
    report["fitted_bundle_sha256"] = sha256(args.fitted_output)
    report["frozen_before_mimic_evaluation"] = True
    report["eicu_hospital_role"] = hospital_role
    report["fit_subject_digest"] = subject_digest(fit_frame.subject_id)
    for role in ("patient_test", "hospital_test"):
        report["evaluation"][f"eicu_{role}"] = evaluate(
            *role_data[role], models, f"eicu_{role}", args.bootstrap
        )
        write_report(report, args.output)
    del frame, arrays, observations, role_data

    print("read aligned MIMIC", flush=True)
    frame, arrays, observations = load_dataset(args.mimic, "mimic", assets)
    eligible = valid_rows(frame, TARGET, assets.max_age_hours)
    eligible &= window_mask(frame, bounds)
    rows = select_role_rows(
        frame,
        eligible,
        np.ones(len(frame), dtype=bool),
        patients=args.max_external_patients,
        anchors=0,
        salt=f"anchor-aligned:{TARGET}:mimic",
    )
    if len(rows) < args.min_rows or frame.iloc[rows].subject_id.nunique() < args.min_patients:
        raise ValueError("insufficient MIMIC external support")
    mimic_data = prepare(frame, rows, assets, arrays, observations, contract)
    report["evaluation"]["mimic_external"] = evaluate(
        *mimic_data, models, "mimic_external", args.bootstrap
    )
    report["interpretation"] = external_interpretation(report["evaluation"]["mimic_external"])
    report["feature_widths"] = {name: int(fit_features[name].shape[1]) for name in METHODS}
    report["complete"] = True
    write_report(report, args.output)
    print(json.dumps({
        "complete": True,
        "mimic_patients": report["evaluation"]["mimic_external"]["patients"],
        "mimic_mae": report["evaluation"]["mimic_external"]["mae"],
        "interpretation": report["interpretation"],
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--eicu", type=Path, default=Path("/private/tmp/eicu_anchor_aligned_examples_6h_5000.parquet"))
    parser.add_argument("--mimic", type=Path, default=Path("/private/tmp/mimiciv_anchor_aligned_examples_6h_5000.parquet"))
    parser.add_argument("--window", choices=("narrow", "broad"), default="narrow")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fitted-output", type=Path, required=True)
    parser.add_argument("--max-fit-patients", type=int, default=3000)
    parser.add_argument("--max-test-patients", type=int, default=1000)
    parser.add_argument("--max-external-patients", type=int, default=2000)
    parser.add_argument("--fit-anchors", type=int, default=4)
    parser.add_argument("--min-patients", type=int, default=100)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
