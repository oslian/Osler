"""Compare target-only and whole-body JEPA at real target-measurement anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_anchor_aligned_signal import careunit_results, window_mask
from scripts.audit_prospective_change import select_role_rows
from scripts.audit_release_masked_cells import input_frame, read_examples
from scripts.screen_cross_variable_history_catalog import (
    comparison,
    fit_regressor,
    target_role_masks,
    valid_rows,
)
from whole_body_jepa.cross_variable_history import matched_wrong_patient_rows
from whole_body_jepa.multisource_intervals import (
    representative_rows,
    sha256,
    stable_hash,
    subject_digest,
)
from whole_body_jepa.prospective_change import trend_features
from whole_body_jepa.runtime import _build_inputs, load_release


TARGET = "creatinine"
HORIZON = 6
METHODS = (
    "own_history",
    "target_only_encoder",
    "whole_body_encoder",
    "whole_body_placebo",
)
COMPARISONS = {
    "target_only_encoder": ("own_history", "persistence"),
    "whole_body_encoder": (
        "target_only_encoder",
        "whole_body_placebo",
        "own_history",
        "persistence",
    ),
}
LATENT_BLOCKS = ("shared_context", "patient_belief", "target_future", "target_organ")


def target_only_arrays(arrays: dict[str, np.ndarray], target_index: int) -> dict[str, np.ndarray]:
    """Neutralize non-target values while preserving identical time metadata."""

    result = dict(arrays)
    result["input"] = arrays["input"].copy()
    result["current"] = arrays["current"].copy()
    count = result["current"].shape[1]
    removed = np.asarray([index for index in range(count) if index != target_index])
    result["input"][:, removed] = 0.0
    result["current"][:, removed] = 0.0
    return result


def encode_latent(assets, arrays, rows, target, contract):
    index = assets.checkpoint["variables"].index(target)
    organ = int(assets.model.variable_module_ids[index])
    blocks = {item["name"]: item for item in contract["blocks"]}
    expected = sum(int(blocks[name]["width"]) for name in LATENT_BLOCKS)
    features = []
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        history = arrays["input"][arrays["history"][batch]]
        with torch.inference_mode():
            output = assets.model(
                torch.from_numpy(history).float(),
                torch.from_numpy(arrays["current"][batch]).float(),
                torch.from_numpy(arrays["mask"][batch]).float(),
                torch.from_numpy(arrays["ages"][batch]).float(),
                torch.from_numpy(arrays["presence"][batch]).float(),
                torch.from_numpy(arrays["horizon"][batch]).float(),
            )
        latent = np.concatenate(
            [
                output["context_latent"].cpu().numpy(),
                output["belief_state"].cpu().numpy(),
                output["predicted_target_latent"][:, index].cpu().numpy(),
                output["organ_latents"][:, organ].cpu().numpy(),
            ],
            axis=1,
        )
        if latent.shape[1] != expected:
            raise ValueError("latent feature contract width mismatch")
        features.append(latent)
    return np.concatenate(features).astype(np.float32)


def prepare(frame, rows, assets, arrays, contract):
    selected = frame.iloc[rows].reset_index(drop=True)
    own = trend_features(frame, rows, TARGET).astype(np.float32)
    index = assets.checkpoint["variables"].index(TARGET)
    target_latent = encode_latent(assets, target_only_arrays(arrays, index), rows, TARGET, contract)
    full_latent = encode_latent(assets, arrays, rows, TARGET, contract)
    donors = matched_wrong_patient_rows(
        selected, seed=stable_hash("measurement_anchor_encoder_donor", TARGET)
    )
    return selected, {
        "own_history": own,
        "target_only_encoder": np.column_stack([own, target_latent]).astype(np.float32),
        "whole_body_encoder": np.column_stack([own, full_latent]).astype(np.float32),
        "whole_body_placebo": np.column_stack([own, full_latent[donors]]).astype(np.float32),
    }


def load_dataset(path, database, assets):
    frame = read_examples(path, assets, database, HORIZON, grid=True)
    filtered = input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours)
    return frame, _build_inputs(filtered, assets, HORIZON)


def eligible_rows(frame, *, fresh):
    eligible = valid_rows(frame, TARGET, 6.0) & window_mask(frame, (5.0, 7.0))
    if fresh:
        age = pd.to_numeric(frame[f"{TARGET}_age_hr"], errors="coerce").to_numpy(float)
        eligible &= np.isclose(age, 0.0, atol=1e-9)
    return eligible


def selected_rows(frame, eligible, patients, *, anchors, salt):
    allowed = eligible & frame.subject_id.isin(patients).to_numpy()
    rows = select_role_rows(
        frame,
        allowed,
        np.ones(len(frame), dtype=bool),
        patients=len(patients),
        anchors=anchors,
        salt=salt,
    )
    selected_patients = set(frame.iloc[rows].subject_id)
    if selected_patients != set(patients):
        missing = len(set(patients) - selected_patients)
        raise ValueError(f"fixed patient support lost {missing} patients")
    return rows


def predict(features, models, current):
    return {
        name: current + models[name].predict(features[name]) for name in METHODS
    } | {"persistence": current.copy()}


def pair_result(frame, predictions, truth, candidate, reference, role, bootstrap):
    return comparison(
        frame,
        predictions,
        truth,
        candidate,
        reference,
        salt=f"measurement-anchor:{TARGET}:{role}:{candidate}:{reference}",
        repetitions=bootstrap,
    )


def evaluate(frame, features, models, role, bootstrap):
    rows = representative_rows(
        frame, stable_hash("measurement_anchor_representative", TARGET, role)
    )
    work = frame.iloc[rows].reset_index(drop=True)
    selected = {name: value[rows] for name, value in features.items()}
    current = work[f"{TARGET}_t"].to_numpy(float)
    truth = work[f"future_{TARGET}"].to_numpy(float)
    predictions = predict(selected, models, current)
    methods = {}
    for candidate, references in COMPARISONS.items():
        comparisons = {
            reference: pair_result(
                work, predictions, truth, candidate, reference, role, bootstrap
            )
            for reference in references
        }
        units = {
            reference: careunit_results(
                work, predictions, truth, candidate, reference, bootstrap
            )
            for reference in references
        }
        methods[candidate] = {
            "comparisons": comparisons,
            "overall_pass": bool(all(value["pass"] for value in comparisons.values())),
            "careunits": units,
            "all_supported_careunits_pass": bool(
                units[references[0]]
                and all(value["pass"] for group in units.values() for value in group.values())
            ),
        }
    return {
        "patients": int(work.subject_id.nunique()),
        "mae": {
            name: float(np.mean(np.abs(value - truth)))
            for name, value in predictions.items()
        },
        "timing": {
            f"{TARGET}_age_hr": {
                str(q): float(work[f"{TARGET}_age_hr"].quantile(q))
                for q in (0, 0.5, 0.9, 1)
            },
            f"future_delta_t_hr_{TARGET}": {
                str(q): float(work[f"future_delta_t_hr_{TARGET}"].quantile(q))
                for q in (0, 0.5, 0.9, 1)
            },
        },
        "methods": methods,
    }


def common_patients(left, left_eligible, right, right_eligible):
    return set(left.loc[left_eligible, "subject_id"]) & set(
        right.loc[right_eligible, "subject_id"]
    )


def role_patient_sets(fresh, fresh_eligible, carried, carried_eligible):
    common = common_patients(fresh, fresh_eligible, carried, carried_eligible)
    support = fresh_eligible & fresh.subject_id.isin(common).to_numpy()
    roles, metadata = target_role_masks(fresh, support, "measurement_anchor:creatinine@6h")
    patients = {
        role: set(fresh.loc[support & mask, "subject_id"])
        for role, mask in roles.items()
    }
    for left, right in (("fit", "patient_test"), ("fit", "hospital_test"), ("patient_test", "hospital_test")):
        if patients[left] & patients[right]:
            raise ValueError("patient roles overlap")
    return patients, metadata


def fit_mode(frame, arrays, eligible, role_patients, assets, contract, mode):
    role_data = {}
    for role in ("fit", "patient_test", "hospital_test"):
        rows = selected_rows(
            frame,
            eligible,
            role_patients[role],
            anchors=4 if role == "fit" else 0,
            salt=f"measurement-anchor:{mode}:{role}",
        )
        role_data[role] = prepare(frame, rows, assets, arrays, contract)
    fit_frame, fit_features = role_data["fit"]
    delta = (
        fit_frame[f"future_{TARGET}"] - fit_frame[f"{TARGET}_t"]
    ).to_numpy(float)
    models = {
        name: fit_regressor(fit_features[name], delta, fit_frame.subject_id)
        for name in METHODS
    }
    return models, role_data


def external_mode(frame, arrays, eligible, patients, assets, contract):
    rows = selected_rows(
        frame, eligible, patients, anchors=0, salt="measurement-anchor:mimic"
    )
    return prepare(frame, rows, assets, arrays, contract)


def run(args):
    if args.output.exists() or args.fitted_output.exists():
        raise ValueError("refusing to overwrite experiment artifacts")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    protocol = Path("MEASUREMENT_ANCHOR_ENCODER_PROTOCOL_20260909.md")
    contract_path = args.model_dir / "FEATURE_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    report = {
        "schema": "whole_body_jepa.measurement_anchor_encoder.v1",
        "complete": False,
        "status": "source_attribution_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "target": TARGET,
        "horizon_hours": HORIZON,
        "actual_future_window_hours": [5.0, 7.0],
        "protocol_sha256": sha256(protocol),
        "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": sha256(args.model_dir / "whole_body_joint_jepa.pt"),
        "feature_contract_sha256": sha256(contract_path),
        "inputs": {name: sha256(path) for name, path in {
            "eicu_fresh": args.eicu_fresh,
            "eicu_carried": args.eicu_carried,
            "mimic_fresh": args.mimic_fresh,
            "mimic_carried": args.mimic_carried,
        }.items()},
        "evaluation": {},
    }

    datasets = {}
    for name, path, database in (
        ("eicu_fresh", args.eicu_fresh, "eicu"),
        ("eicu_carried", args.eicu_carried, "eicu"),
        ("mimic_fresh", args.mimic_fresh, "mimic"),
        ("mimic_carried", args.mimic_carried, "mimic"),
    ):
        print(f"read {name}", flush=True)
        frame, arrays = load_dataset(path, database, assets)
        fresh = name.endswith("fresh")
        eligible = eligible_rows(frame, fresh=fresh)
        if fresh and not np.allclose(frame.loc[eligible, f"{TARGET}_age_hr"], 0.0):
            raise ValueError("fresh anchors are not target measurements")
        datasets[name] = (frame, arrays, eligible)

    eicu_fresh, eicu_fresh_arrays, eicu_fresh_eligible = datasets["eicu_fresh"]
    eicu_carried, eicu_carried_arrays, eicu_carried_eligible = datasets["eicu_carried"]
    role_patients, hospital_metadata = role_patient_sets(
        eicu_fresh, eicu_fresh_eligible, eicu_carried, eicu_carried_eligible
    )
    if min(map(len, role_patients.values())) < args.min_patients:
        raise ValueError("insufficient fixed eICU patient support")
    report["eicu_hospital_role"] = hospital_metadata
    report["role_support"] = {
        role: {"patients": len(values), "subject_digest": subject_digest(values)}
        for role, values in role_patients.items()
    }

    fitted = {}
    eicu_modes = {
        "fresh": (eicu_fresh, eicu_fresh_arrays, eicu_fresh_eligible),
        "carried": (eicu_carried, eicu_carried_arrays, eicu_carried_eligible),
    }
    role_data = {}
    for mode, (frame, arrays, eligible) in eicu_modes.items():
        print(f"fit eICU {mode}", flush=True)
        models, data = fit_mode(
            frame, arrays, eligible, role_patients, assets, contract, mode
        )
        fitted[mode] = models
        role_data[mode] = data
        for role in ("patient_test", "hospital_test"):
            report["evaluation"][f"eicu_{mode}_{role}"] = evaluate(
                *data[role], models, f"eicu_{mode}_{role}", args.bootstrap
            )

    args.fitted_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"models": fitted, "bindings": {
            key: report[key] for key in (
                "protocol_sha256", "script_sha256", "checkpoint_sha256",
                "feature_contract_sha256", "actual_future_window_hours", "inputs",
            )
        }},
        args.fitted_output,
    )
    report["fitted_bundle_sha256"] = sha256(args.fitted_output)
    report["frozen_before_mimic_evaluation"] = True

    mimic_fresh, mimic_fresh_arrays, mimic_fresh_eligible = datasets["mimic_fresh"]
    mimic_carried, mimic_carried_arrays, mimic_carried_eligible = datasets["mimic_carried"]
    mimic_patients = common_patients(
        mimic_fresh, mimic_fresh_eligible, mimic_carried, mimic_carried_eligible
    )
    mimic_patients = set(sorted(
        mimic_patients,
        key=lambda patient: stable_hash("measurement_anchor_mimic_cap", patient),
    )[:args.max_external_patients])
    if len(mimic_patients) < args.min_patients:
        raise ValueError("insufficient fixed MIMIC support")
    report["mimic_support"] = {
        "patients": len(mimic_patients),
        "subject_digest": subject_digest(mimic_patients),
    }
    for mode, frame, arrays, eligible in (
        ("fresh", mimic_fresh, mimic_fresh_arrays, mimic_fresh_eligible),
        ("carried", mimic_carried, mimic_carried_arrays, mimic_carried_eligible),
    ):
        print(f"evaluate MIMIC {mode}", flush=True)
        data = external_mode(frame, arrays, eligible, mimic_patients, assets, contract)
        report["evaluation"][f"mimic_{mode}"] = evaluate(
            *data, fitted[mode], f"mimic_{mode}", args.bootstrap
        )

    report["feature_widths"] = {
        name: int(role_data["fresh"]["fit"][1][name].shape[1]) for name in METHODS
    }
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "complete": True,
        "role_support": report["role_support"],
        "mimic_support": report["mimic_support"],
        "mae": {name: value["mae"] for name, value in report["evaluation"].items()},
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--eicu-fresh", type=Path, default=Path("/private/tmp/eicu_creatinine_measurement_anchor_examples_v2_6h_5000.parquet"))
    parser.add_argument("--eicu-carried", type=Path, default=Path("/private/tmp/eicu_anchor_aligned_examples_6h_5000.parquet"))
    parser.add_argument("--mimic-fresh", type=Path, default=Path("/private/tmp/mimiciv_creatinine_measurement_anchor_examples_v2_6h_5000.parquet"))
    parser.add_argument("--mimic-carried", type=Path, default=Path("/private/tmp/mimiciv_anchor_aligned_examples_6h_5000.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fitted-output", type=Path, required=True)
    parser.add_argument("--max-external-patients", type=int, default=2000)
    parser.add_argument("--min-patients", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
