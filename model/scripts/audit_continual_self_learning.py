"""Prequential continual-learning screen on exact creatinine measurement anchors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_measurement_anchor_encoder import (
    encode_latent,
    eligible_rows,
    load_dataset,
    selected_rows,
)
from scripts.screen_cross_variable_history_catalog import fit_regressor, target_role_masks
from whole_body_jepa.change_evaluation import estimate
from whole_body_jepa.continual_learning import (
    ADAPTER_Z_LIMIT,
    ContinualArms,
    PatientResidualState,
    fit_initial_adapter,
)
from whole_body_jepa.multisource_intervals import SEEDS, sha256, stable_hash, subject_digest
from whole_body_jepa.prospective_change import trend_features
from whole_body_jepa.runtime import load_release


TARGET = "creatinine"
HORIZON = 6
METHODS = (
    "persistence",
    "frozen_global",
    "frozen_patient_state",
    "naive_online",
    "dual_replay",
    "uniform_replay",
)
ONLINE_METHODS = ("naive_online", "dual_replay", "uniform_replay")


@dataclass(frozen=True)
class Prepared:
    frame: pd.DataFrame
    features: np.ndarray
    current: np.ndarray
    truth: np.ndarray


def prepare(frame, arrays, rows, assets, contract) -> Prepared:
    selected = frame.iloc[rows].reset_index(drop=True)
    own = trend_features(frame, rows, TARGET).astype(np.float32)
    latent = encode_latent(assets, arrays, rows, TARGET, contract)
    features = np.column_stack([own, latent]).astype(np.float32)
    if features.shape[1] != 263:
        raise ValueError(f"expected 263 continual features, got {features.shape[1]}")
    return Prepared(
        frame=selected,
        features=features,
        current=selected[f"{TARGET}_t"].to_numpy(float),
        truth=selected[f"future_{TARGET}"].to_numpy(float),
    )


def split_fit_patients(patients: set[str]) -> dict[str, set[str]]:
    ordered = sorted(patients, key=lambda value: stable_hash("continual-fit-role", value))
    first = int(round(0.60 * len(ordered)))
    second = int(round(0.80 * len(ordered)))
    result = {
        "base": set(ordered[:first]),
        "adapter": set(ordered[first:second]),
        "retention": set(ordered[second:]),
    }
    if min(map(len, result.values())) < 30:
        raise ValueError("insufficient source support after continual-role split")
    return result


def patient_delta(frame, candidate, reference, truth, *, salt, repetitions):
    values = np.abs(candidate - truth) - np.abs(reference - truth)
    return estimate(
        frame,
        values,
        seed=stable_hash("continual-bootstrap", salt),
        repetitions=repetitions,
    )


def summarize_predictions(frame, predictions, truth, *, salt, repetitions):
    result = {
        "rows": int(len(frame)),
        "patients": int(frame.subject_id.nunique()),
        "mae": {
            name: float(np.mean(np.abs(np.asarray(value) - truth)))
            for name, value in predictions.items()
        },
        "delta_vs_frozen_global": {},
        "delta_vs_persistence": {},
    }
    for method in METHODS:
        if method != "frozen_global":
            result["delta_vs_frozen_global"][method] = patient_delta(
                frame,
                predictions[method],
                predictions["frozen_global"],
                truth,
                salt=f"{salt}:{method}:frozen",
                repetitions=repetitions,
            )
        if method != "persistence":
            result["delta_vs_persistence"][method] = patient_delta(
                frame,
                predictions[method],
                predictions["persistence"],
                truth,
                salt=f"{salt}:{method}:persistence",
                repetitions=repetitions,
            )
    return result


def initial_prediction(prepared, base_model, scaler, adapter_model, residual_scale):
    base = prepared.current + base_model.predict(prepared.features)
    normalized = adapter_model.predict(scaler.transform(prepared.features))
    correction = np.clip(normalized, -ADAPTER_Z_LIMIT, ADAPTER_Z_LIMIT) * residual_scale
    return base, base + correction


def ordered_patient_rows(frame: pd.DataFrame, seed: int):
    patients = sorted(
        frame.subject_id.astype(str).unique(),
        key=lambda value: stable_hash("continual-patient-order", seed, value),
    )
    subject = frame.subject_id.astype(str).to_numpy()
    for patient in patients:
        rows = np.flatnonzero(subject == patient)
        yield patient, rows[np.argsort(frame.iloc[rows].t.to_numpy())]


def run_stream(
    stream: Prepared,
    base_model,
    scaler,
    adapter_model,
    residual_scale,
    seed_features,
    seed_residual,
    *,
    seed,
    repetitions,
):
    arms = ContinualArms.from_initial(
        scaler,
        adapter_model,
        residual_scale,
        seed_features,
        seed_residual,
        seed=seed,
    )
    base, frozen = initial_prediction(
        stream, base_model, scaler, adapter_model, residual_scale
    )
    records = []
    for patient, rows in ordered_patient_rows(stream.frame, seed):
        state = PatientResidualState()
        pending: list[tuple[pd.Timestamp, np.ndarray, float, float]] = []
        ordinal = 0
        for row in rows:
            anchor = pd.Timestamp(stream.frame.iloc[row].t)
            available, pending = (
                [item for item in pending if item[0] <= anchor],
                [item for item in pending if item[0] > anchor],
            )
            for _, features, residual, patient_residual in available:
                arms.update(features, residual)
                state.update(patient_residual)

            predictions = {
                "persistence": float(stream.current[row]),
                "frozen_global": float(frozen[row]),
                "frozen_patient_state": float(frozen[row] + state.correction),
            }
            for method in ONLINE_METHODS:
                predictions[method] = float(
                    base[row] + arms.predict_correction(method, stream.features[row])
                )
            records.append({
                "subject_id": patient,
                "repeat_prediction": bool(ordinal > 0),
                "truth": float(stream.truth[row]),
                **predictions,
            })
            due = anchor + pd.to_timedelta(
                float(stream.frame.iloc[row][f"future_delta_t_hr_{TARGET}"]), unit="h"
            )
            pending.append((
                due,
                stream.features[row],
                float(stream.truth[row] - base[row]),
                float(stream.truth[row] - frozen[row]),
            ))
            ordinal += 1
        for _, features, residual, patient_residual in sorted(pending, key=lambda item: item[0]):
            arms.update(features, residual)
            state.update(patient_residual)

    scored = pd.DataFrame(records)
    predictions = {name: scored[name].to_numpy(float) for name in METHODS}
    truth = scored.truth.to_numpy(float)
    all_result = summarize_predictions(
        scored, predictions, truth, salt=f"stream:{seed}:all", repetitions=repetitions
    )
    repeated = scored.loc[scored.repeat_prediction].reset_index(drop=True)
    repeated_predictions = {
        name: repeated[name].to_numpy(float) for name in METHODS
    }
    repeat_result = summarize_predictions(
        repeated,
        repeated_predictions,
        repeated.truth.to_numpy(float),
        salt=f"stream:{seed}:repeat",
        repetitions=repetitions,
    )
    return arms, {"all": all_result, "second_or_later": repeat_result}


def static_predictions(prepared, base_model, scaler, adapter_model, residual_scale, arms):
    base, frozen = initial_prediction(
        prepared, base_model, scaler, adapter_model, residual_scale
    )
    result = {
        "persistence": prepared.current.copy(),
        "frozen_global": frozen,
        "frozen_patient_state": frozen.copy(),
    }
    for method in ONLINE_METHODS:
        model = {
            "naive_online": arms.naive,
            "dual_replay": arms.dual,
            "uniform_replay": arms.uniform,
        }[method]
        normalized = model.predict(scaler.transform(prepared.features))
        result[method] = base + np.clip(
            normalized, -ADAPTER_Z_LIMIT, ADAPTER_Z_LIMIT
        ) * residual_scale
    return result


def evaluate_static(
    prepared,
    base_model,
    scaler,
    adapter_model,
    residual_scale,
    arms,
    *,
    salt,
    repetitions,
):
    predictions = static_predictions(
        prepared, base_model, scaler, adapter_model, residual_scale, arms
    )
    return summarize_predictions(
        prepared.frame,
        predictions,
        prepared.truth,
        salt=salt,
        repetitions=repetitions,
    )


def run(args):
    if args.output.exists():
        raise ValueError("refusing to overwrite continual-learning report")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    contract_path = args.model_dir / "FEATURE_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    protocol = Path("CONTINUAL_SELF_LEARNING_PROTOCOL_20260909.md")

    eicu, eicu_arrays = load_dataset(args.eicu, "eicu", assets)
    mimic, mimic_arrays = load_dataset(args.mimic, "mimic", assets)
    eicu_eligible = eligible_rows(eicu, fresh=True)
    mimic_eligible = eligible_rows(mimic, fresh=True)
    roles, hospital_metadata = target_role_masks(
        eicu, eicu_eligible, "continual-self-learning:creatinine@6h"
    )
    role_patients = {
        role: set(eicu.loc[eicu_eligible & mask, "subject_id"].astype(str))
        for role, mask in roles.items()
    }
    source_roles = split_fit_patients(role_patients["fit"])

    prepared = {}
    for role, patients in source_roles.items():
        rows = selected_rows(
            eicu,
            eicu_eligible,
            patients,
            anchors=4 if role != "retention" else 0,
            salt=f"continual:{role}",
        )
        prepared[role] = prepare(eicu, eicu_arrays, rows, assets, contract)
    for role in ("patient_test", "hospital_test"):
        rows = selected_rows(
            eicu,
            eicu_eligible,
            role_patients[role],
            anchors=0,
            salt=f"continual:{role}",
        )
        prepared[role] = prepare(eicu, eicu_arrays, rows, assets, contract)

    mimic_patients = set(mimic.loc[mimic_eligible, "subject_id"].astype(str))
    mimic_patients = set(sorted(
        mimic_patients,
        key=lambda value: stable_hash("continual-mimic-cap", value),
    )[:args.max_external_patients])
    mimic_rows = selected_rows(
        mimic,
        mimic_eligible,
        mimic_patients,
        anchors=0,
        salt="continual:mimic",
    )
    prepared["mimic"] = prepare(
        mimic, mimic_arrays, mimic_rows, assets, contract
    )

    base_delta = prepared["base"].truth - prepared["base"].current
    base_model = fit_regressor(
        prepared["base"].features,
        base_delta,
        prepared["base"].frame.subject_id,
    )
    adapter_base = (
        prepared["adapter"].current
        + base_model.predict(prepared["adapter"].features)
    )
    adapter_residual = prepared["adapter"].truth - adapter_base
    scaler, adapter_model, residual_scale = fit_initial_adapter(
        prepared["adapter"].features,
        adapter_residual,
        seed=20260909,
    )

    report = {
        "schema": "whole_body_jepa.continual_self_learning.v2",
        "complete": False,
        "status": "candidate_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "target": TARGET,
        "horizon_hours": HORIZON,
        "actual_future_window_hours": [5.0, 7.0],
        "feature_width": int(prepared["base"].features.shape[1]),
        "adapter_residual_scale": float(residual_scale),
        "adapter_z_limit": ADAPTER_Z_LIMIT,
        "protocol_sha256": sha256(protocol),
        "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": sha256(args.model_dir / "whole_body_joint_jepa.pt"),
        "feature_contract_sha256": sha256(contract_path),
        "inputs": {"eicu": sha256(args.eicu), "mimic": sha256(args.mimic)},
        "hospital_role": hospital_metadata,
        "support": {},
        "seeds": [],
    }
    for role, data in prepared.items():
        report["support"][role] = {
            "rows": int(len(data.frame)),
            "patients": int(data.frame.subject_id.nunique()),
            "subject_digest": subject_digest(data.frame.subject_id),
        }

    for seed in SEEDS:
        print(f"continual patient order seed {seed}", flush=True)
        arms, stream = run_stream(
            prepared["patient_test"],
            base_model,
            scaler,
            adapter_model,
            residual_scale,
            prepared["adapter"].features,
            adapter_residual,
            seed=seed,
            repetitions=args.bootstrap,
        )
        static = {
            role: evaluate_static(
                prepared[role],
                base_model,
                scaler,
                adapter_model,
                residual_scale,
                arms,
                salt=f"{role}:{seed}",
                repetitions=args.bootstrap,
            )
            for role in ("retention", "hospital_test", "mimic")
        }
        report["seeds"].append({"seed": seed, "stream": stream, "post_stream": static})

    report["summary"] = {}
    for method in ("frozen_patient_state", *ONLINE_METHODS):
        repeat_results = [
            seed["stream"]["second_or_later"]["delta_vs_frozen_global"][method]
            for seed in report["seeds"]
        ]
        persistence_results = [
            seed["stream"]["second_or_later"]["delta_vs_persistence"][method]
            for seed in report["seeds"]
        ]
        repeat_deltas = [item["mean"] for item in repeat_results]
        harms = {}
        for role in ("retention", "hospital_test", "mimic"):
            role_results = [
                seed["post_stream"][role]["delta_vs_frozen_global"][method]
                for seed in report["seeds"]
            ]
            harms[role] = {
                "mean_deltas": [item["mean"] for item in role_results],
                "detectable_harm_seeds": int(sum(
                    item["ci95"] is not None and item["ci95"][0] > 0
                    for item in role_results
                )),
            }
        report["summary"][method] = {
            "second_or_later_delta_vs_frozen_by_seed": repeat_deltas,
            "mean_improves_frozen_all_seeds": bool(
                all(value is not None and value < 0 for value in repeat_deltas)
            ),
            "significantly_improves_frozen_seeds": int(sum(
                item["ci95"] is not None and item["ci95"][1] < 0
                for item in repeat_results
            )),
            "significantly_beats_persistence_seeds": int(sum(
                item["ci95"] is not None and item["ci95"][1] < 0
                for item in persistence_results
            )),
            "post_stream_harm": harms,
        }
    report["source_adaptation_signal"] = bool(any(
        report["summary"][method]["mean_improves_frozen_all_seeds"]
        and report["summary"][method]["significantly_improves_frozen_seeds"] >= 4
        and all(
            report["summary"][method]["post_stream_harm"][role]["detectable_harm_seeds"] == 0
            for role in ("retention", "hospital_test")
        )
        for method in ONLINE_METHODS
    ))
    report["cross_database_continual_pass"] = bool(any(
        report["summary"][method]["significantly_improves_frozen_seeds"] == len(SEEDS)
        and report["summary"][method]["significantly_beats_persistence_seeds"] == len(SEEDS)
        and all(
            report["summary"][method]["post_stream_harm"][role]["detectable_harm_seeds"] == 0
            for role in ("retention", "hospital_test", "mimic")
        )
        for method in ONLINE_METHODS
    ))
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "complete": True,
        "support": report["support"],
        "summary": report["summary"],
        "source_adaptation_signal": report["source_adaptation_signal"],
        "cross_database_continual_pass": report["cross_database_continual_pass"],
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--eicu",
        type=Path,
        default=Path("/private/tmp/eicu_creatinine_measurement_anchor_examples_v2_6h_5000.parquet"),
    )
    parser.add_argument(
        "--mimic",
        type=Path,
        default=Path("/private/tmp/mimiciv_creatinine_measurement_anchor_examples_v2_6h_5000.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/continual_self_learning_20260909_v2.json"),
    )
    parser.add_argument("--max-external-patients", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
