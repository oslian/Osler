"""Test continual residual learning on the validated glucose@6h release source."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from osler_jepa.shared_release_features import build_target_head_features
from scripts.audit_continual_self_learning import (
    METHODS,
    ONLINE_METHODS,
    ordered_patient_rows,
    summarize_predictions,
)
from scripts.audit_measurement_anchor_encoder import load_dataset, selected_rows
from scripts.audit_release_masked_cells import valid_future_rows
from scripts.screen_cross_variable_history_catalog import target_role_masks
from whole_body_jepa.continual_learning import (
    ADAPTER_Z_LIMIT,
    ContinualArms,
    PatientResidualState,
    initialize_zero_adapter,
)
from whole_body_jepa.multisource_intervals import SEEDS, sha256, stable_hash, subject_digest
from whole_body_jepa.runtime import load_release
from whole_body_jepa.source_selection import HEAD, predict_head_values


TARGET = "glucose"
HORIZON = 6
CELL = "glucose@6h"


@dataclass(frozen=True)
class Prepared:
    frame: pd.DataFrame
    features: np.ndarray
    current: np.ndarray
    truth: np.ndarray
    base: np.ndarray


def eligible_rows(frame: pd.DataFrame) -> np.ndarray:
    age = pd.to_numeric(frame[f"{TARGET}_age_hr"], errors="coerce").to_numpy(float)
    delta_t = pd.to_numeric(
        frame[f"future_delta_t_hr_{TARGET}"], errors="coerce"
    ).to_numpy(float)
    return (
        valid_future_rows(frame, TARGET)
        & np.isclose(age, 0.0, atol=1e-9)
        & (delta_t >= 5.0)
        & (delta_t <= 7.0)
    )


def release_features(assets, arrays, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_index = assets.checkpoint["variables"].index(TARGET)
    organ_index = int(assets.model.variable_module_ids[target_index])
    target_scale = float(assets.scaler.scales[target_index])
    target_median = float(assets.scaler.medians[target_index])
    head = assets.heads["heads"][CELL]
    features = []
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            history = torch.from_numpy(arrays["input"][arrays["history"][batch]]).float()
            current = torch.from_numpy(arrays["current"][batch]).float()
            output = assets.model(
                history,
                current,
                torch.from_numpy(arrays["mask"][batch]).float(),
                torch.from_numpy(arrays["ages"][batch]).float(),
                torch.from_numpy(arrays["presence"][batch]).float(),
                torch.from_numpy(arrays["horizon"][batch]).float(),
            )
            base = output["value"][:, target_index].cpu().numpy()
            context = output["context_latent"].cpu().numpy()
            belief = output["belief_state"].cpu().numpy()
            future = output["predicted_target_latent"].cpu().numpy()
            organs = output["organ_latents"].cpu().numpy()
            batch_features = np.stack([
                build_target_head_features(
                    shared_context=context[index],
                    patient_belief=belief[index],
                    target_future=future[index, target_index],
                    target_organ=organs[index, organ_index],
                    base_forecast_normalized=base[index],
                    measurement_pure_state=arrays["current"][row],
                    target_current_normalized=arrays["current"][row, target_index],
                )
                for index, row in enumerate(batch)
            ])
            anchor = (
                arrays["current"][batch, target_index].astype(float) * target_scale
                + target_median
            )
            features.append(batch_features.astype(np.float32))
            predictions.append(
                predict_head_values(head, batch_features, anchor, runtime=True)
            )
    combined = np.concatenate(features)
    if combined.shape[1] != 311:
        raise ValueError(f"expected 311 release features, got {combined.shape[1]}")
    return combined, np.concatenate(predictions).astype(float)


def prepare(frame, arrays, rows, assets) -> Prepared:
    selected = frame.iloc[rows].reset_index(drop=True)
    features, base = release_features(assets, arrays, rows)
    return Prepared(
        frame=selected,
        features=features,
        current=selected[f"{TARGET}_t"].to_numpy(float),
        truth=selected[f"future_{TARGET}"].to_numpy(float),
        base=base,
    )


def split_fit_patients(patients: set[str]) -> dict[str, set[str]]:
    ordered = sorted(
        patients, key=lambda value: stable_hash("continual-validated-fit-role", value)
    )
    midpoint = len(ordered) // 2
    result = {"memory": set(ordered[:midpoint]), "retention": set(ordered[midpoint:])}
    if min(map(len, result.values())) < 100:
        raise ValueError("insufficient validated-cell source support")
    return result


def run_stream(
    stream: Prepared,
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
    records = []
    for patient, rows in ordered_patient_rows(stream.frame, seed):
        patient_state = PatientResidualState()
        pending: list[tuple[pd.Timestamp, np.ndarray, float]] = []
        ordinal = 0
        for row in rows:
            anchor = pd.Timestamp(stream.frame.iloc[row].t)
            available = [item for item in pending if item[0] <= anchor]
            pending = [item for item in pending if item[0] > anchor]
            for _, features, residual in available:
                arms.update(features, residual)
                patient_state.update(residual)

            predictions = {
                "persistence": float(stream.current[row]),
                "frozen_global": float(stream.base[row]),
                "frozen_patient_state": float(
                    stream.base[row] + patient_state.correction
                ),
            }
            for method in ONLINE_METHODS:
                predictions[method] = float(
                    stream.base[row]
                    + arms.predict_correction(method, stream.features[row])
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
                float(stream.truth[row] - stream.base[row]),
            ))
            ordinal += 1
        for _, features, residual in sorted(pending, key=lambda item: item[0]):
            arms.update(features, residual)
            patient_state.update(residual)

    scored = pd.DataFrame(records)
    predictions = {name: scored[name].to_numpy(float) for name in METHODS}
    overall = summarize_predictions(
        scored,
        predictions,
        scored.truth.to_numpy(float),
        salt=f"validated-stream:{seed}:all",
        repetitions=repetitions,
    )
    overall["correction_from_frozen"] = correction_summary(scored, residual_scale)
    repeated = scored.loc[scored.repeat_prediction].reset_index(drop=True)
    repeated_predictions = {name: repeated[name].to_numpy(float) for name in METHODS}
    later = summarize_predictions(
        repeated,
        repeated_predictions,
        repeated.truth.to_numpy(float),
        salt=f"validated-stream:{seed}:later",
        repetitions=repetitions,
    )
    later["correction_from_frozen"] = correction_summary(repeated, residual_scale)
    return arms, {"all": overall, "second_or_later": later}


def correction_summary(frame: pd.DataFrame, residual_scale: float) -> dict:
    limit = ADAPTER_Z_LIMIT * float(residual_scale)
    result = {}
    for method in ("frozen_patient_state", *ONLINE_METHODS):
        correction = np.abs(
            frame[method].to_numpy(float) - frame["frozen_global"].to_numpy(float)
        )
        result[method] = {
            "mean_absolute": float(correction.mean()),
            "quantiles": {
                str(level): float(np.quantile(correction, level))
                for level in (0.5, 0.9, 0.99, 1.0)
            },
            "adapter_limit": limit,
            "fraction_at_adapter_limit": float(np.mean(correction >= 0.999 * limit)),
        }
    return result


def static_predictions(prepared, arms):
    predictions = {
        "persistence": prepared.current.copy(),
        "frozen_global": prepared.base.copy(),
        "frozen_patient_state": prepared.base.copy(),
    }
    normalized = arms.scaler.transform(prepared.features)
    for method, model in (
        ("naive_online", arms.naive),
        ("dual_replay", arms.dual),
        ("uniform_replay", arms.uniform),
    ):
        correction = np.clip(
            model.predict(normalized), -ADAPTER_Z_LIMIT, ADAPTER_Z_LIMIT
        ) * arms.scale
        predictions[method] = prepared.base + correction
    return predictions


def evaluate_static(prepared, arms, *, salt, repetitions):
    predictions = static_predictions(prepared, arms)
    result = summarize_predictions(
        prepared.frame,
        predictions,
        prepared.truth,
        salt=salt,
        repetitions=repetitions,
    )
    diagnostic = prepared.frame.copy()
    for name, values in predictions.items():
        diagnostic[name] = values
    result["correction_from_frozen"] = correction_summary(diagnostic, arms.scale)
    return result


def frozen_evaluation(prepared, *, salt, repetitions):
    predictions = {
        "persistence": prepared.current,
        "frozen_global": prepared.base,
        "frozen_patient_state": prepared.base,
        "naive_online": prepared.base,
        "dual_replay": prepared.base,
        "uniform_replay": prepared.base,
    }
    return summarize_predictions(
        prepared.frame,
        predictions,
        prepared.truth,
        salt=salt,
        repetitions=repetitions,
    )


def method_summary(report, method):
    frozen_results = [
        seed["stream"]["second_or_later"]["delta_vs_frozen_global"][method]
        for seed in report["seeds"]
    ]
    persistence_results = [
        seed["stream"]["second_or_later"]["delta_vs_persistence"][method]
        for seed in report["seeds"]
    ]
    harms = {}
    for role in ("retention", "hospital_test", "mimic"):
        values = [
            seed["post_stream"][role]["delta_vs_frozen_global"][method]
            for seed in report["seeds"]
        ]
        harms[role] = {
            "mean_deltas": [item["mean"] for item in values],
            "detectable_harm_seeds": int(sum(
                item["ci95"] is not None and item["ci95"][0] > 0 for item in values
            )),
        }
    return {
        "second_or_later_delta_vs_frozen_by_seed": [
            item["mean"] for item in frozen_results
        ],
        "mean_improves_frozen_all_seeds": bool(all(
            item["mean"] is not None and item["mean"] < 0 for item in frozen_results
        )),
        "significantly_improves_frozen_seeds": int(sum(
            item["ci95"] is not None and item["ci95"][1] < 0
            for item in frozen_results
        )),
        "significantly_beats_persistence_seeds": int(sum(
            item["ci95"] is not None and item["ci95"][1] < 0
            for item in persistence_results
        )),
        "post_stream_harm": harms,
    }


def run(args):
    if args.output.exists():
        raise ValueError("refusing to overwrite validated-cell report")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    decision = assets.source_selector["cells"][CELL]
    if decision["source_model"] != HEAD or not decision["can_move"]:
        raise ValueError("glucose@6h release source is no longer the validated shared head")

    eicu, eicu_arrays = load_dataset(args.eicu, "eicu", assets)
    mimic, mimic_arrays = load_dataset(args.mimic, "mimic", assets)
    eicu_eligible = eligible_rows(eicu)
    mimic_eligible = eligible_rows(mimic)
    roles, hospital_metadata = target_role_masks(
        eicu, eicu_eligible, "continual-validated:glucose@6h"
    )
    role_patients = {
        role: set(eicu.loc[eicu_eligible & mask, "subject_id"].astype(str))
        for role, mask in roles.items()
    }
    fit_roles = split_fit_patients(role_patients["fit"])

    prepared = {}
    for role, patients in fit_roles.items():
        rows = selected_rows(
            eicu,
            eicu_eligible,
            patients,
            anchors=4 if role == "memory" else 0,
            salt=f"continual-validated:{role}",
        )
        prepared[role] = prepare(eicu, eicu_arrays, rows, assets)
    for role in ("patient_test", "hospital_test"):
        rows = selected_rows(
            eicu,
            eicu_eligible,
            role_patients[role],
            anchors=0,
            salt=f"continual-validated:{role}",
        )
        prepared[role] = prepare(eicu, eicu_arrays, rows, assets)

    mimic_patients = set(mimic.loc[mimic_eligible, "subject_id"].astype(str))
    mimic_patients = set(sorted(
        mimic_patients,
        key=lambda value: stable_hash("continual-validated-mimic-cap", value),
    )[:args.max_external_patients])
    mimic_rows = selected_rows(
        mimic,
        mimic_eligible,
        mimic_patients,
        anchors=0,
        salt="continual-validated:mimic",
    )
    prepared["mimic"] = prepare(mimic, mimic_arrays, mimic_rows, assets)

    memory_residual = prepared["memory"].truth - prepared["memory"].base
    scaler, adapter_model, residual_scale = initialize_zero_adapter(
        prepared["memory"].features,
        memory_residual,
        seed=20260909,
    )
    protocol = Path("CONTINUAL_VALIDATED_CELL_PROTOCOL_20260909.md")
    report = {
        "schema": "whole_body_jepa.continual_validated_cell.v2",
        "complete": False,
        "status": "candidate_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "cell": CELL,
        "target": TARGET,
        "horizon_hours": HORIZON,
        "actual_future_window_hours": [5.0, 7.0],
        "incumbent_source": HEAD,
        "feature_width": int(prepared["memory"].features.shape[1]),
        "adapter_starts_at_zero": True,
        "adapter_residual_scale": float(residual_scale),
        "adapter_z_limit": ADAPTER_Z_LIMIT,
        "protocol_sha256": sha256(protocol),
        "script_sha256": sha256(Path(__file__)),
        "shared_evaluation_script_sha256": sha256(
            Path("scripts/audit_continual_self_learning.py")
        ),
        "continual_module_sha256": sha256(
            Path("whole_body_jepa/continual_learning.py")
        ),
        "checkpoint_sha256": sha256(args.model_dir / "whole_body_joint_jepa.pt"),
        "inputs": {"eicu": sha256(args.eicu), "mimic": sha256(args.mimic)},
        "hospital_role": hospital_metadata,
        "support": {},
        "frozen_incumbent": {},
        "seeds": [],
    }
    for role, data in prepared.items():
        report["support"][role] = {
            "rows": int(len(data.frame)),
            "patients": int(data.frame.subject_id.nunique()),
            "repeat_patients": int(
                data.frame.groupby("subject_id").size().ge(2).sum()
            ),
            "subject_digest": subject_digest(data.frame.subject_id),
        }
        report["frozen_incumbent"][role] = frozen_evaluation(
            data, salt=f"frozen:{role}", repetitions=args.bootstrap
        )

    for seed in SEEDS:
        print(f"validated-cell patient order seed {seed}", flush=True)
        arms, stream = run_stream(
            prepared["patient_test"],
            scaler,
            adapter_model,
            residual_scale,
            prepared["memory"].features,
            memory_residual,
            seed=seed,
            repetitions=args.bootstrap,
        )
        static = {
            role: evaluate_static(
                prepared[role],
                arms,
                salt=f"validated:{role}:{seed}",
                repetitions=args.bootstrap,
            )
            for role in ("retention", "hospital_test", "mimic")
        }
        report["seeds"].append({"seed": seed, "stream": stream, "post_stream": static})

    report["summary"] = {
        method: method_summary(report, method)
        for method in ("frozen_patient_state", *ONLINE_METHODS)
    }
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
        "cross_database_continual_pass": report["cross_database_continual_pass"],
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--eicu",
        type=Path,
        default=Path(
            "/private/tmp/eicu_creatinine_measurement_anchor_examples_v2_6h_5000.parquet"
        ),
    )
    parser.add_argument(
        "--mimic",
        type=Path,
        default=Path(
            "/private/tmp/mimiciv_creatinine_measurement_anchor_examples_v2_6h_5000.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/continual_validated_glucose_6h_20260909_v2.json"),
    )
    parser.add_argument("--max-external-patients", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
