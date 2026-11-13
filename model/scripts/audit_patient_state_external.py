"""Validate patient-local glucose state with full MIMIC prequential replay."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_continual_validated_cell import (
    CELL,
    HORIZON,
    TARGET,
    eligible_rows,
    prepare,
    split_fit_patients,
)
from scripts.audit_measurement_anchor_encoder import load_dataset, selected_rows
from scripts.screen_cross_variable_history_catalog import target_role_masks
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS, estimate, interval_score
from whole_body_jepa.continual_learning import (
    PatientResidualState,
    TimeAwarePatientResidualState,
)
from whole_body_jepa.cross_variable_history import matched_wrong_patient_rows
from whole_body_jepa.multisource_intervals import (
    BOUNDS,
    SEEDS,
    conformal_radius,
    coverage,
    domain_coverage,
    representative_rows,
    sha256,
    stable_hash,
    subject_digest,
)
from whole_body_jepa.runtime import load_release
from whole_body_jepa.source_selection import HEAD


METHODS = (
    "persistence",
    "frozen_release",
    "simple_patient_state",
    "time_aware_patient_state",
)
GRID = tuple(
    {
        "half_life_hours": float(half_life),
        "prior_strength": float(prior),
        "slope_penalty": float(slope_penalty),
        "max_history": 8,
    }
    for half_life, prior, slope_penalty in itertools.product(
        (12, 24, 48), (1, 3), (4, 16)
    )
)


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    return max(float(np.median(np.abs(values - median)) * 1.4826), 0.05)


def ordered_patient_rows(frame: pd.DataFrame):
    subjects = frame.subject_id.astype(str).to_numpy()
    for patient in sorted(np.unique(subjects)):
        rows = np.flatnonzero(subjects == patient)
        yield patient, rows[np.argsort(pd.to_datetime(frame.iloc[rows].t).to_numpy())]


def prequential_predictions(prepared, config: dict, reference_scales: dict) -> pd.DataFrame:
    records = []
    for patient, rows in ordered_patient_rows(prepared.frame):
        simple = PatientResidualState(prior_strength=3.0)
        timed = TimeAwarePatientResidualState(
            reference_scale=reference_scales["frozen_release"],
            **config,
        )
        pending: list[tuple[pd.Timestamp, float]] = []
        for ordinal, row in enumerate(rows):
            anchor = pd.Timestamp(prepared.frame.iloc[row].t)
            resolved = sorted(
                (item for item in pending if item[0] <= anchor),
                key=lambda item: item[0],
            )
            pending = [item for item in pending if item[0] > anchor]
            for resolved_at, residual in resolved:
                simple.update(residual)
                timed.update(resolved_at, residual)

            timed_correction, timed_scale = timed.estimate(anchor, HORIZON)
            simple_correction = simple.correction
            simple_scale = simple.uncertainty_scale(
                reference_scales["frozen_release"]
            )
            source = prepared.frame.iloc[row]
            records.append(
                {
                    "subject_id": str(patient),
                    "stay_id": str(source.stay_id),
                    "t": anchor,
                    "_time_hours": float(source._time_hours),
                    "careunit": str(source.careunit),
                    "ordinal": int(ordinal),
                    "resolved_history_count": int(simple.count),
                    "current": float(prepared.current[row]),
                    "truth": float(prepared.truth[row]),
                    "persistence": float(prepared.current[row]),
                    "frozen_release": float(prepared.base[row]),
                    "simple_patient_state": float(
                        prepared.base[row] + simple_correction
                    ),
                    "time_aware_patient_state": float(
                        prepared.base[row] + timed_correction
                    ),
                    "simple_correction": float(simple_correction),
                    "time_aware_correction": float(timed_correction),
                    "persistence_scale": float(reference_scales["persistence"]),
                    "frozen_release_scale": float(
                        reference_scales["frozen_release"]
                    ),
                    "simple_patient_state_scale": float(simple_scale),
                    "time_aware_patient_state_scale": float(timed_scale),
                }
            )
            due = anchor + pd.to_timedelta(
                float(source[f"future_delta_t_hr_{TARGET}"]), unit="h"
            )
            pending.append((due, float(prepared.truth[row] - prepared.base[row])))
    return pd.DataFrame(records)


def add_shuffled_placebo(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    donors = matched_wrong_patient_rows(
        frame, seed=stable_hash("patient-state-placebo", seed, CELL)
    )
    result = frame.copy()
    result["shuffled_patient_state"] = (
        result["frozen_release"].to_numpy(float)
        + result["time_aware_correction"].to_numpy(float)[donors]
    )
    result["shuffled_patient_state_scale"] = result[
        "time_aware_patient_state_scale"
    ].to_numpy(float)[donors]
    return result


def patient_equal_mae(frame: pd.DataFrame, method: str) -> float:
    error = np.abs(frame[method].to_numpy(float) - frame.truth.to_numpy(float))
    return float(
        pd.Series(error, index=frame.subject_id).groupby(level=0).mean().mean()
    )


def paired_delta(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    *,
    salt: str,
    repetitions: int,
) -> dict:
    values = np.abs(frame[candidate] - frame.truth) - np.abs(
        frame[reference] - frame.truth
    )
    return estimate(
        frame,
        values.to_numpy(float),
        seed=stable_hash("patient-state-bootstrap", salt),
        repetitions=repetitions,
    )


def point_metrics(frame: pd.DataFrame, *, salt: str, repetitions: int) -> dict:
    truth = frame.truth.to_numpy(float)
    result = {
        "rows": int(len(frame)),
        "patients": int(frame.subject_id.nunique()),
        "patient_equal_mae": {
            method: patient_equal_mae(frame, method)
            for method in (*METHODS, "shuffled_patient_state")
        },
        "time_aware_comparisons": {},
    }
    for reference in (
        "frozen_release",
        "simple_patient_state",
        "persistence",
        "shuffled_patient_state",
    ):
        result["time_aware_comparisons"][reference] = paired_delta(
            frame,
            "time_aware_patient_state",
            reference,
            salt=f"{salt}:{reference}",
            repetitions=repetitions,
        )
    threshold = CHANGE_THRESHOLDS[TARGET]
    actual_change = np.abs(truth - frame.current.to_numpy(float)) >= threshold
    result["behavior"] = {}
    for method in (*METHODS, "shuffled_patient_state"):
        predicted_change = np.abs(
            frame[method].to_numpy(float) - frame.current.to_numpy(float)
        ) >= threshold
        stable = ~actual_change
        changed = actual_change
        result["behavior"][method] = {
            "false_move_rate_on_stable": (
                float(predicted_change[stable].mean()) if stable.any() else None
            ),
            "direction_accuracy_on_changed": (
                float(
                    np.mean(
                        np.sign(frame[method].to_numpy(float)[changed]
                                - frame.current.to_numpy(float)[changed])
                        == np.sign(truth[changed] - frame.current.to_numpy(float)[changed])
                    )
                )
                if changed.any()
                else None
            ),
            "stable_rows": int(stable.sum()),
            "changed_rows": int(changed.sum()),
        }
    return result


def evaluate_role(frame: pd.DataFrame, *, salt: str, repetitions: int) -> dict:
    personalized = frame.loc[frame.resolved_history_count > 0].reset_index(drop=True)
    threshold = CHANGE_THRESHOLDS[TARGET]
    changed = np.abs(personalized.truth - personalized.current) >= threshold
    strata = {
        "personalized": personalized,
        "stable": personalized.loc[~changed].reset_index(drop=True),
        "changed": personalized.loc[changed].reset_index(drop=True),
    }
    return {
        name: point_metrics(
            selected,
            salt=f"{salt}:{name}",
            repetitions=repetitions,
        )
        for name, selected in strata.items()
        if selected.subject_id.nunique() >= 30
    }


def fit_interval_radius(frame: pd.DataFrame, method: str, seed: int) -> float:
    selected = frame.loc[frame.resolved_history_count > 0].reset_index(drop=True)
    scale = selected[f"{method}_scale"].to_numpy(float)
    scores = np.abs(selected[method] - selected.truth).to_numpy(float) / scale
    work = selected[["subject_id", "stay_id", "t"]].copy()
    work["score"] = scores
    rows = representative_rows(work, seed)
    return conformal_radius(scores[rows])


def interval_metrics(frame: pd.DataFrame, method: str, radius: float) -> dict:
    selected = frame.loc[frame.resolved_history_count > 0].reset_index(drop=True)
    point = selected[method].to_numpy(float)
    truth = selected.truth.to_numpy(float)
    scale = selected[f"{method}_scale"].to_numpy(float)
    half_width = radius * scale
    score = np.abs(point - truth) / scale
    normalized = selected[["subject_id", "careunit", "t"]].copy()
    normalized["residual"] = score
    cover = coverage(normalized, radius)
    units = domain_coverage(normalized, radius)
    widths = 2.0 * half_width
    interval_scores = interval_score(
        truth, point - half_width, point + half_width, alpha=0.1
    )
    return {
        "radius": float(radius),
        "row_and_patient_coverage": cover,
        "careunit_coverage": units,
        "patient_equal_width": estimate(selected, widths),
        "patient_equal_interval_score": estimate(selected, interval_scores),
    }


def interval_width_delta(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    radii: dict,
    *,
    salt: str,
    repetitions: int,
) -> dict:
    selected = frame.loc[frame.resolved_history_count > 0].reset_index(drop=True)
    candidate_width = (
        2.0
        * radii[candidate]
        * selected[f"{candidate}_scale"].to_numpy(float)
    )
    reference_width = (
        2.0
        * radii[reference]
        * selected[f"{reference}_scale"].to_numpy(float)
    )
    return estimate(
        selected,
        candidate_width - reference_width,
        seed=stable_hash("patient-state-width", salt),
        repetitions=repetitions,
    )


def config_key(config: dict) -> tuple:
    return (
        -config["prior_strength"],
        -config["half_life_hours"],
        -config["slope_penalty"],
    )


def select_config(prepared, reference_scales: dict) -> tuple[dict, list[dict]]:
    leaderboard = []
    for config in GRID:
        predictions = prequential_predictions(prepared, config, reference_scales)
        personalized = predictions.loc[
            predictions.resolved_history_count > 0
        ].reset_index(drop=True)
        leaderboard.append(
            {
                **config,
                "rows": int(len(personalized)),
                "patients": int(personalized.subject_id.nunique()),
                "patient_equal_mae": patient_equal_mae(
                    personalized, "time_aware_patient_state"
                ),
            }
        )
    leaderboard.sort(
        key=lambda item: (item["patient_equal_mae"], *config_key(item))
    )
    selected = {
        key: leaderboard[0][key]
        for key in (
            "half_life_hours",
            "prior_strength",
            "slope_penalty",
            "max_history",
        )
    }
    return selected, leaderboard


def support(frame: pd.DataFrame) -> dict:
    personalized = frame.resolved_history_count > 0
    return {
        "rows": int(len(frame)),
        "patients": int(frame.subject_id.nunique()),
        "personalized_rows": int(personalized.sum()),
        "personalized_patients": int(frame.loc[personalized, "subject_id"].nunique()),
        "subject_digest": subject_digest(frame.subject_id),
    }


def run(args) -> None:
    if args.output.exists():
        raise ValueError("refusing to overwrite patient-state report")
    assets = load_release(args.model_dir)
    decision = assets.source_selector["cells"][CELL]
    if decision["source_model"] != HEAD or not decision["can_move"]:
        raise ValueError("glucose@6h is no longer backed by the validated shared head")

    eicu, eicu_arrays = load_dataset(args.eicu, "eicu", assets)
    mimic, mimic_arrays = load_dataset(args.mimic, "mimic", assets)
    eicu_eligible = eligible_rows(eicu)
    mimic_eligible = eligible_rows(mimic)
    roles, hospital_metadata = target_role_masks(
        eicu, eicu_eligible, "patient-state-external:glucose@6h"
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
            salt=f"patient-state-external:{role}",
        )
        prepared[role] = prepare(eicu, eicu_arrays, rows, assets)
    for role in ("patient_test", "hospital_test"):
        rows = selected_rows(
            eicu,
            eicu_eligible,
            role_patients[role],
            anchors=0,
            salt=f"patient-state-external:{role}",
        )
        prepared[role] = prepare(eicu, eicu_arrays, rows, assets)

    mimic_patients = set(mimic.loc[mimic_eligible, "subject_id"].astype(str))
    mimic_rows = selected_rows(
        mimic,
        mimic_eligible,
        mimic_patients,
        anchors=0,
        salt="patient-state-external:mimic",
    )
    prepared["mimic"] = prepare(mimic, mimic_arrays, mimic_rows, assets)

    reference_scales = {
        "frozen_release": robust_scale(
            prepared["memory"].truth - prepared["memory"].base
        ),
        "persistence": robust_scale(
            prepared["memory"].truth - prepared["memory"].current
        ),
    }
    selected_config, leaderboard = select_config(
        prepared["patient_test"], reference_scales
    )

    frames = {
        role: prequential_predictions(data, selected_config, reference_scales)
        for role, data in prepared.items()
    }
    primary_seed = SEEDS[0]
    primary_frames = {
        role: add_shuffled_placebo(frame, primary_seed)
        for role, frame in frames.items()
    }
    radii = {
        method: fit_interval_radius(primary_frames["retention"], method, primary_seed)
        for method in (*METHODS, "shuffled_patient_state")
    }

    report = {
        "schema": "whole_body_jepa.patient_state_external.v1",
        "complete": False,
        "status": "candidate_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "cell": CELL,
        "actual_future_window_hours": [5.0, 7.0],
        "incumbent_source": HEAD,
        "protocol_sha256": sha256(Path("PATIENT_STATE_EXTERNAL_PROTOCOL_20260909.md")),
        "script_sha256": sha256(Path(__file__)),
        "continual_module_sha256": sha256(
            Path("whole_body_jepa/continual_learning.py")
        ),
        "checkpoint_sha256": sha256(args.model_dir / "whole_body_joint_jepa.pt"),
        "inputs": {"eicu": sha256(args.eicu), "mimic": sha256(args.mimic)},
        "hospital_role": hospital_metadata,
        "reference_scales": reference_scales,
        "selection": {
            "database": "eicu",
            "role": "patient_test",
            "selected_config": selected_config,
            "leaderboard": leaderboard,
            "mimic_outcomes_used": False,
        },
        "support": {role: support(frame) for role, frame in frames.items()},
        "point": {},
        "placebo_seeds": {},
        "interval": {"calibration_role": "eicu_retention", "radii": radii},
    }

    for role in ("patient_test", "hospital_test", "mimic"):
        report["point"][role] = evaluate_role(
            primary_frames[role], salt=role, repetitions=args.bootstrap
        )
        report["placebo_seeds"][role] = []
        personalized = frames[role].loc[
            frames[role].resolved_history_count > 0
        ].reset_index(drop=True)
        for seed in SEEDS:
            placebo = add_shuffled_placebo(personalized, seed)
            report["placebo_seeds"][role].append(
                {
                    "seed": seed,
                    "comparison": paired_delta(
                        placebo,
                        "time_aware_patient_state",
                        "shuffled_patient_state",
                        salt=f"{role}:placebo:{seed}",
                        repetitions=args.bootstrap,
                    ),
                }
            )

    for role in ("hospital_test", "mimic"):
        frame = primary_frames[role]
        report["interval"][role] = {
            method: interval_metrics(frame, method, radii[method])
            for method in (*METHODS, "shuffled_patient_state")
        }
        report["interval"][role]["time_aware_width_delta"] = {
            reference: interval_width_delta(
                frame,
                "time_aware_patient_state",
                reference,
                radii,
                salt=f"{role}:{reference}",
                repetitions=args.bootstrap,
            )
            for reference in ("frozen_release", "simple_patient_state")
        }

    def point_pass(role: str) -> bool:
        comparisons = report["point"][role]["personalized"][
            "time_aware_comparisons"
        ]
        fixed = all(
            comparisons[reference]["ci95"] is not None
            and comparisons[reference]["ci95"][1] < 0
            for reference in (
                "frozen_release",
                "simple_patient_state",
                "persistence",
            )
        )
        placebos = all(
            item["comparison"]["ci95"] is not None
            and item["comparison"]["ci95"][1] < 0
            for item in report["placebo_seeds"][role]
        )
        return bool(fixed and placebos)

    def interval_pass(role: str) -> bool:
        result = report["interval"][role]["time_aware_patient_state"]
        coverage_pass = result["row_and_patient_coverage"]["pass"]
        width = report["interval"][role]["time_aware_width_delta"]
        width_pass = all(
            item["ci95"] is not None and item["ci95"][1] < 0
            for item in width.values()
        )
        return bool(coverage_pass and width_pass)

    report["gate"] = {
        "eicu_hospital_point": point_pass("hospital_test"),
        "mimic_point": point_pass("mimic"),
        "eicu_hospital_interval": interval_pass("hospital_test"),
        "mimic_interval": interval_pass("mimic"),
    }
    report["cross_database_patient_state_pass"] = bool(all(report["gate"].values()))
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "selected_config": selected_config,
                "support": report["support"],
                "point": report["point"],
                "interval": report["interval"],
                "gate": report["gate"],
                "cross_database_patient_state_pass": report[
                    "cross_database_patient_state_pass"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
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
        default=Path("artifacts/patient_state_external_glucose_6h_20260909.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
