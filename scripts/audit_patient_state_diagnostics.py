"""Localize patient-state failure with fixed support and causal diagnostics."""

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
from scripts.audit_patient_state_external import (
    patient_equal_mae,
    robust_scale,
    support,
)
from scripts.audit_release_masked_cells import input_frame
from scripts.screen_cross_variable_history_catalog import (
    fit_regressor,
    prepare_features,
    target_role_masks,
)
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS, estimate, interval_score
from whole_body_jepa.continual_learning import (
    DecayedPatientResidualState,
    PatientResidualState,
    TimeAwarePatientResidualState,
)
from whole_body_jepa.cross_variable_history import (
    causal_wrong_patient_rows,
    matched_wrong_patient_rows,
)
from whole_body_jepa.multisource_intervals import (
    BOUNDS,
    SEEDS,
    conformal_radius,
    coverage,
    domain_coverage,
    patient_split,
    representative_rows,
    sha256,
    stable_hash,
    subject_digest,
)
from whole_body_jepa.runtime import load_release
from whole_body_jepa.source_selection import HEAD


PROTOCOL = Path("PATIENT_STATE_DIAGNOSTIC_PROTOCOL_20260909.md")
STATE_METHODS = ("level", "decayed_level", "level_slope")
INTERVAL_METHODS = {
    "constant": "constant_scale",
    "level": "level_scale",
    "decayed": "decayed_scale",
}
LEVEL_PRIORS = (1.0, 3.0, 6.0)
DECAY_GRID = tuple(
    {"half_life_hours": float(half_life), "prior_strength": float(prior)}
    for half_life, prior in itertools.product((12, 24, 48), (1, 3))
)
SLOPE_GRID = tuple(
    {
        "half_life_hours": float(half_life),
        "prior_strength": float(prior),
        "slope_penalty": float(penalty),
    }
    for half_life, prior, penalty in itertools.product(
        (12, 24, 48), (1, 3), (4, 16)
    )
)


def ordered_patient_rows(frame: pd.DataFrame):
    subjects = frame.subject_id.astype(str).to_numpy()
    anchors = pd.to_datetime(frame.t).to_numpy()
    for patient in sorted(np.unique(subjects)):
        rows = np.flatnonzero(subjects == patient)
        yield patient, rows[np.argsort(anchors[rows])]


def _resolved_updates(pending: list[dict], anchor: pd.Timestamp) -> tuple[list[dict], list[dict], int]:
    """Resolve one error per future measurement, preferring the latest recent forecast."""

    available = [item for item in pending if item["due"] <= anchor]
    remaining = [item for item in pending if item["due"] > anchor]
    latest: dict[pd.Timestamp, dict] = {}
    for item in available:
        key = item["due"]
        if key not in latest or item["forecast_at"] > latest[key]["forecast_at"]:
            latest[key] = item
    resolved = sorted(latest.values(), key=lambda item: item["due"])
    return resolved, remaining, len(available) - len(resolved)


def prequential_predictions(prepared, configs: dict, reference_scale: float):
    records = []
    duplicate_updates = 0
    cross_stay_patients = 0
    for patient, rows in ordered_patient_rows(prepared.frame):
        if prepared.frame.iloc[rows].stay_id.astype(str).nunique() > 1:
            cross_stay_patients += 1
        level = PatientResidualState(prior_strength=configs["level"]["prior_strength"])
        decayed = DecayedPatientResidualState(
            reference_scale=reference_scale,
            max_history=8,
            **configs["decayed_level"],
        )
        slope = TimeAwarePatientResidualState(
            reference_scale=reference_scale,
            max_history=8,
            **configs["level_slope"],
        )
        pending: list[dict] = []
        for row in rows:
            source = prepared.frame.iloc[row]
            anchor = pd.Timestamp(source.t)
            resolved, pending, duplicates = _resolved_updates(pending, anchor)
            duplicate_updates += duplicates
            for item in resolved:
                level.update(item["residual"])
                decayed.update(item["due"], item["residual"])
                slope.update(item["due"], item["residual"])

            decayed_correction, decayed_scale = decayed.estimate(anchor)
            slope_correction, slope_scale = slope.estimate(anchor, HORIZON)
            level_correction = level.correction
            level_scale = level.uncertainty_scale(reference_scale)
            records.append(
                {
                    "subject_id": str(patient),
                    "stay_id": str(source.stay_id),
                    "t": anchor,
                    "_time_hours": float(source._time_hours),
                    "careunit": str(source.careunit),
                    "resolved_history_count": int(level.count),
                    "current": float(prepared.current[row]),
                    "truth": float(prepared.truth[row]),
                    "frozen_release": float(prepared.base[row]),
                    "persistence": float(prepared.current[row]),
                    "level": float(prepared.base[row] + level_correction),
                    "decayed_level": float(prepared.base[row] + decayed_correction),
                    "level_slope": float(prepared.base[row] + slope_correction),
                    "level_correction": float(level_correction),
                    "decayed_level_correction": float(decayed_correction),
                    "level_slope_correction": float(slope_correction),
                    "constant_scale": float(reference_scale),
                    "level_scale": float(level_scale),
                    "decayed_scale": float(decayed_scale),
                    "level_slope_scale": float(slope_scale),
                }
            )
            event_time = pd.to_datetime(source[f"future_event_time_{TARGET}"])
            if pd.isna(event_time):
                event_time = anchor + pd.to_timedelta(
                    float(source[f"future_delta_t_hr_{TARGET}"]), unit="h"
                )
            if event_time <= anchor:
                raise ValueError("future glucose measurement does not follow anchor")
            pending.append(
                {
                    "due": event_time,
                    "forecast_at": anchor,
                    "residual": float(prepared.truth[row] - prepared.base[row]),
                }
            )
    return pd.DataFrame(records), {
        "duplicate_resolved_forecast_errors_removed": int(duplicate_updates),
        "patients_spanning_multiple_stays": int(cross_stay_patients),
        "result_availability_semantics": "future_event_time_not_laboratory_storetime",
    }


def select_configs(prepared, reference_scale: float) -> tuple[dict, dict]:
    boards: dict[str, list[dict]] = {method: [] for method in STATE_METHODS}
    defaults = {
        "level": {"prior_strength": 3.0},
        "decayed_level": {"half_life_hours": 24.0, "prior_strength": 3.0},
        "level_slope": {
            "half_life_hours": 24.0,
            "prior_strength": 3.0,
            "slope_penalty": 16.0,
        },
    }
    candidates = {
        "level": ({"prior_strength": prior} for prior in LEVEL_PRIORS),
        "decayed_level": iter(DECAY_GRID),
        "level_slope": iter(SLOPE_GRID),
    }
    for method, options in candidates.items():
        for option in options:
            configs = {name: dict(value) for name, value in defaults.items()}
            configs[method] = dict(option)
            frame, _ = prequential_predictions(prepared, configs, reference_scale)
            selected = frame.loc[frame.resolved_history_count > 0]
            boards[method].append(
                {
                    **option,
                    "patient_equal_mae": patient_equal_mae(selected, method),
                    "rows": int(len(selected)),
                    "patients": int(selected.subject_id.nunique()),
                }
            )
        boards[method].sort(key=lambda item: item["patient_equal_mae"])
    selected = {
        method: {
            key: value
            for key, value in boards[method][0].items()
            if key not in {"patient_equal_mae", "rows", "patients"}
        }
        for method in STATE_METHODS
    }
    return selected, boards


def add_causal_placebo(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    donors = causal_wrong_patient_rows(
        frame, seed=stable_hash("patient-state-diagnostic-placebo", seed, CELL)
    )
    result = frame.copy()
    present = donors >= 0
    for method in STATE_METHODS:
        correction = np.zeros(len(frame), dtype=float)
        correction[present] = frame[f"{method}_correction"].to_numpy(float)[donors[present]]
        result[f"placebo_{method}"] = frame.frozen_release.to_numpy(float) + correction
    return result, {
        "seed": int(seed),
        "rows_with_available_donor_state": int(present.sum()),
        "rows_without_available_donor_state": int((~present).sum()),
        "same_support_digest": subject_digest(frame.subject_id),
        "latest_donor_not_after_recipient": True,
    }


def paired_delta(frame: pd.DataFrame, candidate: str, reference: str, *, salt: str, repetitions: int):
    values = np.abs(frame[candidate] - frame.truth) - np.abs(frame[reference] - frame.truth)
    return estimate(
        frame,
        values.to_numpy(float),
        seed=stable_hash("patient-state-diagnostic-bootstrap", salt),
        repetitions=repetitions,
    )


def point_report(frame: pd.DataFrame, *, role: str, repetitions: int) -> dict:
    personalized = frame.loc[frame.resolved_history_count > 0].reset_index(drop=True)
    threshold = float(CHANGE_THRESHOLDS[TARGET])
    changed = np.abs(personalized.truth - personalized.current) >= threshold
    result = {
        "rows": int(len(personalized)),
        "patients": int(personalized.subject_id.nunique()),
        "mae": {
            method: patient_equal_mae(personalized, method)
            for method in ("persistence", "frozen_release", *STATE_METHODS)
        },
        "comparisons": {},
        "retrospective_behavior": {},
    }
    references = {
        "level": ("frozen_release", "persistence"),
        "decayed_level": ("level", "frozen_release", "persistence"),
        "level_slope": ("decayed_level", "level", "frozen_release", "persistence"),
    }
    for method, refs in references.items():
        result["comparisons"][method] = {
            reference: paired_delta(
                personalized,
                method,
                reference,
                salt=f"{role}:{method}:{reference}",
                repetitions=repetitions,
            )
            for reference in refs
        }
    for method in ("persistence", "frozen_release", *STATE_METHODS):
        prediction = personalized[method].to_numpy(float)
        current = personalized.current.to_numpy(float)
        truth = personalized.truth.to_numpy(float)
        result["retrospective_behavior"][method] = {
            "stable_patient_equal_mae": patient_equal_mae(
                personalized.loc[~changed].reset_index(drop=True), method
            ),
            "changed_patient_equal_mae": patient_equal_mae(
                personalized.loc[changed].reset_index(drop=True), method
            ),
            "false_move_rate_on_stable": float(
                np.mean(np.abs(prediction[~changed] - current[~changed]) >= threshold)
            ),
            "direction_accuracy_on_changed": float(
                np.mean(
                    np.sign(prediction[changed] - current[changed])
                    == np.sign(truth[changed] - current[changed])
                )
            ),
        }
    return result


def placebo_report(frame: pd.DataFrame, *, role: str, repetitions: int) -> dict:
    personalized = frame.resolved_history_count > 0
    results = {method: [] for method in STATE_METHODS}
    audits = []
    for seed in SEEDS:
        placebo, audit = add_causal_placebo(frame, seed)
        selected = placebo.loc[personalized].reset_index(drop=True)
        audits.append(audit)
        for method in STATE_METHODS:
            results[method].append(
                {
                    "seed": int(seed),
                    "comparison": paired_delta(
                        selected,
                        method,
                        f"placebo_{method}",
                        salt=f"{role}:{method}:placebo:{seed}",
                        repetitions=repetitions,
                    ),
                }
            )
    return {"support_audit": audits, "methods": results}


def fit_radius(frame: pd.DataFrame, point: str, scale: str, seed: int) -> float:
    selected = frame.loc[frame.resolved_history_count > 0].reset_index(drop=True)
    scores = np.abs(selected[point] - selected.truth).to_numpy(float) / selected[scale].to_numpy(float)
    rows = representative_rows(selected[["subject_id", "stay_id", "t"]], seed)
    return conformal_radius(scores[rows])


def interval_metrics(frame: pd.DataFrame, point: str, scale: str, radius: float) -> dict:
    selected = frame.loc[frame.resolved_history_count > 0].reset_index(drop=True)
    center = selected[point].to_numpy(float)
    truth = selected.truth.to_numpy(float)
    half_width = radius * selected[scale].to_numpy(float)
    normalized = selected[["subject_id", "careunit", "t"]].copy()
    normalized["residual"] = np.abs(center - truth) / selected[scale].to_numpy(float)
    widths = 2.0 * half_width
    scores = interval_score(truth, center - half_width, center + half_width, alpha=0.1)
    return {
        "radius": float(radius),
        "coverage": coverage(normalized, radius),
        "careunit_coverage": domain_coverage(normalized, radius),
        "patient_equal_width": estimate(selected, widths),
        "patient_equal_interval_score": estimate(selected, scores),
    }


def source_interval_report(frames: dict[str, pd.DataFrame]) -> dict:
    radii = {
        method: fit_radius(frames["retention"], "frozen_release", scale, SEEDS[0])
        for method, scale in INTERVAL_METHODS.items()
    }
    return {
        "calibration": "eicu_retention",
        "radii": radii,
        "roles": {
            role: {
                method: interval_metrics(frame, "frozen_release", scale, radii[method])
                for method, scale in INTERVAL_METHODS.items()
            }
            for role, frame in frames.items()
            if role in {"hospital_test", "mimic"}
        },
    }


def target_domain_interval_report(frame: pd.DataFrame) -> dict:
    selected = frame.loc[frame.resolved_history_count > 0].reset_index(drop=True)
    results = []
    for seed in SEEDS:
        calibration, test = patient_split(selected.subject_id, seed)
        split = {"seed": int(seed), "methods": {}}
        for method, scale in INTERVAL_METHODS.items():
            radius = fit_radius(
                selected.loc[calibration].reset_index(drop=True),
                "frozen_release",
                scale,
                seed,
            )
            split["methods"][method] = interval_metrics(
                selected.loc[test].reset_index(drop=True),
                "frozen_release",
                scale,
                radius,
            )
        results.append(split)
    return {
        "status": "diagnostic_target_domain_recalibration_not_external_validation",
        "database": "mimic",
        "patient_splits": results,
    }


def representation_views(frame, arrays, rows, prepared, assets):
    filtered = input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours)
    observations = assets.scaler.normalize_current(filtered)
    selected, raw = prepare_features(
        frame, rows, TARGET, assets, arrays, observations
    )
    own = raw["own_history"].astype(np.float32)
    latent = prepared.features[:, :256].astype(np.float32)
    donors = matched_wrong_patient_rows(
        selected, seed=stable_hash("patient-state-representation-placebo", CELL)
    )
    return selected, {
        "own_history": own,
        "current_cross": raw["current_cross"].astype(np.float32),
        "cross_history": raw["cross_history"].astype(np.float32),
        "jepa_latent": np.column_stack([own, latent]).astype(np.float32),
        "jepa_latent_placebo": np.column_stack([own, latent[donors]]).astype(np.float32),
    }


def representation_report(role_data: dict, *, repetitions: int) -> dict:
    fit_frame, fit_views = role_data["memory"]
    fit_delta = (fit_frame[f"future_{TARGET}"] - fit_frame[f"{TARGET}_t"]).to_numpy(float)
    models = {
        name: fit_regressor(features, fit_delta, fit_frame.subject_id)
        for name, features in fit_views.items()
    }
    report = {"fit_patients": int(fit_frame.subject_id.nunique()), "roles": {}}
    for role in ("patient_test", "hospital_test", "mimic"):
        frame, views = role_data[role]
        current = frame[f"{TARGET}_t"].to_numpy(float)
        truth = frame[f"future_{TARGET}"].to_numpy(float)
        predictions = {
            name: current + models[name].predict(features)
            for name, features in views.items()
        }
        role_result = {
            "rows": int(len(frame)),
            "patients": int(frame.subject_id.nunique()),
            "patient_equal_mae": {},
            "comparisons": {},
        }
        scored = frame[["subject_id", "careunit", "t"]].copy()
        scored["truth"] = truth
        for name, prediction in predictions.items():
            scored[name] = prediction
            role_result["patient_equal_mae"][name] = patient_equal_mae(scored, name)
        comparisons = {
            "current_cross": ("own_history",),
            "cross_history": ("own_history", "current_cross"),
            "jepa_latent": ("own_history", "jepa_latent_placebo"),
        }
        for candidate, references in comparisons.items():
            role_result["comparisons"][candidate] = {
                reference: paired_delta(
                    scored,
                    candidate,
                    reference,
                    salt=f"representation:{role}:{candidate}:{reference}",
                    repetitions=repetitions,
                )
                for reference in references
            }
        report["roles"][role] = role_result
    return report


def run(args) -> None:
    if args.output.exists():
        raise ValueError("refusing to overwrite patient-state diagnostic report")
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
    patient_sets = {**fit_roles, "patient_test": role_patients["patient_test"],
                    "hospital_test": role_patients["hospital_test"]}

    prepared = {}
    selected = {}
    for role, patients in patient_sets.items():
        rows = selected_rows(
            eicu,
            eicu_eligible,
            patients,
            anchors=4 if role == "memory" else 0,
            salt=f"patient-state-external:{role}",
        )
        selected[role] = rows
        prepared[role] = prepare(eicu, eicu_arrays, rows, assets)
    mimic_patients = set(mimic.loc[mimic_eligible, "subject_id"].astype(str))
    mimic_rows = selected_rows(
        mimic,
        mimic_eligible,
        mimic_patients,
        anchors=0,
        salt="patient-state-external:mimic",
    )
    selected["mimic"] = mimic_rows
    prepared["mimic"] = prepare(mimic, mimic_arrays, mimic_rows, assets)

    reference_scale = robust_scale(prepared["memory"].truth - prepared["memory"].base)
    configs, leaderboards = select_configs(prepared["patient_test"], reference_scale)
    frames = {}
    replay_audit = {}
    for role, data in prepared.items():
        frames[role], replay_audit[role] = prequential_predictions(
            data, configs, reference_scale
        )

    state_report = {
        "selected_configs": configs,
        "selection_leaderboards": leaderboards,
        "roles": {
            role: point_report(frame, role=role, repetitions=args.bootstrap)
            for role, frame in frames.items()
            if role in {"patient_test", "hospital_test", "mimic"}
        },
        "placebos": {
            role: placebo_report(frame, role=role, repetitions=args.bootstrap)
            for role, frame in frames.items()
            if role in {"patient_test", "hospital_test", "mimic"}
        },
    }

    role_data = {}
    for role in ("memory", "patient_test", "hospital_test"):
        role_data[role] = representation_views(
            eicu, eicu_arrays, selected[role], prepared[role], assets
        )
    role_data["mimic"] = representation_views(
        mimic, mimic_arrays, selected["mimic"], prepared["mimic"], assets
    )

    legacy = json.loads(args.legacy_report.read_text())
    report = {
        "schema": "whole_body_jepa.patient_state_diagnostics.v1",
        "complete": True,
        "status": "diagnostic_only",
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "cell": CELL,
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": sha256(args.model_dir / "whole_body_joint_jepa.pt"),
        "inputs": {"eicu": sha256(args.eicu), "mimic": sha256(args.mimic)},
        "hospital_role": hospital_metadata,
        "support": {role: support(frame) for role, frame in frames.items()},
        "evaluation_integrity": {
            "legacy_findings_claimed_all_placebos_failed": True,
            "legacy_json_placebo_pass_counts": {
                role: sum(
                    item["comparison"]["ci95"] is not None
                    and item["comparison"]["ci95"][1] < 0
                    for item in legacy["placebo_seeds"][role]
                )
                for role in ("patient_test", "hospital_test", "mimic")
            },
            "legacy_primary_and_seed_placebo_support_were_different": True,
            "corrected_contract": "full_role_donors_then_identical_personalized_support",
            "causal_relative_time_matching": True,
            "replay": replay_audit,
        },
        "patient_state_components": state_report,
        "uncertainty_only": {
            "point_source_fixed": "frozen_release",
            "source_calibrated": source_interval_report(frames),
            "mimic_recalibrated": target_domain_interval_report(frames["mimic"]),
            "coverage_bounds": list(BOUNDS),
        },
        "representation_location": representation_report(
            role_data, repetitions=args.bootstrap
        ),
        "known_limitation": (
            "future_event_time is the measurement event time; laboratory result "
            "store/availability time is absent from these cached examples"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "complete": True,
        "selected_configs": configs,
        "evaluation_integrity": report["evaluation_integrity"],
        "patient_state_components": state_report["roles"],
        "uncertainty_only": report["uncertainty_only"],
        "representation_location": report["representation_location"],
    }, indent=2), flush=True)


def main() -> None:
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
        "--legacy-report",
        type=Path,
        default=Path("artifacts/patient_state_external_glucose_6h_20260909.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/patient_state_diagnostics_glucose_6h_20260909.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
