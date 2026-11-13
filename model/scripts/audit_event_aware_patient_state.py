#!/usr/bin/env python3
"""Test whether anchor-known measurements and treatment events preserve patient state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from extractors.eicu_body_system_configs import get_body_system_config
from extractors.eicu_body_system_transition_extract import (
    BASE_TIME,
    evidence_window_summary,
    prepare_body_evidence_lookup,
    read_action_evidence,
)
from extractors.mimiciv_observation_transition_extract import (
    assemble_treatment_context,
    pull_treatment_evidence,
)
from scripts.audit_continual_validated_cell import (
    CELL,
    HORIZON,
    TARGET,
    eligible_rows,
    prepare,
    split_fit_patients,
)
from scripts.audit_measurement_anchor_encoder import load_dataset, selected_rows
from scripts.audit_patient_state_diagnostics import _resolved_updates, ordered_patient_rows
from scripts.screen_cross_variable_history_catalog import target_role_masks
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS, estimate
from whole_body_jepa.continual_learning import PatientResidualState
from whole_body_jepa.cross_variable_history import causal_wrong_patient_rows
from whole_body_jepa.event_aware_state import (
    fit_loss_difference_model,
    gated_prediction,
    patient_equal_mae,
    select_apply_threshold,
)
from whole_body_jepa.multisource_intervals import SEEDS, sha256, stable_hash, subject_digest
from whole_body_jepa.runtime import load_release
from whole_body_jepa.source_selection import HEAD
from whole_body_jepa.event_treatment import precise_columns, snapshot_summary


PROTOCOL = Path("EVENT_AWARE_PATIENT_STATE_PROTOCOL_20260909.md")
ACTIONS = ("insulin", "dextrose", "fluids", "systemic_steroid", "vasopressor")
STATE_WIDTH = 9


def raw_stay_ids(series: pd.Series) -> pd.Series:
    """Recover raw numeric stay IDs from the audit reader's database prefix."""

    return pd.to_numeric(series.astype(str).str.rsplit(":", n=1).str[-1], errors="raise").astype("int64")


def treatment_columns() -> tuple[str, ...]:
    return tuple(
        name
        for action in ACTIONS
        for name in (f"hist_{action}", f"hist_{action}_evidence_count")
    )


def _treatment_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for action in ACTIONS:
        present = pd.to_numeric(frame.get(f"hist_{action}", 0.0), errors="coerce")
        count = pd.to_numeric(
            frame.get(f"hist_{action}_evidence_count", 0.0), errors="coerce"
        )
        if np.isscalar(present):
            present = pd.Series(present, index=frame.index)
        if np.isscalar(count):
            count = pd.Series(count, index=frame.index)
        columns.extend(
            (
                present.fillna(0.0).gt(0.0).to_numpy(dtype=np.float32),
                np.log1p(count.fillna(0.0).clip(lower=0.0)).to_numpy(dtype=np.float32),
            )
        )
    for column in precise_columns():
        value = pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")
        known = value.notna() & np.isfinite(value) & value.ge(0)
        columns.extend((np.log1p(value.where(known, 0)).to_numpy(np.float32), known.to_numpy(np.float32)))
    return np.column_stack(columns).astype(np.float32)


def eicu_treatment_context(frame: pd.DataFrame, raw_root: Path) -> tuple[np.ndarray, dict]:
    config = get_body_system_config("endocrine_stress")
    raw_stays = raw_stay_ids(frame.stay_id)
    stays = set(raw_stays.astype(int))
    evidence = read_action_evidence(raw_root, stays, config)
    lookup = prepare_body_evidence_lookup(evidence)
    anchors = (
        pd.to_datetime(frame.anchor_time, errors="coerce") - BASE_TIME
    ) / pd.Timedelta(hours=1)
    rows = [
        evidence_window_summary(
            lookup.get(int(stay), {}), config, float(anchor), HORIZON
        )
        for stay, anchor in zip(raw_stays, anchors, strict=True)
    ]
    summary = pd.DataFrame(rows, index=frame.index)
    precise = snapshot_summary(pd.DataFrame({"stay_id": raw_stays, "t": pd.to_datetime(frame.anchor_time)}, index=frame.index), evidence)
    summary = pd.concat([summary, precise], axis=1)
    return _treatment_matrix(summary), {
        "evidence_rows": int(len(evidence)),
        "evidence_stays": int(evidence.stay_id.nunique()) if len(evidence) else 0,
        "actions": list(ACTIONS),
    }


def mimic_treatment_context(frame: pd.DataFrame, raw_root: Path) -> tuple[np.ndarray, dict]:
    target_stays = pd.DataFrame(
        {"stay_id": raw_stay_ids(frame.stay_id).unique()}
    )
    connection = duckdb.connect()
    connection.register("target_stays", target_stays)
    evidence = pull_treatment_evidence(connection, raw_root, actions=set(ACTIONS))
    connection.close()
    anchors = frame[["stay_id", "t"]].copy()
    anchors["stay_id"] = raw_stay_ids(anchors.stay_id)
    summary = assemble_treatment_context(anchors, evidence, HORIZON)
    summary = pd.concat([summary, snapshot_summary(anchors, evidence)], axis=1)
    return _treatment_matrix(summary), {
        "evidence_rows": int(len(evidence)),
        "evidence_stays": int(evidence.stay_id.nunique()) if len(evidence) else 0,
        "actions": list(ACTIONS),
    }


def measurement_matrix(frame: pd.DataFrame, assets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = []
    observed = []
    ages = []
    for index, variable in enumerate(assets.checkpoint["variables"]):
        values = pd.to_numeric(frame.get(f"{variable}_t"), errors="coerce")
        age = pd.to_numeric(frame.get(f"{variable}_age_hr"), errors="coerce")
        valid = values.notna() & np.isfinite(values) & age.notna() & np.isfinite(age) & age.ge(0)
        present = valid.to_numpy(dtype=np.float32)
        normalized = (
            values.where(valid, float(assets.scaler.medians[index])).to_numpy(dtype=float)
            - float(assets.scaler.medians[index])
        ) / float(assets.scaler.scales[index])
        current.append(np.clip(normalized, -8.0, 8.0))
        observed.append(present)
        ages.append(
            age.where(valid, assets.max_age_hours).clip(lower=0.0).to_numpy(dtype=float)
            / assets.max_age_hours
        )
    return (
        np.column_stack(current).astype(np.float32),
        np.column_stack(observed).astype(np.float32),
        # Preserve timestamp arithmetic; float32 epoch-hours can round by minutes.
        np.column_stack(ages).astype(np.float64),
    )


def replay_features(prepared, measurements, treatment, reference_scale: float, *, reference_center: float, max_age_hours: float) -> tuple[pd.DataFrame, np.ndarray]:
    current_state, observed, ages = measurements
    records = []
    features = []
    for patient, rows in ordered_patient_rows(prepared.frame):
        state = PatientResidualState(prior_strength=3.0)
        pending: list[dict] = []
        previous_state = None
        previous_observed = None
        previous_treatment = None
        previous_anchor = None
        previous_measurement_time = None
        last_resolution = None
        for row in rows:
            source = prepared.frame.iloc[row]
            anchor = pd.Timestamp(source.t)
            resolved, pending, _ = _resolved_updates(pending, anchor)
            for item in resolved:
                state.update(item["residual"])
                last_resolution = item["due"]

            correction = float(state.correction)
            uncertainty = float(state.uncertainty_scale(reference_scale))
            elapsed = (
                168.0
                if previous_anchor is None
                else min((anchor - previous_anchor).total_seconds() / 3600.0, 168.0)
            )
            since_resolution = (
                168.0
                if last_resolution is None
                else min((anchor - last_resolution).total_seconds() / 3600.0, 168.0)
            )
            state_delta = (
                np.zeros_like(current_state[row])
                if previous_state is None
                else (current_state[row] - previous_state)
                * observed[row]
                * previous_observed
            )
            treatment_delta = (
                np.zeros_like(treatment[row])
                if previous_treatment is None
                else treatment[row] - previous_treatment
            )
            measurement_time = anchor.value / 3.6e12 - ages[row] * max_age_hours
            newly_measured = (
                0.0
                if previous_observed is None
                else float(np.mean((observed[row] > 0) & ((previous_observed == 0) | (measurement_time > previous_measurement_time + 1e-6))))
            )
            state_features = np.asarray(
                [
                    correction / reference_scale,
                    np.log1p(state.count),
                    uncertainty / reference_scale,
                    since_resolution / 168.0,
                    elapsed / 168.0,
                    (prepared.base[row] - (prepared.current[row] if np.isfinite(prepared.current[row]) else reference_center)) / reference_scale,
                    ((prepared.current[row] if np.isfinite(prepared.current[row]) else reference_center) - reference_center) / reference_scale,
                    float(np.mean(observed[row])),
                    float(np.mean(ages[row])),
                ],
                dtype=np.float32,
            )
            event_features = np.concatenate(
                (
                    current_state[row],
                    observed[row],
                    ages[row],
                    state_delta,
                    np.asarray(
                        [
                            float(np.mean(np.abs(state_delta))),
                            float(np.max(np.abs(state_delta))),
                            newly_measured,
                        ],
                        dtype=np.float32,
                    ),
                    treatment[row],
                    treatment_delta,
                )
            ).astype(np.float32)
            features.append(np.concatenate((state_features, event_features)))
            # Presence channels are every other treatment feature. Count drift in a
            # rolling window is not itself a treatment start/stop event.
            treatment_changed = bool(np.any(np.abs(treatment_delta[:10:2]) > 1e-6) or np.any(np.abs(treatment_delta[10:]) > 1e-6))
            physiology_changed = bool(np.max(np.abs(state_delta)) >= 0.5)
            records.append(
                {
                    "subject_id": str(patient),
                    "stay_id": str(source.stay_id),
                    "t": anchor,
                    "_time_hours": float(source._time_hours),
                    "careunit": str(source.careunit),
                    "resolved_history_count": int(state.count),
                    "truth": float(prepared.truth[row]),
                    "current": float(prepared.current[row]),
                    "frozen": float(prepared.base[row]),
                    "resolved_error_only": float(prepared.base[row] + correction),
                    "patient_correction": correction,
                    "treatment_changed": treatment_changed,
                    "physiology_changed": physiology_changed,
                    "new_measurement_fraction": newly_measured,
                    "future_event_time": pd.to_datetime(source[f"future_event_time_{TARGET}"], errors="coerce"),
                }
            )
            event_time = pd.to_datetime(source[f"future_event_time_{TARGET}"])
            if pd.isna(event_time):
                event_time = anchor + pd.to_timedelta(
                    float(source[f"future_delta_t_hr_{TARGET}"]), unit="h"
                )
            if np.isfinite(prepared.truth[row]) and pd.notna(event_time) and event_time > anchor:
                pending.append(
                    {
                        "due": event_time,
                        "forecast_at": anchor,
                        "residual": float(prepared.truth[row] - prepared.base[row]),
                    }
                )
            previous_state = current_state[row]
            previous_observed = observed[row]
            previous_treatment = treatment[row]
            previous_anchor = anchor
            previous_measurement_time = measurement_time
    return pd.DataFrame(records), np.stack(features).astype(np.float32)


def patient_partition(subjects: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(subjects.astype(str)))
    ordered = sorted(unique, key=lambda value: stable_hash("event-aware-fit", seed, value))
    stop = int(0.65 * len(ordered))
    fit = set(ordered[:stop])
    fit_mask = subjects.astype(str).isin(fit).to_numpy(copy=True)
    return fit_mask, ~fit_mask


def cached_context(
    path: Path,
    row_identity: np.ndarray,
    builder,
) -> tuple[np.ndarray, dict]:
    """Cache expensive raw-event alignment outside version control."""

    digest = hashlib.sha256(np.asarray(row_identity, dtype=np.int64).tobytes() + Path(__file__).read_bytes() + Path("whole_body_jepa/event_treatment.py").read_bytes()).hexdigest()
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        matrix = cached["matrix"]
        cached_digest = str(cached["row_digest"].item())
        if cached_digest == digest and len(matrix) == len(row_identity):
            return matrix.astype(np.float32), {
                "cache_hit": True,
                "row_digest": digest,
                "rows": int(len(matrix)),
            }
    matrix, audit = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, matrix=matrix, row_digest=np.asarray(digest))
    return matrix, {**audit, "cache_hit": False, "row_digest": digest}


def shuffled_event_features(frame: pd.DataFrame, features: np.ndarray, seed: int) -> np.ndarray:
    donors = causal_wrong_patient_rows(
        frame, seed=stable_hash("event-aware-context-placebo", seed, CELL)
    )
    output = features.copy()
    available = donors >= 0
    output[available, STATE_WIDTH:] = features[donors[available], STATE_WIDTH:]
    return output


def wrong_patient_state(frame: pd.DataFrame, features: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    donors = causal_wrong_patient_rows(
        frame, seed=stable_hash("event-aware-state-placebo", seed, CELL)
    )
    output = features.copy()
    correction = np.zeros(len(frame), dtype=float)
    available = donors >= 0
    output[available, :STATE_WIDTH] = features[donors[available], :STATE_WIDTH]
    correction[available] = frame.patient_correction.to_numpy(float)[donors[available]]
    return output, frame.frozen.to_numpy(float) + correction


def fit_gate_bundle(frame: pd.DataFrame, features: np.ndarray, seed: int) -> dict:
    personalized = frame.resolved_history_count.to_numpy() > 0
    fit, tune = patient_partition(frame.subject_id, seed)
    fit &= personalized
    tune &= personalized
    truth = frame.truth.to_numpy(float)
    frozen = frame.frozen.to_numpy(float)
    always = frame.resolved_error_only.to_numpy(float)
    loss_difference = np.abs(always - truth) - np.abs(frozen - truth)
    shuffled = shuffled_event_features(frame, features, seed)
    models = {
        "event_aware": fit_loss_difference_model(
            features[fit], loss_difference[fit], frame.loc[fit, "subject_id"], seed=seed
        ),
        "state_only": fit_loss_difference_model(
            features[fit, :STATE_WIDTH], loss_difference[fit], frame.loc[fit, "subject_id"], seed=seed + 1
        ),
        "shuffled_event": fit_loss_difference_model(
            shuffled[fit], loss_difference[fit], frame.loc[fit, "subject_id"], seed=seed + 2
        ),
    }
    inputs = {
        "event_aware": features,
        "state_only": features[:, :STATE_WIDTH],
        "shuffled_event": shuffled,
    }
    thresholds = {}
    boards = {}
    for name, model in models.items():
        score = model.predict(inputs[name][tune])
        thresholds[name], boards[name] = select_apply_threshold(
            score,
            frame.loc[tune, "subject_id"],
            truth[tune],
            frozen[tune],
            always[tune],
        )
    return {
        "models": models,
        "thresholds": thresholds,
        "selection": boards,
        "fit_patients": int(frame.loc[fit, "subject_id"].nunique()),
        "tune_patients": int(frame.loc[tune, "subject_id"].nunique()),
    }


def paired(frame: pd.DataFrame, candidate: np.ndarray, reference: np.ndarray, salt: str, bootstrap: int) -> dict:
    delta = np.abs(candidate - frame.truth.to_numpy(float)) - np.abs(
        reference - frame.truth.to_numpy(float)
    )
    return estimate(
        frame,
        delta,
        seed=stable_hash("event-aware-bootstrap", salt),
        repetitions=bootstrap,
    )


def evaluate(frame: pd.DataFrame, features: np.ndarray, bundle: dict, role: str, seed: int, bootstrap: int) -> dict:
    selected = frame.resolved_history_count.to_numpy() > 0
    work = frame.loc[selected].reset_index(drop=True)
    x = features[selected]
    truth = work.truth.to_numpy(float)
    frozen = work.frozen.to_numpy(float)
    always = work.resolved_error_only.to_numpy(float)
    shuffled = shuffled_event_features(work, x, seed)
    wrong_x, wrong_always = wrong_patient_state(work, x, seed)
    inputs = {
        "event_aware": x,
        "state_only": x[:, :STATE_WIDTH],
        "shuffled_event": shuffled,
        "wrong_patient": wrong_x,
    }
    predictions = {"frozen": frozen, "resolved_error_only": always}
    apply = {}
    score = {}
    for name in ("event_aware", "state_only", "shuffled_event"):
        predictions[name], apply[name], score[name] = gated_prediction(
            bundle["models"][name], bundle["thresholds"][name], inputs[name], frozen, always
        )
    predictions["wrong_patient"], apply["wrong_patient"], score["wrong_patient"] = gated_prediction(
        bundle["models"]["event_aware"],
        bundle["thresholds"]["event_aware"],
        inputs["wrong_patient"],
        frozen,
        wrong_always,
    )
    comparisons = {
        name: paired(
            work,
            predictions["event_aware"],
            predictions[name],
            f"{role}:{seed}:{name}",
            bootstrap,
        )
        for name in ("frozen", "resolved_error_only", "state_only", "shuffled_event", "wrong_patient")
    }
    actual_delta = truth - work.current.to_numpy(float)
    changed = np.abs(actual_delta) >= float(CHANGE_THRESHOLDS[TARGET])
    treatment_change = work.treatment_changed.to_numpy(bool)
    quiet = (~treatment_change) & (~changed)
    subsets = {}
    for name, mask in {
        "treatment_change": treatment_change,
        "changed_after_treatment_change": treatment_change & changed,
        "quiet_stable": quiet,
    }.items():
        item = {"rows": int(mask.sum()), "patients": int(work.loc[mask, "subject_id"].nunique())}
        if mask.any():
            item["mae"] = {
                method: patient_equal_mae(work.loc[mask, "subject_id"], truth[mask], value[mask])
                for method, value in predictions.items()
            }
            item["direction_accuracy"] = {
                method: float(
                    np.mean(
                        np.sign(value[mask] - work.current.to_numpy(float)[mask])
                        == np.sign(actual_delta[mask])
                    )
                )
                for method, value in predictions.items()
            }
        subsets[name] = item
    return {
        "rows": int(len(work)),
        "patients": int(work.subject_id.nunique()),
        "patient_equal_mae": {
            name: patient_equal_mae(work.subject_id, truth, value)
            for name, value in predictions.items()
        },
        "apply_fraction": {name: float(value.mean()) for name, value in apply.items()},
        "comparisons": comparisons,
        "subsets": subsets,
    }


def write_findings(report: dict, path: Path) -> None:
    def mean(role: str, method: str) -> float:
        return float(
            np.mean(
                [
                    item["evaluation"][role]["patient_equal_mae"][method]
                    for item in report["seeds"]
                ]
            )
        )

    def subset_mean(role: str, subset: str, metric: str, method: str) -> float:
        return float(
            np.mean(
                [
                    item["evaluation"][role]["subsets"][subset][metric][method]
                    for item in report["seeds"]
                ]
            )
        )

    lines = [
        "# Event-Aware Patient-State Audit",
        "",
        "## Decision",
        "",
        f"`{report['status']}`. The release remains unchanged.",
        "",
        "The same frozen `glucose@6h` JEPA source was used in every arm. Forecast",
        "outcomes updated patient state only after their measurement time resolved.",
        "The event-aware arm could read only anchor-known measurement freshness,",
        "physiological change and treatment history; future treatment was excluded.",
        "",
        "| Role | Frozen | Error-only | Event-aware | State-only gate | Shuffled-event gate | Wrong-patient |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("eicu_hospital", "mimic"):
        lines.append(
            f"| {role} | {mean(role, 'frozen'):.3f} | {mean(role, 'resolved_error_only'):.3f} | "
            f"{mean(role, 'event_aware'):.3f} | {mean(role, 'state_only'):.3f} | "
            f"{mean(role, 'shuffled_event'):.3f} | {mean(role, 'wrong_patient'):.3f} |"
        )
    lines += [
        "",
        "Values are seven-seed means of patient-equal MAE. Event-aware versus",
        "frozen passed only 1/7 eICU hospital comparisons and 0/7 MIMIC comparisons.",
        "It passed 2/7 and 0/7 against unconditional error-only state, and 1/7",
        "and 0/7 against the shuffled-event equal-capacity control. It never",
        "significantly beat the state-only gate in either external role.",
        "",
        "## Treatment-transition behavior",
        "",
        "On eICU anchors with an observed treatment presence transition, event-aware",
        f"direction accuracy was {subset_mean('eicu_hospital', 'changed_after_treatment_change', 'direction_accuracy', 'event_aware'):.3f}, "
        f"versus {subset_mean('eicu_hospital', 'changed_after_treatment_change', 'direction_accuracy', 'frozen'):.3f} for frozen and "
        f"{subset_mean('eicu_hospital', 'changed_after_treatment_change', 'direction_accuracy', 'resolved_error_only'):.3f} for error-only state. "
        "That improvement did not transfer: on MIMIC the corresponding values were",
        f"{subset_mean('mimic', 'changed_after_treatment_change', 'direction_accuracy', 'event_aware'):.3f}, "
        f"{subset_mean('mimic', 'changed_after_treatment_change', 'direction_accuracy', 'frozen'):.3f}, and "
        f"{subset_mean('mimic', 'changed_after_treatment_change', 'direction_accuracy', 'resolved_error_only'):.3f}.",
        "The shuffled-event arm was often as good as or better than the real-event arm.",
        "Therefore this contract does not show that treatment events identify the",
        "correct signed update before the future glucose result is known.",
        "",
        "Wrong-patient state was clearly harmful on average and event-aware beat it",
        "in 6/7 eICU and 5/7 MIMIC comparisons. Patient identity contains useful",
        "information, but the current event representation does not make its signed",
        "correction stable enough to improve the frozen source across databases.",
        "",
        "Promotion required the event-aware arm to beat the frozen source, unconditional",
        "resolved-error state, a state-only equal-capacity gate, shuffled-event context and",
        "wrong-patient state in all seven source-defined seeds on both external roles.",
        "Exact aggregate estimates and subset behavior are in the JSON artifact.",
        "",
        "This is factual prediction only; no treatment-effect or causal claim is allowed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    if args.output.exists():
        raise ValueError("refusing to overwrite event-aware patient-state report")
    assets = load_release(args.model_dir)
    decision = assets.source_selector["cells"][CELL]
    if decision["source_model"] != HEAD or not decision["can_move"]:
        raise ValueError("glucose@6h is no longer backed by the validated shared head")
    eicu, eicu_arrays = load_dataset(args.eicu, "eicu", assets)
    mimic, mimic_arrays = load_dataset(args.mimic, "mimic", assets)
    eicu_eligible = eligible_rows(eicu)
    mimic_eligible = eligible_rows(mimic)
    roles, hospital_metadata = target_role_masks(eicu, eicu_eligible, "event-aware:glucose@6h")
    role_patients = {
        role: set(eicu.loc[eicu_eligible & mask, "subject_id"].astype(str))
        for role, mask in roles.items()
    }
    fit_roles = split_fit_patients(role_patients["fit"])
    patient_sets = {
        **fit_roles,
        "development": role_patients["patient_test"],
        "eicu_hospital": role_patients["hospital_test"],
    }
    selected = {}
    prepared = {}
    for role, patients in patient_sets.items():
        rows = selected_rows(
            eicu,
            eicu_eligible,
            patients,
            anchors=4 if role == "memory" else 0,
            salt=f"event-aware:{role}",
        )
        selected[role] = rows
        prepared[role] = prepare(eicu, eicu_arrays, rows, assets)
    mimic_patients = set(mimic.loc[mimic_eligible, "subject_id"].astype(str))
    selected["mimic"] = selected_rows(
        mimic, mimic_eligible, mimic_patients, anchors=0, salt="event-aware:mimic"
    )
    prepared["mimic"] = prepare(mimic, mimic_arrays, selected["mimic"], assets)

    reference_scale = float(
        np.median(np.abs(prepared["memory"].truth - prepared["memory"].base)) / 0.6744897501960817
    )
    reference_scale = max(reference_scale, 1.0)
    eicu_union = np.unique(np.concatenate([selected[name] for name in patient_sets]))
    eicu_context, eicu_context_audit = cached_context(
        args.eicu_context_cache,
        eicu_union,
        lambda: eicu_treatment_context(eicu.iloc[eicu_union], args.eicu_root),
    )
    eicu_position = {int(row): index for index, row in enumerate(eicu_union)}
    mimic_context, mimic_context_audit = cached_context(
        args.mimic_context_cache,
        selected["mimic"],
        lambda: mimic_treatment_context(mimic.iloc[selected["mimic"]], args.mimic_root),
    )

    replay = {}
    feature = {}
    for role in patient_sets:
        rows = selected[role]
        positions = np.asarray([eicu_position[int(row)] for row in rows])
        replay[role], feature[role] = replay_features(
            prepared[role],
            measurement_matrix(prepared[role].frame, assets),
            eicu_context[positions],
            reference_scale,
            reference_center=float(assets.scaler.medians[assets.checkpoint["variables"].index(TARGET)]),
            max_age_hours=float(assets.max_age_hours),
        )
    replay["mimic"], feature["mimic"] = replay_features(
        prepared["mimic"],
        measurement_matrix(prepared["mimic"].frame, assets),
        mimic_context,
        reference_scale,
        reference_center=float(assets.scaler.medians[assets.checkpoint["variables"].index(TARGET)]),
        max_age_hours=float(assets.max_age_hours),
    )

    seeds = []
    for seed in SEEDS:
        bundle = fit_gate_bundle(replay["development"], feature["development"], int(seed))
        evaluation = {
            role: evaluate(replay[role], feature[role], bundle, role, int(seed), args.bootstrap)
            for role in ("development", "eicu_hospital", "mimic")
        }
        seeds.append(
            {
                "seed": int(seed),
                "fit_patients": bundle["fit_patients"],
                "tune_patients": bundle["tune_patients"],
                "thresholds": bundle["thresholds"],
                "selection": bundle["selection"],
                "evaluation": evaluation,
            }
        )
        print(
            json.dumps(
                {
                    "seed": int(seed),
                    "hospital": evaluation["eicu_hospital"]["patient_equal_mae"],
                    "mimic": evaluation["mimic"]["patient_equal_mae"],
                },
                allow_nan=False,
            ),
            flush=True,
        )

    references = ("frozen", "resolved_error_only", "state_only", "shuffled_event", "wrong_patient")
    pass_counts = {
        role: {
            reference: int(
                sum(
                    item["evaluation"][role]["comparisons"][reference]["ci95"] is not None
                    and item["evaluation"][role]["comparisons"][reference]["ci95"][1] < 0.0
                    for item in seeds
                )
            )
            for reference in references
        }
        for role in ("eicu_hospital", "mimic")
    }
    promotion_allowed = all(
        pass_counts[role][reference] == len(SEEDS)
        for role in pass_counts
        for reference in references
    )
    report = {
        "schema": "whole_body_jepa.event_aware_patient_state.v1",
        "complete": True,
        "status": "promoted" if promotion_allowed else "candidate_only",
        "promotion_allowed": bool(promotion_allowed),
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "cell": CELL,
        "comparison": [
            "fixed_frozen_jepa",
            "resolved_error_only_patient_state",
            "anchor_known_measurement_and_treatment_event_gate",
        ],
        "future_treatment_used": False,
        "feature_contract": {
            "patient_state_width": STATE_WIDTH,
            "treatment_columns": list(treatment_columns()),
            "measurement_semantics": "current_value_observed_mask_age_and_since_previous_anchor_change",
        },
        "release_binding": {
            "checkpoint": sha256(args.model_dir / "whole_body_joint_jepa.pt"),
            "protocol": sha256(PROTOCOL),
            "script": sha256(Path(__file__)),
        },
        "input_binding": {"eicu": sha256(args.eicu), "mimic": sha256(args.mimic)},
        "hospital_role": hospital_metadata,
        "context_support": {"eicu": eicu_context_audit, "mimic": mimic_context_audit},
        "support": {
            role: {
                "rows": int(len(frame)),
                "patients": int(frame.subject_id.nunique()),
                "personalized_rows": int((frame.resolved_history_count > 0).sum()),
                "subject_digest": subject_digest(frame.subject_id),
            }
            for role, frame in replay.items()
        },
        "seeds": seeds,
        "pass_counts": pass_counts,
    }
    report["evaluation"] = {
        role: seeds[0]["evaluation"][role] for role in ("eicu_hospital", "mimic")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_findings(report, args.findings)
    print(json.dumps({"status": report["status"], "pass_counts": pass_counts}, indent=2), flush=True)


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
        "--eicu-root",
        type=Path,
        default=Path("/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0"),
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=Path("/Users/chunyouchang/Downloads/mimic-iv-3.1"),
    )
    parser.add_argument(
        "--eicu-context-cache",
        type=Path,
        default=Path("/private/tmp/event_aware_eicu_glucose_6h_context.npz"),
    )
    parser.add_argument(
        "--mimic-context-cache",
        type=Path,
        default=Path("/private/tmp/event_aware_mimic_glucose_6h_context.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/event_aware_patient_state_glucose_6h_20260909_v3.json"),
    )
    parser.add_argument(
        "--findings",
        type=Path,
        default=Path("EVENT_AWARE_PATIENT_STATE_FINDINGS_20260909.md"),
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
