"""Development-only eICU audit for stable patient phenotype adaptation.

This script never reads MIMIC-IV. It combines causal trajectory features with
pre-anchor demographics, comorbidity documentation, or measurement-history
summaries. Target adapters are compared with the release base, row-level and
trajectory-only subtyping, and a matched patient-shuffled placebo under
patient, unseen-hospital, and forward-time splits. Passing this audit is not
external promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.validate_subtype_external import _load_cell, stable_seed
from whole_body_jepa.runtime import load_release
from whole_body_jepa.subtypes import (
    SoftPhysiologySubtypeModel,
    causal_history_covariates,
    causal_trajectory_features,
)


TARGET_CONFIG = {
    "map@6h": {"window_hours": 6.0, "current_weight": 0.75},
    "map@12h": {"window_hours": 12.0, "current_weight": 0.50},
    "respiratory_rate@6h": {"window_hours": 6.0, "current_weight": 0.75},
}

BACKGROUND_CONDITIONS = {
    "diabetes": r"diabet",
    "ckd": r"chronic kidney|chronic renal|renal failure|\bckd\b",
    "heart_failure": r"heart failure|congestive heart|\bchf\b",
    "copd": r"chronic obstructive|\bcopd\b",
}

CLINICAL_EVENT_PATTERNS = {
    "hemodynamic": r"shock|hypotens|vasopress|pressor|hemodynamic",
    "respiratory_support": r"intubat|ventilat|airway|oxygen|hypoxi|respiratory distress",
    "infection": r"sepsis|septic|infection|fever|antibiotic",
    "renal_support": r"dialysis|renal replacement|oliguri|anuri",
    "neurologic": r"altered mental|confus|deliri|sedat|gcs|unresponsive",
    "fluid_resuscitation": r"fluid bolus|volume resusc|crystalloid|colloid",
}

CLINICAL_EVENT_TABLES = (
    (
        "treatment.csv.gz",
        ("treatmentoffset",),
        ("treatmentstring",),
    ),
    (
        "carePlanGeneral.csv.gz",
        ("cplitemoffset",),
        ("cplgroup", "cplitemvalue"),
    ),
    (
        "carePlanGoal.csv.gz",
        ("cplgoaloffset",),
        ("cplgoalcategory", "cplgoalvalue", "cplgoalstatus"),
    ),
    (
        "nurseAssessment.csv.gz",
        ("nurseassessoffset", "nurseassessentryoffset"),
        ("cellattributepath", "celllabel", "cellattribute", "cellattributevalue"),
    ),
    (
        "physicalExam.csv.gz",
        ("physicalexamoffset",),
        ("physicalexampath", "physicalexamvalue", "physicalexamtext"),
    ),
    (
        "respiratoryCharting.csv.gz",
        ("respchartoffset", "respchartentryoffset"),
        ("respcharttypecat", "respchartvaluelabel", "respchartvalue"),
    ),
)

FUTURE_EVENT_POLICY = {
    "future_event_value_imputation_allowed": False,
    "runtime_action_without_external_validation": "validated_fallback_or_abstain",
    "uncertainty_action": "retain_calibrated_interval_or_abstain",
    "clinical_event_context_is_observation_only": True,
}


def _numeric_age(value: object) -> float:
    text = str(value).strip()
    if text.startswith(">"):
        text = text[1:].strip()
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else np.nan


def _earliest_offsets(
    path: Path,
    *,
    stay_ids: set[int],
    text_column: str,
    offset_column: str,
) -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    conditions = {name: {} for name in BACKGROUND_CONDITIONS}
    earliest_any: dict[int, float] = {}
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=["patientunitstayid", text_column, offset_column],
        chunksize=250_000,
        low_memory=False,
    ):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)].copy()
        if chunk.empty:
            continue
        chunk[offset_column] = pd.to_numeric(chunk[offset_column], errors="coerce")
        chunk = chunk[np.isfinite(chunk[offset_column])]
        text = chunk[text_column].fillna("").astype(str)
        any_offsets = chunk.groupby("patientunitstayid")[offset_column].min()
        for stay_id, offset in any_offsets.items():
            earliest_any[int(stay_id)] = min(
                earliest_any.get(int(stay_id), float("inf")), float(offset)
            )
        for name, pattern in BACKGROUND_CONDITIONS.items():
            matches = chunk[text.str.contains(pattern, case=False, regex=True)]
            offsets = matches.groupby("patientunitstayid")[offset_column].min()
            for stay_id, offset in offsets.items():
                conditions[name][int(stay_id)] = min(
                    conditions[name].get(int(stay_id), float("inf")), float(offset)
                )
    return conditions, earliest_any


def _load_stable_background(raw_root: Path, stay_ids: set[int]) -> dict[str, Any]:
    patient = pd.read_csv(
        raw_root / "patient.csv.gz",
        compression="gzip",
        usecols=[
            "patientunitstayid",
            "gender",
            "age",
            "apacheadmissiondx",
            "admissionheight",
            "admissionweight",
        ],
        low_memory=False,
    )
    patient = patient[patient["patientunitstayid"].isin(stay_ids)].copy()
    patient.set_index("patientunitstayid", inplace=True)

    diagnosis, diagnosis_any = _earliest_offsets(
        raw_root / "diagnosis.csv.gz",
        stay_ids=stay_ids,
        text_column="diagnosisstring",
        offset_column="diagnosisoffset",
    )
    history, history_any = _earliest_offsets(
        raw_root / "pastHistory.csv.gz",
        stay_ids=stay_ids,
        text_column="pasthistorypath",
        offset_column="pasthistoryenteredoffset",
    )
    condition_offsets = {name: {} for name in BACKGROUND_CONDITIONS}
    for name, pattern in BACKGROUND_CONDITIONS.items():
        for source in (diagnosis[name], history[name]):
            for stay_id, offset in source.items():
                condition_offsets[name][stay_id] = min(
                    condition_offsets[name].get(stay_id, float("inf")), offset
                )
        admission = patient["apacheadmissiondx"].fillna("").astype(str)
        for stay_id in admission[admission.str.contains(pattern, case=False, regex=True)].index:
            condition_offsets[name][int(stay_id)] = 0.0
    earliest_any = dict(diagnosis_any)
    for stay_id, offset in history_any.items():
        earliest_any[stay_id] = min(earliest_any.get(stay_id, float("inf")), offset)
    return {
        "patient": patient,
        "condition_offsets": condition_offsets,
        "earliest_documentation": earliest_any,
    }


def _load_clinical_event_offsets(
    raw_root: Path,
    stay_ids: set[int],
) -> dict[str, dict[int, np.ndarray]]:
    """Load timestamp-valid pre-anchor assessment and care-intent events."""

    collected: dict[str, dict[int, list[float]]] = {
        category: {} for category in CLINICAL_EVENT_PATTERNS
    }
    for filename, offset_columns, text_columns in CLINICAL_EVENT_TABLES:
        path = raw_root / filename
        usecols = ["patientunitstayid", *offset_columns, *text_columns]
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=usecols,
            chunksize=250_000,
            low_memory=False,
        ):
            chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)].copy()
            if chunk.empty:
                continue
            offsets = chunk[list(offset_columns)].apply(
                pd.to_numeric, errors="coerce"
            ).max(axis=1, skipna=True)
            valid = np.isfinite(offsets)
            chunk = chunk.loc[valid]
            offsets = offsets.loc[valid]
            if chunk.empty:
                continue
            text = chunk[list(text_columns)].fillna("").astype(str).agg(" ".join, axis=1)
            for category, pattern in CLINICAL_EVENT_PATTERNS.items():
                matched = text.str.contains(pattern, case=False, regex=True)
                if not matched.any():
                    continue
                for stay_id, values in offsets[matched].groupby(
                    chunk.loc[matched, "patientunitstayid"]
                ):
                    collected[category].setdefault(int(stay_id), []).extend(
                        values.astype(float).tolist()
                    )
    return {
        category: {
            stay_id: np.sort(np.asarray(values, dtype=np.float64))
            for stay_id, values in by_stay.items()
        }
        for category, by_stay in collected.items()
    }


def _clinical_event_matrix(
    frame: pd.DataFrame,
    source: dict[str, dict[int, np.ndarray]],
) -> np.ndarray:
    """Build causal 1h/6h/24h event counts and time-since-last features."""

    onset_hour = pd.to_numeric(
        frame.get("onset_hour", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    since_onset = pd.to_numeric(
        frame.get("hours_since_onset", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    anchors = 60.0 * (onset_hour + since_onset)
    output = np.zeros((len(frame), len(CLINICAL_EVENT_PATTERNS) * 4), dtype=np.float32)
    for row, (stay_value, anchor) in enumerate(zip(frame["stay_id"], anchors)):
        stay_id = int(stay_value)
        for category_index, category in enumerate(CLINICAL_EVENT_PATTERNS):
            offsets = source[category].get(stay_id, np.empty(0, dtype=np.float64))
            stop = int(np.searchsorted(offsets, float(anchor), side="right"))
            available = offsets[:stop]
            base = category_index * 4
            for width_index, minutes in enumerate((60.0, 360.0, 1440.0)):
                start = int(
                    np.searchsorted(
                        available, float(anchor) - minutes, side="left"
                    )
                )
                output[row, base + width_index] = np.log1p(len(available) - start)
            output[row, base + 3] = (
                min(1.0, max(0.0, (float(anchor) - available[-1]) / 1440.0))
                if len(available)
                else 1.0
            )
    return output


def _stable_background_matrix(
    frame: pd.DataFrame,
    source: dict[str, Any],
) -> np.ndarray:
    patient = source["patient"]
    condition_offsets = source["condition_offsets"]
    earliest_any = source["earliest_documentation"]
    output = np.zeros((len(frame), 12), dtype=np.float32)
    onset_hour = frame.get("onset_hour", pd.Series(0.0, index=frame.index))
    since_onset = frame.get(
        "hours_since_onset", pd.Series(0.0, index=frame.index)
    )
    anchor_minutes = 60.0 * (
        pd.to_numeric(onset_hour, errors="coerce").fillna(0.0)
        + pd.to_numeric(since_onset, errors="coerce").fillna(0.0)
    )
    for row, (stay_value, anchor_minute) in enumerate(
        zip(frame["stay_id"], anchor_minutes)
    ):
        stay_id = int(stay_value)
        if stay_id not in patient.index:
            continue
        record = patient.loc[stay_id]
        age = _numeric_age(record["age"])
        gender = str(record["gender"]).strip().lower()
        height = pd.to_numeric(pd.Series([record["admissionheight"]]), errors="coerce").iloc[0]
        weight = pd.to_numeric(pd.Series([record["admissionweight"]]), errors="coerce").iloc[0]
        bmi = (
            float(weight) / np.square(float(height) / 100.0)
            if pd.notna(height) and pd.notna(weight) and float(height) > 0.0
            else np.nan
        )
        output[row, 0] = 0.0 if not np.isfinite(age) else age
        output[row, 1] = float(np.isfinite(age))
        output[row, 2] = float(gender == "male")
        output[row, 3] = float(gender == "female")
        output[row, 4] = float(gender in {"male", "female"})
        output[row, 5] = 0.0 if not np.isfinite(bmi) else bmi
        output[row, 6] = float(np.isfinite(bmi))
        for index, name in enumerate(BACKGROUND_CONDITIONS, start=7):
            offset = condition_offsets[name].get(stay_id)
            output[row, index] = float(
                offset is not None and float(offset) <= float(anchor_minute)
            )
        documented_at = earliest_any.get(stay_id)
        output[row, 11] = float(
            documented_at is not None and documented_at <= float(anchor_minute)
        )
    return output


def _phenotype_covariates(
    payload: dict[str, Any],
    *,
    assets: Any,
    background_source: dict[str, Any],
    window_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame = payload["frame"]
    variables = tuple(assets.checkpoint["variables"])
    raw = np.column_stack(
        [
            pd.to_numeric(frame.get(f"{variable}_t"), errors="coerce").to_numpy()
            if f"{variable}_t" in frame
            else np.full(len(frame), np.nan)
            for variable in variables
        ]
    ).astype(np.float64)
    observed = np.isfinite(raw)
    normalized = np.divide(
        raw - assets.scaler.medians,
        assets.scaler.scales,
        out=np.zeros_like(raw),
        where=observed,
    )
    ages = np.column_stack(
        [
            pd.to_numeric(frame.get(f"{variable}_age_hr"), errors="coerce").to_numpy()
            if f"{variable}_age_hr" in frame
            else np.full(len(frame), np.nan)
            for variable in variables
        ]
    ).astype(np.float64)
    history = payload.get("pre_anchor_history_covariates")
    if history is None:
        history = causal_history_covariates(
            normalized,
            observed_mask=observed,
            measurement_ages=ages,
            subject_ids=payload["trajectory_groups"],
            time_hours=payload["time_hours"],
            window_hours=window_hours,
        )
    stable = _stable_background_matrix(frame, background_source)
    return stable.astype(np.float32), history.astype(np.float32)


def _treatment_covariates(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> np.ndarray:
    """Return anchor-preceding treatment context; future ``act_*`` is excluded."""

    values = np.zeros((len(frame), len(columns)), dtype=np.float32)
    for index, column in enumerate(columns):
        if not column.startswith("hist_"):
            raise ValueError(f"non-history treatment column rejected: {column}")
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        if column.endswith("_evidence_count"):
            numeric = np.log1p(np.clip(numeric, 0.0, None))
        else:
            numeric = numeric.gt(0.0).astype(float)
        values[:, index] = numeric.to_numpy(dtype=np.float32)
    return values


def _contract_by_cell(assets: Any) -> dict[str, dict[str, Any]]:
    return {str(item["cell"]): item for item in assets.registry["cells"]}


def _cell_indices(
    payload: dict[str, Any],
    *,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if mode == "patient":
        groups = np.unique(payload["subjects"])
        rng.shuffle(groups)
        fit_groups = set(groups[: max(1, int(0.70 * len(groups)))])
        fit = np.asarray([value in fit_groups for value in payload["subjects"]])
    elif mode == "hospital":
        groups = np.unique(payload["hospitals"])
        rng.shuffle(groups)
        fit_groups = set(groups[: max(1, int(0.70 * len(groups)))])
        fit = np.asarray([value in fit_groups for value in payload["hospitals"]])
    elif mode == "forward_time":
        fit = np.zeros(len(payload["truth"]), dtype=bool)
        for subject in np.unique(payload["subjects"]):
            rows = np.flatnonzero(payload["subjects"] == subject)
            rows = rows[np.argsort(payload["time_hours"][rows], kind="stable")]
            boundary = max(1, int(0.70 * len(rows)))
            fit[rows[:boundary]] = True
    else:
        raise ValueError(f"unknown split mode: {mode}")
    test = ~fit
    return np.flatnonzero(fit), np.flatnonzero(test)


def _training_arrays(
    cells: tuple[str, ...],
    data: dict[str, dict[str, Any]],
    rows_by_cell: dict[str, np.ndarray],
    *,
    trajectory: bool,
    include_background: bool,
) -> dict[str, Any]:
    count = sum(len(rows_by_cell[cell]) for cell in cells)
    feature_parts = []
    subjects = []
    hospitals = []
    covariates = []
    base = np.zeros((count, len(cells)), dtype=np.float32)
    truth = np.zeros_like(base)
    mask = np.zeros_like(base, dtype=bool)
    offset = 0
    for cell_index, cell in enumerate(cells):
        payload = data[cell]
        rows = rows_by_cell[cell]
        source = payload["trajectory_features"] if trajectory else payload["features"]
        feature_parts.append(source[rows])
        subjects.append(payload["subjects"][rows])
        hospitals.append(payload["hospitals"][rows])
        if include_background:
            covariates.append(payload["phenotype_covariates"][rows])
        width = len(rows)
        base[offset : offset + width, cell_index] = payload["base"][rows]
        truth[offset : offset + width, cell_index] = payload["truth"][rows]
        mask[offset : offset + width, cell_index] = True
        offset += width
    return {
        "features": np.concatenate(feature_parts),
        "subjects": np.concatenate(subjects),
        "hospitals": np.concatenate(hospitals),
        "covariates": np.concatenate(covariates) if covariates else None,
        "base": base,
        "truth": truth,
        "mask": mask,
    }


def _fit_ensemble(
    training: dict[str, Any],
    *,
    cells: tuple[str, ...],
    member_seeds: tuple[int, ...],
    args: argparse.Namespace,
    placebo: bool,
) -> list[SoftPhysiologySubtypeModel]:
    output = []
    for seed in member_seeds:
        model = SoftPhysiologySubtypeModel(
            subtype_count=args.subtypes,
            min_effective_subjects=args.min_effective_subjects,
            subtype_prior_strength=args.subtype_prior_strength,
            max_rows_per_subject=args.subtype_rows_per_subject,
            embedding_dim=args.embedding_dim,
            random_state=seed,
            target_adapter=args.target_adapter,
            target_adapter_alpha=args.target_adapter_alpha,
        ).fit(
            training["features"],
            base_points=training["base"],
            observed_targets=training["truth"],
            observed_mask=training["mask"],
            subject_ids=training["subjects"],
            cell_keys=cells,
            balance_group_ids=training["hospitals"],
            phenotype_covariates=training["covariates"],
            responsibility_permutation_seed=(
                stable_seed(seed, "trajectory_placebo") if placebo else None
            ),
        )
        output.append(model)
    return output


def _predict_ensemble(
    models: list[SoftPhysiologySubtypeModel],
    *,
    cell: str,
    features: np.ndarray,
    base: np.ndarray,
    phenotype_covariates: np.ndarray | None = None,
    consensus_gate: bool = False,
) -> np.ndarray:
    corrections = np.stack(
        [
            model.subtype_corrections(
                features,
                cell=cell,
                phenotype_covariates=phenotype_covariates,
            )
            for model in models
        ]
    )
    correction = (
        _consensus_correction(corrections)
        if consensus_gate
        else corrections.mean(axis=0)
    )
    return base + correction


def _consensus_correction(corrections: np.ndarray) -> np.ndarray:
    """Keep only corrections with unanimous direction and low seed spread."""

    values = np.asarray(corrections, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("corrections must have shape [members, rows]")
    mean = values.mean(axis=0)
    spread = values.std(axis=0)
    unanimous = np.all(values > 0.0, axis=0) | np.all(values < 0.0, axis=0)
    reliable = unanimous & (np.abs(mean) >= spread)
    return np.where(reliable, mean, 0.0)


def _patient_equal_delta(
    truth: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    subjects: np.ndarray,
) -> float:
    values = []
    for subject in np.unique(subjects):
        rows = subjects == subject
        values.append(
            np.mean(np.abs(candidate[rows] - truth[rows]))
            - np.mean(np.abs(comparator[rows] - truth[rows]))
        )
    return float(np.mean(values))


def _group_patient_equal_mae(
    truth: np.ndarray,
    prediction: np.ndarray,
    subjects: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    output = {}
    for group in np.unique(groups):
        group_rows = groups == group
        patient_mae = []
        for subject in np.unique(subjects[group_rows]):
            rows = group_rows & (subjects == subject)
            patient_mae.append(float(np.mean(np.abs(prediction[rows] - truth[rows]))))
        if len(patient_mae) >= 5:
            output[str(group)] = float(np.mean(patient_mae))
    return output


def _select_group_robust_stack(
    *,
    truth: np.ndarray,
    base: np.ndarray,
    phenotype: np.ndarray,
    row_level: np.ndarray,
    subjects: np.ndarray,
    hospitals: np.ndarray,
) -> dict[str, float]:
    """Choose a convex residual stack by worst discovery-hospital delta."""

    baseline = _group_patient_equal_mae(
        truth, base, subjects, hospitals
    )
    candidates = []
    for phenotype_weight in np.linspace(0.0, 1.0, 5):
        correction = (
            phenotype_weight * (phenotype - base)
            + (1.0 - phenotype_weight) * (row_level - base)
        )
        for correction_scale in np.linspace(0.0, 1.0, 5):
            prediction = base + correction_scale * correction
            group_mae = _group_patient_equal_mae(
                truth, prediction, subjects, hospitals
            )
            common = sorted(set(baseline) & set(group_mae))
            if not common:
                continue
            deltas = np.asarray(
                [group_mae[group] - baseline[group] for group in common],
                dtype=np.float64,
            )
            candidates.append(
                (
                    float(np.max(deltas)),
                    float(np.mean(deltas)),
                    -float(correction_scale),
                    float(phenotype_weight),
                    float(correction_scale),
                )
            )
    if not candidates:
        return {"phenotype_weight": 0.0, "correction_scale": 0.0}
    selected = min(candidates)
    return {
        "phenotype_weight": selected[3],
        "correction_scale": selected[4],
        "worst_discovery_hospital_delta": selected[0],
        "mean_discovery_hospital_delta": selected[1],
    }


def _apply_stack(
    *,
    base: np.ndarray,
    phenotype: np.ndarray,
    row_level: np.ndarray,
    contract: dict[str, float],
) -> np.ndarray:
    weight = float(contract["phenotype_weight"])
    scale = float(contract["correction_scale"])
    correction = weight * (phenotype - base) + (1.0 - weight) * (
        row_level - base
    )
    return base + scale * correction


def _selective_gate_features(
    *,
    base: np.ndarray,
    candidate: np.ndarray,
    phenotype_covariates: np.ndarray,
) -> np.ndarray:
    """Build pre-anchor features for deciding whether a residual may move."""

    base = np.asarray(base, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    covariates = np.asarray(phenotype_covariates, dtype=np.float64)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    if len(base) != len(candidate) or len(base) != len(covariates):
        raise ValueError("selective-gate inputs must align")
    correction = candidate - base
    return np.column_stack((base, correction, np.abs(correction), covariates))


def _balanced_gate_weights(
    subjects: np.ndarray,
    hospitals: np.ndarray,
) -> np.ndarray:
    """Give each hospital and each patient within it equal total influence."""

    subjects = np.asarray(subjects, dtype=object)
    hospitals = np.asarray(hospitals, dtype=object)
    weights = np.zeros(len(subjects), dtype=np.float64)
    unique_hospitals = np.unique(hospitals)
    if not len(unique_hospitals):
        return np.ones(len(subjects), dtype=np.float64)
    for hospital in unique_hospitals:
        hospital_rows = hospitals == hospital
        patient_ids = np.unique(subjects[hospital_rows])
        if not len(patient_ids):
            continue
        for patient in patient_ids:
            rows = hospital_rows & (subjects == patient)
            weights[rows] = 1.0 / (len(unique_hospitals) * len(patient_ids) * rows.sum())
    total = weights.sum()
    return weights * (len(weights) / total) if total > 0.0 else np.ones(len(weights))


def _choose_gate_threshold(
    *,
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    probabilities: np.ndarray,
    subjects: np.ndarray,
    hospitals: np.ndarray,
) -> dict[str, float | bool]:
    """Select a threshold only when every discovery hospital improves."""

    base_mae = _group_patient_equal_mae(truth, base, subjects, hospitals)
    choices: list[tuple[float, float, float, float]] = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        move = probabilities >= threshold
        prediction = np.where(move, candidate, base)
        candidate_mae = _group_patient_equal_mae(
            truth, prediction, subjects, hospitals
        )
        common = sorted(set(base_mae) & set(candidate_mae))
        if not common:
            continue
        deltas = np.asarray(
            [candidate_mae[group] - base_mae[group] for group in common],
            dtype=np.float64,
        )
        choices.append(
            (
                float(np.max(deltas)),
                float(np.mean(deltas)),
                float(move.mean()),
                float(threshold),
            )
        )
    improving = [choice for choice in choices if choice[0] < 0.0 and choice[1] < 0.0]
    if not improving:
        return {
            "enabled": False,
            "threshold": 2.0,
            "worst_oof_hospital_delta": 0.0,
            "mean_oof_hospital_delta": 0.0,
            "oof_move_fraction": 0.0,
        }
    selected = min(improving, key=lambda item: (item[0], item[1], -item[2]))
    return {
        "enabled": True,
        "threshold": selected[3],
        "worst_oof_hospital_delta": selected[0],
        "mean_oof_hospital_delta": selected[1],
        "oof_move_fraction": selected[2],
    }


def _fit_crossfit_selective_gates(
    *,
    cells: tuple[str, ...],
    data: dict[str, dict[str, Any]],
    fit_rows: dict[str, np.ndarray],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Fit row-level residual gates from unseen discovery-hospital predictions."""

    hospitals = np.unique(
        np.concatenate(
            [data[cell]["hospitals"][fit_rows[cell]] for cell in cells]
        )
    )
    rng = np.random.default_rng(stable_seed(seed, "selective_gate_hospitals"))
    rng.shuffle(hospitals)
    hospital_folds = [fold for fold in np.array_split(hospitals, 3) if len(fold)]
    oof = {
        cell: np.full(len(fit_rows[cell]), np.nan, dtype=np.float64)
        for cell in cells
    }
    for fold_index, held_hospitals in enumerate(hospital_folds):
        inner_fit: dict[str, np.ndarray] = {}
        inner_test_positions: dict[str, np.ndarray] = {}
        usable = True
        for cell in cells:
            rows = fit_rows[cell]
            held = np.isin(data[cell]["hospitals"][rows], held_hospitals)
            inner_fit[cell] = rows[~held]
            inner_test_positions[cell] = np.flatnonzero(held)
            usable &= len(inner_fit[cell]) >= 100 and held.sum() >= 30
        if not usable:
            continue
        training = _training_arrays(
            cells,
            data,
            inner_fit,
            trajectory=True,
            include_background=True,
        )
        models = _fit_ensemble(
            training,
            cells=cells,
            member_seeds=(stable_seed(seed, "selective_gate", fold_index),),
            args=args,
            placebo=False,
        )
        for cell in cells:
            positions = inner_test_positions[cell]
            rows = fit_rows[cell][positions]
            payload = data[cell]
            oof[cell][positions] = _predict_ensemble(
                models,
                cell=cell,
                features=payload["trajectory_features"][rows],
                base=payload["base"][rows],
                phenotype_covariates=payload["phenotype_covariates"][rows],
            )

    output: dict[str, dict[str, Any]] = {}
    for cell in cells:
        payload = data[cell]
        rows = fit_rows[cell]
        valid = np.isfinite(oof[cell])
        base = payload["base"][rows][valid]
        truth = payload["truth"][rows][valid]
        candidate = oof[cell][valid]
        subjects = payload["subjects"][rows][valid]
        cell_hospitals = payload["hospitals"][rows][valid]
        covariates = payload["phenotype_covariates"][rows][valid]
        improvement = np.abs(candidate - truth) < np.abs(base - truth)
        if len(base) < 200 or len(np.unique(improvement)) < 2:
            output[cell] = {
                "enabled": False,
                "threshold": 2.0,
                "reason": "insufficient_crossfit_support",
                "oof_rows": int(len(base)),
            }
            continue
        features = _selective_gate_features(
            base=base,
            candidate=candidate,
            phenotype_covariates=covariates,
        )
        scaler = StandardScaler().fit(features)
        model = LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            random_state=stable_seed(seed, cell, "selective_gate_model"),
        )
        model.fit(
            scaler.transform(features),
            improvement.astype(int),
            sample_weight=_balanced_gate_weights(subjects, cell_hospitals),
        )
        probabilities = model.predict_proba(scaler.transform(features))[:, 1]
        contract = _choose_gate_threshold(
            truth=truth,
            base=base,
            candidate=candidate,
            probabilities=probabilities,
            subjects=subjects,
            hospitals=cell_hospitals,
        )
        contract.update(
            {
                "oof_rows": int(len(base)),
                "oof_hospitals": int(len(np.unique(cell_hospitals))),
                "scaler": scaler,
                "model": model,
            }
        )
        output[cell] = contract
    return output


def _apply_selective_gate(
    *,
    base: np.ndarray,
    candidate: np.ndarray,
    phenotype_covariates: np.ndarray,
    contract: dict[str, Any],
) -> tuple[np.ndarray, float]:
    if not contract.get("enabled", False):
        return np.asarray(base, dtype=np.float64), 0.0
    features = _selective_gate_features(
        base=base,
        candidate=candidate,
        phenotype_covariates=phenotype_covariates,
    )
    probabilities = contract["model"].predict_proba(
        contract["scaler"].transform(features)
    )[:, 1]
    move = probabilities >= float(contract["threshold"])
    return np.where(move, candidate, base), float(move.mean())


def _actual_future_delta(
    frame: pd.DataFrame,
    *,
    target: str,
    horizon_hours: float,
) -> np.ndarray:
    column = f"future_delta_t_hr_{target}"
    if column not in frame:
        return np.full(len(frame), float(horizon_hours), dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    return np.where(
        np.isfinite(values) & (values > 0.0), values, float(horizon_hours)
    )


def _causal_patient_residual_update(
    *,
    base: np.ndarray,
    candidate: np.ndarray,
    truth: np.ndarray,
    subjects: np.ndarray,
    anchor_times: np.ndarray,
    future_delta_hours: np.ndarray,
    prior_strength: float,
    history_base: np.ndarray | None = None,
    history_truth: np.ndarray | None = None,
    history_subjects: np.ndarray | None = None,
    history_anchor_times: np.ndarray | None = None,
    history_future_delta_hours: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink subtype corrections toward causally resolved patient errors."""

    base = np.asarray(base, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    subjects = np.asarray(subjects, dtype=object)
    anchor_times = np.asarray(anchor_times, dtype=np.float64)
    future_delta_hours = np.asarray(future_delta_hours, dtype=np.float64)
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive")
    prediction = candidate.copy()
    patient_weight = np.zeros(len(base), dtype=np.float64)
    resolved: dict[object, list[tuple[float, float]]] = {}
    if history_base is not None:
        for item in zip(
            np.asarray(history_subjects, dtype=object),
            np.asarray(history_anchor_times, dtype=np.float64),
            np.asarray(history_future_delta_hours, dtype=np.float64),
            np.asarray(history_truth, dtype=np.float64),
            np.asarray(history_base, dtype=np.float64),
        ):
            subject, anchor, delta, observed, frozen_base = item
            if np.isfinite(anchor + delta) and np.isfinite(observed - frozen_base):
                resolved.setdefault(subject, []).append(
                    (float(anchor + delta), float(observed - frozen_base))
                )
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        rows = rows[np.argsort(anchor_times[rows], kind="stable")]
        pending = resolved.setdefault(subject, [])
        for row in rows:
            available = [error for time, error in pending if time <= anchor_times[row]]
            count = len(available)
            if count:
                weight = count / (count + float(prior_strength))
                subtype_correction = candidate[row] - base[row]
                correction = (
                    (1.0 - weight) * subtype_correction
                    + weight * float(np.mean(available))
                )
                prediction[row] = base[row] + correction
                patient_weight[row] = weight
            resolution_time = anchor_times[row] + future_delta_hours[row]
            error = truth[row] - base[row]
            if np.isfinite(resolution_time) and np.isfinite(error):
                pending.append((float(resolution_time), float(error)))
    return prediction, patient_weight


def _run_split(
    *,
    cells: tuple[str, ...],
    data: dict[str, dict[str, Any]],
    mode: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    split = {
        cell: _cell_indices(data[cell], mode=mode, seed=stable_seed(seed, cell, mode))
        for cell in cells
    }
    fit_rows = {cell: split[cell][0] for cell in cells}
    test_rows = {cell: split[cell][1] for cell in cells}
    if any(len(fit_rows[cell]) < 100 or len(test_rows[cell]) < 50 for cell in cells):
        return {"mode": mode, "seed": seed, "status": "insufficient_split_support"}

    member_seeds = tuple(stable_seed(seed, "member", index) for index in range(args.members))
    background_training = _training_arrays(
        cells,
        data,
        fit_rows,
        trajectory=True,
        include_background=True,
    )
    trajectory_training = _training_arrays(
        cells,
        data,
        fit_rows,
        trajectory=True,
        include_background=False,
    )
    row_training = _training_arrays(
        cells,
        data,
        fit_rows,
        trajectory=False,
        include_background=False,
    )
    background_models = _fit_ensemble(
        background_training,
        cells=cells,
        member_seeds=member_seeds,
        args=args,
        placebo=False,
    )
    placebo_models = _fit_ensemble(
        background_training,
        cells=cells,
        member_seeds=member_seeds,
        args=args,
        placebo=True,
    )
    trajectory_models = _fit_ensemble(
        trajectory_training,
        cells=cells,
        member_seeds=member_seeds,
        args=args,
        placebo=False,
    )
    row_models = _fit_ensemble(
        row_training,
        cells=cells,
        member_seeds=member_seeds,
        args=args,
        placebo=False,
    )

    selective_gates = (
        _fit_crossfit_selective_gates(
            cells=cells,
            data=data,
            fit_rows=fit_rows,
            seed=seed,
            args=args,
        )
        if args.crossfit_selective_gate
        else {}
    )

    stack_contracts: dict[str, dict[str, dict[str, float]]] = {}
    for cell in cells:
        payload = data[cell]
        rows = fit_rows[cell]
        train_base = payload["base"][rows]
        train_covariates = payload["phenotype_covariates"][rows]
        train_background = _predict_ensemble(
            background_models,
            cell=cell,
            features=payload["trajectory_features"][rows],
            base=train_base,
            phenotype_covariates=train_covariates,
            consensus_gate=args.consensus_gate,
        )
        train_placebo = _predict_ensemble(
            placebo_models,
            cell=cell,
            features=payload["trajectory_features"][rows],
            base=train_base,
            phenotype_covariates=train_covariates,
            consensus_gate=args.consensus_gate,
        )
        train_row = _predict_ensemble(
            row_models,
            cell=cell,
            features=payload["features"][rows],
            base=train_base,
        )
        if args.group_robust_stack:
            candidate_contract = _select_group_robust_stack(
                truth=payload["truth"][rows],
                base=train_base,
                phenotype=train_background,
                row_level=train_row,
                subjects=payload["subjects"][rows],
                hospitals=payload["hospitals"][rows],
            )
            placebo_contract = _select_group_robust_stack(
                truth=payload["truth"][rows],
                base=train_base,
                phenotype=train_placebo,
                row_level=train_row,
                subjects=payload["subjects"][rows],
                hospitals=payload["hospitals"][rows],
            )
        else:
            candidate_contract = {
                "phenotype_weight": 1.0,
                "correction_scale": 1.0,
            }
            placebo_contract = dict(candidate_contract)
        stack_contracts[cell] = {
            "candidate": candidate_contract,
            "placebo": placebo_contract,
        }

    results = []
    for cell in cells:
        payload = data[cell]
        rows = test_rows[cell]
        base = payload["base"][rows]
        truth = payload["truth"][rows]
        subjects = payload["subjects"][rows]
        covariates = payload["phenotype_covariates"][rows]
        unstacked_background = _predict_ensemble(
            background_models,
            cell=cell,
            features=payload["trajectory_features"][rows],
            base=base,
            phenotype_covariates=covariates,
            consensus_gate=args.consensus_gate,
        )
        placebo = _predict_ensemble(
            placebo_models,
            cell=cell,
            features=payload["trajectory_features"][rows],
            base=base,
            phenotype_covariates=covariates,
            consensus_gate=args.consensus_gate,
        )
        trajectory = _predict_ensemble(
            trajectory_models,
            cell=cell,
            features=payload["trajectory_features"][rows],
            base=base,
        )
        row_level = _predict_ensemble(
            row_models,
            cell=cell,
            features=payload["features"][rows],
            base=base,
        )
        background = _apply_stack(
            base=base,
            phenotype=unstacked_background,
            row_level=row_level,
            contract=stack_contracts[cell]["candidate"],
        )
        placebo = _apply_stack(
            base=base,
            phenotype=placebo,
            row_level=row_level,
            contract=stack_contracts[cell]["placebo"],
        )
        patient_weights = np.zeros(len(rows), dtype=np.float64)
        if args.causal_patient_update:
            target, horizon_text = cell.split("@")
            horizon_hours = float(horizon_text.removesuffix("h"))
            fit_history_rows = fit_rows[cell]
            test_delta = _actual_future_delta(
                payload["frame"].iloc[rows],
                target=target,
                horizon_hours=horizon_hours,
            )
            history_delta = _actual_future_delta(
                payload["frame"].iloc[fit_history_rows],
                target=target,
                horizon_hours=horizon_hours,
            )
            update = {
                "base": base,
                "truth": truth,
                "subjects": subjects,
                "anchor_times": payload["time_hours"][rows],
                "future_delta_hours": test_delta,
                "prior_strength": float(args.patient_prior_strength),
                "history_base": payload["base"][fit_history_rows],
                "history_truth": payload["truth"][fit_history_rows],
                "history_subjects": payload["subjects"][fit_history_rows],
                "history_anchor_times": payload["time_hours"][fit_history_rows],
                "history_future_delta_hours": history_delta,
            }
            background, patient_weights = _causal_patient_residual_update(
                candidate=background,
                **update,
            )
            placebo, _ = _causal_patient_residual_update(
                candidate=placebo,
                **update,
            )
        move_fraction = 1.0
        placebo_move_fraction = 1.0
        if args.crossfit_selective_gate:
            background, move_fraction = _apply_selective_gate(
                base=base,
                candidate=background,
                phenotype_covariates=covariates,
                contract=selective_gates[cell],
            )
            placebo, placebo_move_fraction = _apply_selective_gate(
                base=base,
                candidate=placebo,
                phenotype_covariates=covariates,
                contract=selective_gates[cell],
            )
        deltas = {
            "vs_base": _patient_equal_delta(truth, background, base, subjects),
            "vs_placebo": _patient_equal_delta(
                truth, background, placebo, subjects
            ),
            "vs_trajectory": _patient_equal_delta(
                truth, background, trajectory, subjects
            ),
            "vs_row_level": _patient_equal_delta(
                truth, background, row_level, subjects
            ),
        }
        results.append(
            {
                "cell": cell,
                "rows": int(len(rows)),
                "patients": int(len(np.unique(subjects))),
                "base_mae": float(np.mean(np.abs(base - truth))),
                "background_mae": float(np.mean(np.abs(background - truth))),
                "unstacked_background_mae": float(
                    np.mean(np.abs(unstacked_background - truth))
                ),
                "placebo_mae": float(np.mean(np.abs(placebo - truth))),
                "trajectory_mae": float(np.mean(np.abs(trajectory - truth))),
                "row_level_mae": float(np.mean(np.abs(row_level - truth))),
                "patient_equal_deltas": deltas,
                "stack_contract": stack_contracts[cell],
                "selective_gate": {
                    key: value
                    for key, value in selective_gates.get(cell, {}).items()
                    if key not in {"model", "scaler"}
                },
                "selective_gate_move_fraction": move_fraction,
                "placebo_gate_move_fraction": placebo_move_fraction,
                "causal_patient_update_fraction": float(
                    np.mean(patient_weights > 0.0)
                ),
                "mean_patient_update_weight": float(patient_weights.mean()),
                "pass": bool(all(value < 0.0 for value in deltas.values())),
            }
        )
    return {"mode": mode, "seed": seed, "status": "complete", "cells": results}


def run(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_release(args.model_dir)
    contracts = _contract_by_cell(assets)
    requested_cells = {
        value.strip() for value in args.cells.split(",") if value.strip()
    }
    cells = tuple(
        cell
        for cell in TARGET_CONFIG
        if cell in contracts and (not requested_cells or cell in requested_cells)
    )
    if not cells:
        raise ValueError("no requested subtype cells exist in the runtime registry")
    data: dict[str, dict[str, Any]] = {}
    for index, cell in enumerate(cells, start=1):
        target, horizon_text = cell.split("@")
        horizon = int(horizon_text.removesuffix("h"))
        print(f"[{index}/{len(cells)}] extracting eICU {cell}", flush=True)
        payload = _load_cell(
            database="eICU",
            project_root=args.project_root,
            eicu_root=args.eicu_root,
            target=target,
            horizon=horizon,
            cell_contract=contracts[cell],
            assets=assets,
            max_rows=args.rows_per_cell,
            max_rows_per_subject=args.max_rows_per_subject,
            batch_size=args.batch_size,
            include_complete_history=True,
        )
        if payload is None:
            raise ValueError(f"insufficient eICU support for {cell}")
        trajectory_contract = TARGET_CONFIG[cell]
        payload["trajectory_features"] = causal_trajectory_features(
            payload["features"],
            subject_ids=payload["trajectory_groups"],
            time_hours=payload["time_hours"],
            window_hours=trajectory_contract["window_hours"],
            current_weight=trajectory_contract["current_weight"],
        )
        data[cell] = payload

    stay_ids = {
        int(stay_id)
        for payload in data.values()
        for stay_id in payload["frame"]["stay_id"].unique()
    }
    print("loading pre-anchor stable patient background", flush=True)
    background_source = _load_stable_background(args.raw_eicu_root, stay_ids)
    clinical_event_source = None
    if args.phenotype_mode == "clinical_event_state":
        print("loading timestamp-valid pre-anchor clinical events", flush=True)
        clinical_event_source = _load_clinical_event_offsets(
            args.raw_eicu_root, stay_ids
        )
    treatment_columns = tuple(
        sorted(
            {
                column
                for payload in data.values()
                for column in payload["frame"].columns
                if column.startswith("hist_")
            }
        )
    )
    for cell, payload in data.items():
        stable, history = _phenotype_covariates(
            payload,
            assets=assets,
            background_source=background_source,
            window_hours=TARGET_CONFIG[cell]["window_hours"],
        )
        treatment = _treatment_covariates(payload["frame"], treatment_columns)
        if args.phenotype_mode == "stable":
            payload["phenotype_covariates"] = stable
        elif args.phenotype_mode == "history":
            payload["phenotype_covariates"] = history
        elif args.phenotype_mode == "treatment":
            payload["phenotype_covariates"] = treatment
        elif args.phenotype_mode == "stable_treatment":
            payload["phenotype_covariates"] = np.concatenate(
                (stable, treatment), axis=1
            ).astype(np.float32)
        elif args.phenotype_mode == "event_state":
            payload["phenotype_covariates"] = np.concatenate(
                (stable, history, treatment), axis=1
            ).astype(np.float32)
        elif args.phenotype_mode == "clinical_event_state":
            if clinical_event_source is None:
                raise ValueError("clinical event source is missing")
            clinical = _clinical_event_matrix(
                payload["frame"], clinical_event_source
            )
            payload["phenotype_covariates"] = np.concatenate(
                (stable, history, treatment, clinical), axis=1
            ).astype(np.float32)
        else:
            payload["phenotype_covariates"] = np.concatenate(
                (stable, history), axis=1
            ).astype(np.float32)

    jobs = []
    for seed in (7, 19, 31, 43, 59, 71, 83):
        jobs.append(("patient", seed))
    for seed in (107, 131, 163):
        jobs.append(("hospital", seed))
    jobs.append(("forward_time", 211))

    split_results = []
    for index, (mode, seed) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] fitting {mode} split seed {seed}", flush=True)
        split_results.append(
            _run_split(
                cells=cells,
                data=data,
                mode=mode,
                seed=seed,
                args=args,
            )
        )

    summary = {}
    for cell in cells:
        by_mode = {}
        for mode in ("patient", "hospital", "forward_time"):
            matches = [
                item
                for split in split_results
                if split.get("status") == "complete" and split["mode"] == mode
                for item in split["cells"]
                if item["cell"] == cell
            ]
            by_mode[mode] = {
                "splits": len(matches),
                "pass_splits": sum(item["pass"] for item in matches),
                "median_delta_vs_base": float(
                    np.median(
                        [item["patient_equal_deltas"]["vs_base"] for item in matches]
                    )
                ),
                "median_delta_vs_placebo": float(
                    np.median(
                        [item["patient_equal_deltas"]["vs_placebo"] for item in matches]
                    )
                ),
                "median_delta_vs_trajectory": float(
                    np.median(
                        [
                            item["patient_equal_deltas"]["vs_trajectory"]
                            for item in matches
                        ]
                    )
                ),
                "median_delta_vs_row_level": float(
                    np.median(
                        [item["patient_equal_deltas"]["vs_row_level"] for item in matches]
                    )
                ),
            }
        development_pass = bool(
            by_mode["patient"]["pass_splits"] == 7
            and by_mode["hospital"]["pass_splits"] == 3
            and by_mode["forward_time"]["pass_splits"] == 1
        )
        summary[cell] = {
            "split_modes": by_mode,
            "development_pass": development_pass,
            "runtime_eligible": False,
        }

    result = {
        "schema": "whole_body_jepa.stable_background_subtype_eicu_audit.v2",
        "database": "eICU",
        "mimic_iv_read": False,
        "discovery_sources": {
            cell: {
                "module": payload["module"],
                "file": Path(payload["path"]).name,
                "rows": int(len(payload["truth"])),
                "patients": int(len(np.unique(payload["subjects"]))),
                "hospitals": int(len(np.unique(payload["hospitals"]))),
            }
            for cell, payload in data.items()
        },
        "trajectory_contract": TARGET_CONFIG,
        "phenotype_covariates": {
            "mode": args.phenotype_mode,
            "stable_background": [
                "age",
                "sex",
                "admission_bmi",
                "diabetes_documented_pre_anchor",
                "ckd_documented_pre_anchor",
                "heart_failure_documented_pre_anchor",
                "copd_documented_pre_anchor",
            ],
            "history_summary": [
                "level",
                "slope",
                "median_absolute_deviation",
                "observation_density",
                "measurement_age",
            ],
            "hospital_or_careunit_feature_used": False,
            "treatment_context_columns": list(treatment_columns),
            "future_treatment_used": False,
            "clinical_event_categories": (
                list(CLINICAL_EVENT_PATTERNS)
                if args.phenotype_mode == "clinical_event_state"
                else []
            ),
            "clinical_event_availability": (
                "max(event_offset, entry_offset) <= anchor"
                if args.phenotype_mode == "clinical_event_state"
                else None
            ),
        },
        "consensus_members": args.members,
        "target_adapter": args.target_adapter,
        "target_adapter_alpha": args.target_adapter_alpha,
        "group_robust_stack": bool(args.group_robust_stack),
        "consensus_gate": bool(args.consensus_gate),
        "crossfit_selective_gate": bool(args.crossfit_selective_gate),
        "causal_patient_update": bool(args.causal_patient_update),
        "patient_prior_strength": float(args.patient_prior_strength),
        "cells": summary,
        "split_results": split_results,
        "development_pass_cells": [
            cell for cell, item in summary.items() if item["development_pass"]
        ],
        "runtime_changed": False,
        "external_promotion_allowed": False,
        "causal_claim_allowed": False,
        "future_event_policy": FUTURE_EVENT_POLICY,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--eicu-root", type=Path, required=True)
    parser.add_argument("--raw-eicu-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cells",
        default="",
        help="optional comma-separated development cells; empty uses all candidates",
    )
    parser.add_argument(
        "--phenotype-mode",
        choices=(
            "stable",
            "history",
            "combined",
            "treatment",
            "stable_treatment",
            "event_state",
            "clinical_event_state",
        ),
        required=True,
    )
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--subtypes", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument(
        "--target-adapter",
        choices=("subtype_prior", "ridge", "direct_ridge"),
        required=True,
    )
    parser.add_argument("--target-adapter-alpha", type=float, default=100.0)
    parser.add_argument("--group-robust-stack", action="store_true")
    parser.add_argument("--consensus-gate", action="store_true")
    parser.add_argument("--crossfit-selective-gate", action="store_true")
    parser.add_argument("--causal-patient-update", action="store_true")
    parser.add_argument("--patient-prior-strength", type=float, default=4.0)
    parser.add_argument("--min-effective-subjects", type=float, default=20.0)
    parser.add_argument("--subtype-prior-strength", type=float, default=20.0)
    parser.add_argument("--subtype-rows-per-subject", type=int, default=4)
    parser.add_argument("--max-rows-per-subject", type=int, default=10)
    parser.add_argument("--rows-per-cell", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "development_pass_cells": result["development_pass_cells"],
                "runtime_changed": result["runtime_changed"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
