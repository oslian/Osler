"""CSV preparation and validated BUN@3h patient-state inference."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aki_renal_belief import renal_belief_state_v2_features
from osler_jepa.patient_state import (
    LoadedPatientStateArtifact,
    load_patient_state_artifact,
    predict_patient_state_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = ROOT / "patient_state_precision_bun3_v1"
TARGET = "bun"
HORIZON_HOURS = 3
MINIMUM_ROWS = 6
MAXIMUM_ROWS = 500

MEASUREMENTS = (
    "creatinine",
    "bun",
    "urine_output",
    "potassium",
    "bicarbonate",
    "sodium",
    "map",
)
TREATMENTS = (
    "hist_fluids",
    "hist_vasopressor",
    "hist_diuretics",
    "hist_renal_replacement",
    "hist_nephrotoxin",
)

ALIASES = {
    "time": "hours_since_onset",
    "hour": "hours_since_onset",
    "hours": "hours_since_onset",
    "hours_since_start": "hours_since_onset",
    "cr": "creatinine",
    "scr": "creatinine",
    "blood_urea_nitrogen": "bun",
    "urine": "urine_output",
    "urine_ml": "urine_output",
    "k": "potassium",
    "hco3": "bicarbonate",
    "bicarb": "bicarbonate",
    "na": "sodium",
    "mean_arterial_pressure": "map",
    "fluids": "hist_fluids",
    "vasopressor": "hist_vasopressor",
    "diuretics": "hist_diuretics",
    "rrt": "hist_renal_replacement",
    "dialysis": "hist_renal_replacement",
    "nephrotoxin": "hist_nephrotoxin",
}


@lru_cache(maxsize=1)
def artifact() -> LoadedPatientStateArtifact:
    loaded = load_patient_state_artifact(ARTIFACT_DIRECTORY)
    if loaded.metadata.get("target") != TARGET:
        raise ValueError("serialized artifact target is not BUN")
    if int(loaded.metadata.get("horizon_hours", -1)) != HORIZON_HOURS:
        raise ValueError("serialized artifact horizon is not 3 hours")
    if loaded.metadata.get("causal_claim_allowed") is not False:
        raise ValueError("serialized artifact lacks factual-only boundary")
    return loaded


def capabilities() -> dict[str, Any]:
    loaded = artifact()
    return {
        "target": TARGET,
        "horizon_hours": HORIZON_HOURS,
        "artifact": ARTIFACT_DIRECTORY.name,
        "history_rows_required": int(loaded.metadata["history_steps"]),
        "required_columns": ["hours_since_onset", "bun"],
        "optional_columns": [*MEASUREMENTS, *TREATMENTS],
        "units": {
            "bun": "mg/dL",
            "creatinine": "mg/dL",
            "urine_output": "mL per source observation window",
            "potassium": "mEq/L",
            "bicarbonate": "mEq/L",
            "sodium": "mEq/L",
            "map": "mmHg",
        },
        "reference_note": (
            "BUN@6h motivated this demo, but exact 3-hour forecasting uses the "
            "separately validated BUN@3h artifact."
        ),
        "clinical_claim_allowed": False,
        "causal_claim_allowed": False,
        "patient_data_persisted": False,
    }


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    if not content:
        raise ValueError("CSV file is empty")
    if len(content) > 1_000_000:
        raise ValueError("CSV file must be smaller than 1 MB")
    try:
        raw = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("CSV must be valid UTF-8 comma-separated data") from exc
    return prepare_trajectory(raw)


def prepare_trajectory(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("CSV contains no data rows")
    if len(raw) < MINIMUM_ROWS:
        raise ValueError(f"at least {MINIMUM_ROWS} time-ordered rows are required")
    if len(raw) > MAXIMUM_ROWS:
        raise ValueError(f"CSV may contain at most {MAXIMUM_ROWS} rows")

    renamed = {}
    for original in raw.columns:
        normalized = str(original).strip().lower().replace(" ", "_")
        renamed[original] = ALIASES.get(normalized, normalized)
    frame = raw.rename(columns=renamed)
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError(f"duplicate columns after alias normalization: {duplicates}")
    if "hours_since_onset" not in frame:
        raise ValueError("CSV must include hours_since_onset")
    if "bun" not in frame:
        raise ValueError("CSV must include BUN in mg/dL")

    frame["hours_since_onset"] = pd.to_numeric(
        frame["hours_since_onset"], errors="coerce"
    )
    if frame["hours_since_onset"].isna().any():
        raise ValueError("hours_since_onset must contain only numeric values")
    frame = frame.sort_values("hours_since_onset", kind="stable").reset_index(drop=True)
    if frame["hours_since_onset"].duplicated().any():
        raise ValueError("hours_since_onset values must be unique")
    if (frame["hours_since_onset"].diff().dropna() <= 0).any():
        raise ValueError("hours_since_onset must increase strictly")

    output = pd.DataFrame({
        "stay_id": "uploaded-patient",
        "hours_since_onset": frame["hours_since_onset"].astype(float),
    })
    hours = output["hours_since_onset"].to_numpy(dtype=np.float64)
    for variable in MEASUREMENTS:
        observed = (
            pd.to_numeric(frame[variable], errors="coerce")
            if variable in frame
            else pd.Series(np.nan, index=frame.index, dtype=float)
        )
        values = np.full(len(frame), np.nan, dtype=np.float64)
        ages = np.full(len(frame), np.nan, dtype=np.float64)
        latest_value = np.nan
        latest_hour = np.nan
        for index, value in enumerate(observed.to_numpy(dtype=np.float64)):
            if np.isfinite(value):
                latest_value = float(value)
                latest_hour = float(hours[index])
            if np.isfinite(latest_value):
                values[index] = latest_value
                ages[index] = float(hours[index] - latest_hour)
        output[f"{variable}_t"] = values
        output[f"{variable}_age_hr"] = ages

    for treatment in TREATMENTS:
        output[treatment] = (
            pd.to_numeric(frame[treatment], errors="coerce").astype(float)
            if treatment in frame
            else np.nan
        )

    if not np.isfinite(output["bun_t"].iloc[-1]):
        raise ValueError("at least one BUN measurement is required")
    return output


def predict(frame: pd.DataFrame) -> dict[str, Any]:
    loaded = artifact()
    columns = list(loaded.metadata["feature_columns"])
    physical = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    steps = int(loaded.metadata["history_steps"])
    history = physical.tail(steps)
    if len(history) != steps:
        raise ValueError(f"exactly {steps} usable history rows are required")
    history_hours = frame.loc[history.index, "hours_since_onset"].to_numpy(
        dtype=np.float32
    )
    elapsed = history_hours - float(history_hours[-1])
    belief = renal_belief_state_v2_features(frame).iloc[-1].to_numpy(dtype=np.float32)
    current_bun = float(frame["bun_t"].iloc[-1])
    prediction = predict_patient_state_artifact(
        loaded,
        current_values=physical.iloc[-1].to_numpy(dtype=np.float32),
        history_values=history.to_numpy(dtype=np.float32),
        history_elapsed_hours=elapsed,
        history_mask=np.ones(steps, dtype=bool),
        belief_values=belief,
        current_target=current_bun,
        anchor_hours_since_onset=float(history_hours[-1]),
        coverage=0.90,
    )

    observation_counts = {
        variable: int((frame[f"{variable}_age_hr"] == 0.0).sum())
        for variable in MEASUREMENTS
    }
    latest_ages = {
        variable: _finite_or_none(frame[f"{variable}_age_hr"].iloc[-1])
        for variable in MEASUREMENTS
    }
    warnings: list[dict[str, Any]] = []
    missing_optional = [
        variable for variable, count in observation_counts.items() if count == 0
    ]
    if missing_optional:
        warnings.append({
            "code": "missing_optional_measurements",
            "variables": missing_optional,
        })
    missing_treatments = [
        treatment for treatment in TREATMENTS if frame[treatment].isna().all()
    ]
    if missing_treatments:
        warnings.append({
            "code": "missing_treatment_context",
            "variables": missing_treatments,
        })

    return {
        "target": TARGET,
        "horizon_hours": HORIZON_HOURS,
        "unit": "mg/dL",
        "current_bun": round(current_bun, 4),
        "current_bun_measurement_age_hours": _finite_or_none(
            frame["bun_age_hr"].iloc[-1]
        ),
        "population_point": round(prediction.population_point, 4),
        "personalized_point": round(prediction.personalized_point, 4),
        "personalized_delta_vs_population": round(
            prediction.personalized_point - prediction.population_point, 4
        ),
        "change_from_current": round(prediction.personalized_point - current_bun, 4),
        "lower": round(prediction.lower, 4),
        "upper": round(prediction.upper, 4),
        "interval_full_width": round(2.0 * prediction.half_width, 4),
        "interval_target_coverage": prediction.coverage,
        "source_model": ARTIFACT_DIRECTORY.name,
        "input_quality": {
            "rows_received": int(len(frame)),
            "history_rows_used": steps,
            "history_span_hours": round(float(history_hours[-1] - history_hours[0]), 4),
            "observation_counts": observation_counts,
            "latest_measurement_ages_hours": latest_ages,
            "warnings": warnings,
        },
        "clinical_claim_allowed": False,
        "causal_claim_allowed": False,
        "patient_data_persisted": False,
    }


def predict_csv(content: bytes) -> dict[str, Any]:
    return predict(read_csv_bytes(content))


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if np.isfinite(number) else None
