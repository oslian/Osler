"""Measured values the Medication Safety engine may read from a monitored patient.

Only observations, never forecasts: the latest delivered value of a variable the
engine's patient flags already use, when it is recent enough and in the unit the
engine's thresholds are written in. Everything else the engine needs (age,
kidney function, blood pressure, allergies) still comes from the case itself.

Blood pressure is left out on purpose: the engine reads a systolic pressure and
the stream's canonical variable is the mean pressure, and one is not derived
from the other here. Kidney function is left out too: the engine reads an eGFR,
and an eGFR from one creatinine is not valid while kidney function is changing.
"""

from __future__ import annotations

from typing import Any

from .timeline import LOOKBACK_HOURS, delivered_timeline, variable_state

# stream variable -> (group, field) in the case fields, and the unit the engine's threshold assumes
MEASURED_FIELDS: dict[str, tuple[str, str, str]] = {
    "heart_rate": ("vitals", "heart_rate", "beats/min"),
    "o2sat": ("vitals", "spo2", "%"),
    "potassium": ("labs", "potassium", "mEq/L"),
}
MAX_AGE_HOURS = LOOKBACK_HOURS


def measured_inputs(service, patient_id: str, session_id: str | None = None) -> dict[str, Any] | None:
    """What the engine may read for this patient now, with its provenance; None for an unknown patient."""
    session = service.session_for_patient(patient_id, session_id)
    if session is None:
        return None
    timeline = delivered_timeline(service.repo, session.session_id, session.epoch)
    as_of = max((row["effective_time"] for row in timeline), default=None)
    state = variable_state(timeline, as_of)
    used, not_used = [], []
    for variable, (group, field, unit) in MEASURED_FIELDS.items():
        item = state.get(variable)
        if item is None:
            not_used.append({"variable": variable, "reason": "not_measured"})
        elif item["unit"] != unit:
            not_used.append({"variable": variable, "reason": f"unit {item['unit']} is not {unit}"})
        elif item["age_hours"] > MAX_AGE_HOURS:
            not_used.append({"variable": variable, "reason": "stale", "age_hours": item["age_hours"],
                             "effective_time": item["effective_time"]})
        else:
            used.append({"group": group, "field": field, "variable": variable, "value": item["value"], "unit": unit,
                         "effective_time": item["effective_time"], "age_hours": item["age_hours"],
                         "reference": f"Observation/{item['resource_id']}/_history/{item['version_id']}"})
    return {"patient_id": patient_id, "session_id": session.session_id, "as_of": as_of,
            "max_age_hours": MAX_AGE_HOURS, "used": used, "not_used": not_used, "forecasts_used": False}
