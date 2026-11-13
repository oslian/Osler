"""Canonical, causal event ledger for whole-body factual forecasting.

The ledger is intentionally upstream of every model. It normalizes event
identity and provenance, produces anchor-time views using only information
available at the anchor, and reconstructs target-specific future labels from
real post-anchor measurement events. Hospital and database provenance remain
auditable metadata and are never returned as physiological model features.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class EventKind(str, Enum):
    MEASUREMENT = "measurement"
    MEDICATION = "medication"
    INFUSION = "infusion"
    PROCEDURE = "procedure"


class EventQuality(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    DERIVED_INTERVAL_RATE = "derived_interval_rate"   # interval-normalized urine output (osler_jepa.urine_output)
    TEXT_EXTRACTED = "text_extracted"
    INVALID = "invalid"


@dataclass(frozen=True)
class TargetLabelPolicy:
    """Target-specific timing contract for a factual future value."""

    strict_tolerance_hours: float
    near_tolerance_hours: float
    minimum_delta_hours: float = 1.0 / 60.0
    allow_interpolation: bool = False
    interpolation_max_gap_hours: float = 0.0

    def __post_init__(self) -> None:
        if self.strict_tolerance_hours < 0:
            raise ValueError("strict_tolerance_hours must be non-negative")
        if self.near_tolerance_hours < self.strict_tolerance_hours:
            raise ValueError("near_tolerance_hours must cover the strict window")
        if self.minimum_delta_hours <= 0:
            raise ValueError("minimum_delta_hours must be positive")


DEFAULT_TARGET_LABEL_POLICIES: dict[str, TargetLabelPolicy] = {
    # Dense bedside variables.
    "heart_rate": TargetLabelPolicy(0.25, 0.75, allow_interpolation=True, interpolation_max_gap_hours=0.25),
    "map": TargetLabelPolicy(0.25, 0.75, allow_interpolation=True, interpolation_max_gap_hours=0.25),
    "o2sat": TargetLabelPolicy(0.25, 0.75, allow_interpolation=True, interpolation_max_gap_hours=0.25),
    "respiratory_rate": TargetLabelPolicy(0.25, 0.75, allow_interpolation=True, interpolation_max_gap_hours=0.25),
    "temperature": TargetLabelPolicy(0.50, 1.50),
    # Common laboratories.
    "glucose": TargetLabelPolicy(1.0, 3.0),
    "potassium": TargetLabelPolicy(2.0, 6.0),
    "sodium": TargetLabelPolicy(2.0, 6.0),
    "bicarbonate": TargetLabelPolicy(2.0, 6.0),
    "anion_gap": TargetLabelPolicy(2.0, 6.0),
    "creatinine": TargetLabelPolicy(4.0, 12.0),
    "bun": TargetLabelPolicy(4.0, 12.0),
    # Sparse targets are event-time tasks; wide "near" support is diagnostic
    # only and never receives strict JEPA supervision.
    "inr": TargetLabelPolicy(3.0, 12.0),
    "ptt": TargetLabelPolicy(3.0, 12.0),
    "fibrinogen": TargetLabelPolicy(6.0, 18.0),
    "troponin_t": TargetLabelPolicy(6.0, 18.0),
    "cortisol": TargetLabelPolicy(6.0, 18.0),
    "tsh": TargetLabelPolicy(12.0, 36.0),
}

DEFAULT_MEASUREMENT_AGE_HOURS: dict[str, float] = {
    # Bedside signals are stale quickly.
    "heart_rate": 2.0,
    "map": 2.0,
    "o2sat": 2.0,
    "respiratory_rate": 2.0,
    "temperature": 4.0,
    # Fast metabolic and gas variables.
    "glucose": 6.0,
    "ph": 6.0,
    "paco2": 6.0,
    "pao2": 6.0,
    "bicarbonate": 12.0,
    "anion_gap": 12.0,
    "lactate": 12.0,
    "potassium": 12.0,
    "sodium": 12.0,
    "chloride": 12.0,
    "calcium": 24.0,
    "ionized_calcium": 12.0,
    "magnesium": 24.0,
    "phosphate": 24.0,
    # Slower common labs.
    "creatinine": 24.0,
    "bun": 24.0,
    "wbc": 24.0,
    "hemoglobin": 24.0,
    "hematocrit": 24.0,
    "platelets": 24.0,
    "albumin": 48.0,
    "bilirubin": 48.0,
    # Sparse event targets retain an older observation, but age is explicit.
    "inr": 48.0,
    "ptt": 48.0,
    "fibrinogen": 72.0,
    "troponin_i": 48.0,
    "troponin_t": 48.0,
    "cortisol": 72.0,
    "tsh": 168.0,
}


VARIABLE_ALIASES = {
    "spo2": "o2sat",
    "oxygen_saturation": "o2sat",
    "mean_arterial_pressure": "map",
    "hco3": "bicarbonate",
    "hco3_": "bicarbonate",
    "blood_urea_nitrogen": "bun",
    "creat": "creatinine",
    "troponin-t": "troponin_t",
}


UNIT_CONVERSIONS: dict[tuple[str, str], tuple[str, float, float]] = {
    ("temperature", "f"): ("c", 5.0 / 9.0, -32.0 * 5.0 / 9.0),
    ("glucose", "mmol/l"): ("mg/dl", 18.0182, 0.0),
    ("creatinine", "umol/l"): ("mg/dl", 1.0 / 88.4, 0.0),
    ("bilirubin_total", "umol/l"): ("mg/dl", 1.0 / 17.104, 0.0),
}


def canonical_variable(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    return VARIABLE_ALIASES.get(key, key)


def canonical_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("event_time must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


@dataclass(frozen=True)
class CanonicalEvent:
    patient_id: str
    encounter_id: str
    event_time: pd.Timestamp
    variable: str
    kind: EventKind
    value: float | str | None = None
    unit: str | None = None
    database: str = "unknown"
    source_table: str = "unknown"
    source_id: str | None = None
    hospital_id: str | None = None
    care_unit: str | None = None
    quality: EventQuality = EventQuality.OBSERVED
    route: str | None = None
    rate: float | None = None
    rate_unit: str | None = None
    dose: float | None = None
    dose_unit: str | None = None
    end_time: pd.Timestamp | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patient_id", str(self.patient_id))
        object.__setattr__(self, "encounter_id", str(self.encounter_id))
        object.__setattr__(self, "event_time", canonical_timestamp(self.event_time))
        object.__setattr__(self, "variable", canonical_variable(self.variable))
        object.__setattr__(self, "kind", EventKind(self.kind))
        object.__setattr__(self, "quality", EventQuality(self.quality))
        if self.end_time is not None:
            end_time = canonical_timestamp(self.end_time)
            if end_time < self.event_time:
                raise ValueError("end_time cannot precede event_time")
            object.__setattr__(self, "end_time", end_time)

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "CanonicalEvent":
        allowed = set(cls.__dataclass_fields__)
        payload = {key: value for key, value in record.items() if key in allowed}
        return cls(**payload)

    def normalized(self) -> "CanonicalEvent":
        if not isinstance(self.value, (int, float, np.number)) or self.unit is None:
            return self
        unit = str(self.unit).strip().lower()
        conversion = UNIT_CONVERSIONS.get((self.variable, unit))
        if conversion is None:
            return self
        target_unit, scale, offset = conversion
        payload = asdict(self)
        payload["value"] = float(self.value) * scale + offset
        payload["unit"] = target_unit
        return CanonicalEvent(**payload)


class EventLedger:
    """Sorted per-encounter event store with fail-closed causal views."""

    def __init__(self, events: Iterable[CanonicalEvent | Mapping[str, Any]] = ()) -> None:
        normalized = [
            (event if isinstance(event, CanonicalEvent) else CanonicalEvent.from_mapping(event)).normalized()
            for event in events
        ]
        self._events = tuple(
            sorted(
                normalized,
                key=lambda event: (
                    event.encounter_id,
                    event.event_time,
                    event.kind.value,
                    event.variable,
                    event.source_id or "",
                ),
            )
        )
        by_encounter: dict[str, list[CanonicalEvent]] = defaultdict(list)
        measurements: dict[tuple[str, str], list[CanonicalEvent]] = defaultdict(list)
        treatments: dict[str, list[CanonicalEvent]] = defaultdict(list)
        for event in self._events:
            by_encounter[event.encounter_id].append(event)
            if event.kind == EventKind.MEASUREMENT:
                measurements[(event.encounter_id, event.variable)].append(event)
            elif event.kind in {
                EventKind.MEDICATION,
                EventKind.INFUSION,
                EventKind.PROCEDURE,
            }:
                treatments[event.encounter_id].append(event)
        self._by_encounter = {
            key: tuple(value) for key, value in by_encounter.items()
        }
        self._measurements = {
            key: tuple(value) for key, value in measurements.items()
        }
        self._measurement_times = {
            key: tuple(event.event_time for event in value)
            for key, value in self._measurements.items()
        }
        self._treatments = {
            key: tuple(value) for key, value in treatments.items()
        }
        self._treatment_times = {
            key: tuple(event.event_time for event in value)
            for key, value in self._treatments.items()
        }

    @property
    def events(self) -> tuple[CanonicalEvent, ...]:
        return self._events

    def encounter_events(self, encounter_id: str) -> tuple[CanonicalEvent, ...]:
        return self._by_encounter.get(str(encounter_id), ())

    def anchor_view(
        self,
        encounter_id: str,
        anchor_time: Any,
        variables: Sequence[str],
        *,
        max_age_hours: float | Mapping[str, float] = 72.0,
    ) -> dict[str, Any]:
        """Return latest pre-anchor measurements and observed treatment history."""

        anchor = canonical_timestamp(anchor_time)
        requested = tuple(canonical_variable(variable) for variable in variables)
        encounter_id = str(encounter_id)
        latest: dict[str, CanonicalEvent] = {}
        for variable in requested:
            key = (encounter_id, variable)
            events = self._measurements.get(key, ())
            times = self._measurement_times.get(key, ())
            index = bisect_right(times, anchor) - 1
            while index >= 0 and events[index].quality == EventQuality.INVALID:
                index -= 1
            if index >= 0:
                latest[variable] = events[index]
        treatment_events = self._treatments.get(encounter_id, ())
        treatment_times = self._treatment_times.get(encounter_id, ())
        treatments = treatment_events[:bisect_right(treatment_times, anchor)]

        values: dict[str, float | None] = {}
        observed: dict[str, bool] = {}
        ages: dict[str, float | None] = {}
        provenance: dict[str, Any] = {}
        for variable in requested:
            event = latest.get(variable)
            age_limit = (
                float(max_age_hours.get(variable, 72.0))
                if isinstance(max_age_hours, Mapping)
                else float(max_age_hours)
            )
            age = (
                (anchor - event.event_time).total_seconds() / 3600.0
                if event is not None
                else None
            )
            valid = bool(
                event is not None
                and age is not None
                and 0.0 <= age <= age_limit
                and isinstance(event.value, (int, float, np.number))
                and np.isfinite(float(event.value))
            )
            values[variable] = float(event.value) if valid else None
            observed[variable] = valid
            ages[variable] = float(age) if valid else None
            provenance[variable] = (
                {
                    "database": event.database,
                    "source_table": event.source_table,
                    "source_id": event.source_id,
                    "event_time": event.event_time.isoformat(),
                    "unit": event.unit,
                    "quality": event.quality.value,
                }
                if valid
                else None
            )

        treatment_context = [
            {
                "variable": event.variable,
                "kind": event.kind.value,
                "event_time": event.event_time.isoformat(),
                "end_time": event.end_time.isoformat() if event.end_time is not None else None,
                "route": event.route,
                "dose": event.dose,
                "dose_unit": event.dose_unit,
                "rate": event.rate,
                "rate_unit": event.rate_unit,
                "source_table": event.source_table,
            }
            for event in treatments
        ]
        return {
            "encounter_id": str(encounter_id),
            "anchor_time": anchor.isoformat(),
            "values": values,
            "observed": observed,
            "measurement_age_hours": ages,
            "provenance": provenance,
            "observed_treatment_context": treatment_context,
            "model_features_exclude": ["database", "hospital_id", "care_unit", "source_table"],
        }

    def future_label(
        self,
        encounter_id: str,
        variable: str,
        anchor_time: Any,
        nominal_horizon_hours: float,
        *,
        policy: TargetLabelPolicy | None = None,
        require_observed_anchor: bool = True,
    ) -> dict[str, Any]:
        """Reconstruct one target-specific label from real event timestamps."""

        variable = canonical_variable(variable)
        anchor = canonical_timestamp(anchor_time)
        horizon = float(nominal_horizon_hours)
        if horizon <= 0:
            raise ValueError("nominal_horizon_hours must be positive")
        policy = policy or DEFAULT_TARGET_LABEL_POLICIES.get(
            variable, TargetLabelPolicy(1.0, 4.0)
        )
        anchor_view = self.anchor_view(
            encounter_id,
            anchor,
            (variable,),
            max_age_hours=max(72.0, horizon),
        )
        anchor_observed = bool(anchor_view["observed"][variable])
        target_time = anchor + pd.Timedelta(hours=horizon)
        key = (str(encounter_id), variable)
        events = self._measurements.get(key, ())
        times = self._measurement_times.get(key, ())
        first = bisect_right(times, anchor)
        last = bisect_right(
            times,
            target_time + pd.Timedelta(hours=policy.near_tolerance_hours),
        )
        candidates = []
        for event in events[first:last]:
            if event.quality == EventQuality.INVALID:
                continue
            if not isinstance(event.value, (int, float, np.number)):
                continue
            delta = (event.event_time - anchor).total_seconds() / 3600.0
            if delta < policy.minimum_delta_hours:
                continue
            distance = abs((event.event_time - target_time).total_seconds() / 3600.0)
            candidates.append((distance, delta, event))
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen = candidates[0] if candidates else None
        if chosen is None:
            quality = "missing"
        elif chosen[0] <= policy.strict_tolerance_hours:
            quality = "strict"
        elif chosen[0] <= policy.near_tolerance_hours:
            quality = "near"
        else:
            quality = "missing"
        eligible = bool(
            quality == "strict" and (anchor_observed or not require_observed_anchor)
        )
        event = chosen[2] if chosen is not None and quality != "missing" else None
        actual_delta = chosen[1] if event is not None else None
        return {
            "target": variable,
            "anchor_time": anchor.isoformat(),
            "nominal_horizon_hours": horizon,
            "actual_delta_t_hours": float(actual_delta) if actual_delta is not None else None,
            "value": float(event.value) if event is not None else None,
            "unit": event.unit if event is not None else None,
            "event_time": event.event_time.isoformat() if event is not None else None,
            "quality": quality,
            "training_weight": 1.0 if eligible else 0.0,
            "anchor_observed": anchor_observed,
            "eligible_for_physiology_supervision": eligible,
            "source_table": event.source_table if event is not None else None,
        }

    def next_measurement_time(
        self, encounter_id: str, variable: str, anchor_time: Any
    ) -> float | None:
        variable = canonical_variable(variable)
        anchor = canonical_timestamp(anchor_time)
        key = (str(encounter_id), variable)
        events = self._measurements.get(key, ())
        times = self._measurement_times.get(key, ())
        index = bisect_right(times, anchor)
        while index < len(events):
            event = events[index]
            if event.quality != EventQuality.INVALID:
                return float(
                    (event.event_time - anchor).total_seconds() / 3600.0
                )
            index += 1
        return None

    def to_frame(self) -> pd.DataFrame:
        records = []
        for event in self._events:
            payload = asdict(event)
            payload["kind"] = event.kind.value
            payload["quality"] = event.quality.value
            payload["event_time"] = event.event_time
            payload["end_time"] = event.end_time
            records.append(payload)
        return pd.DataFrame.from_records(records)


__all__ = [
    "CanonicalEvent",
    "DEFAULT_MEASUREMENT_AGE_HOURS",
    "DEFAULT_TARGET_LABEL_POLICIES",
    "EventKind",
    "EventLedger",
    "EventQuality",
    "TargetLabelPolicy",
    "canonical_timestamp",
    "canonical_variable",
]
