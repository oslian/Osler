"""Raw MIMIC-IV/eICU adapters for the canonical factual event ledger.

These adapters intentionally produce a plain, auditable event table before
any anchor or future label is constructed. They reuse the project's validated
variable maps and plausibility cleaning, but do not reuse transition labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .event_ledger import (
    CanonicalEvent,
    EventKind,
    EventLedger,
    EventQuality,
    canonical_variable,
)


CANONICAL_EVENT_COLUMNS = (
    "patient_id",
    "encounter_id",
    "event_time",
    "end_time",
    "variable",
    "kind",
    "value",
    "unit",
    "database",
    "source_table",
    "source_id",
    "hospital_id",
    "care_unit",
    "quality",
    "route",
    "rate",
    "rate_unit",
    "dose",
    "dose_unit",
    "metadata",
)


CANONICAL_UNITS = {
    "heart_rate": "beats/min",
    "map": "mmHg",
    "respiratory_rate": "breaths/min",
    "o2sat": "%",
    "temperature": "degC",
    "glucose": "mg/dL",
    "ph": "pH",
    "paco2": "mmHg",
    "pao2": "mmHg",
    "bicarbonate": "mEq/L",
    "anion_gap": "mEq/L",
    "chloride": "mEq/L",
    "potassium": "mEq/L",
    "sodium": "mEq/L",
    "calcium": "mg/dL",
    "ionized_calcium": "mmol/L",
    "magnesium": "mg/dL",
    "phosphate": "mg/dL",
    "lactate": "mmol/L",
    "creatinine": "mg/dL",
    "bun": "mg/dL",
    "wbc": "K/uL",
    "hemoglobin": "g/dL",
    "hematocrit": "%",
    "bilirubin": "mg/dL",
    "bilirubin_direct": "mg/dL",
    "platelets": "K/uL",
    "inr": "ratio",
    "ptt": "sec",
    "fibrinogen": "mg/dL",
    "albumin": "g/dL",
    "total_protein": "g/dL",
    "ast": "IU/L",
    "alt": "IU/L",
    "alkaline_phos": "IU/L",
    "troponin_i": "ng/mL",
    "troponin_t": "ng/mL",
    "cortisol": "ug/dL",
    "tsh": "uIU/mL",
    "urine_output": "mL/hour",
}

EICU_RELATIVE_TIME_ORIGIN = pd.Timestamp("2100-01-01")


def eicu_relative_timestamp(offset_minutes: pd.Series) -> pd.Series:
    """Map eICU relative-minute offsets onto the shared synthetic epoch.

    eICU has no absolute chart date. Measurements and treatments must use the
    same synthetic origin or every treatment disappears from anchor history.
    """

    return EICU_RELATIVE_TIME_ORIGIN + pd.to_timedelta(
        pd.to_numeric(offset_minutes, errors="coerce"), unit="m"
    )


def _assert_eicu_event_time_alignment(parts: list[pd.DataFrame]) -> None:
    """Reject a ledger whose event kinds silently use different centuries."""

    nonempty = [part for part in parts if part is not None and not part.empty]
    if len(nonempty) < 2:
        return
    medians = []
    for part in nonempty:
        times = pd.to_datetime(part["event_time"], errors="coerce").dropna()
        if not times.empty:
            # Pandas may preserve datetime64[us] on newer versions. Convert
            # explicitly to nanoseconds before applying the fixed ns-to-year
            # scale below, otherwise a century can be misread as 0.1 years.
            nanoseconds = times.map(lambda value: pd.Timestamp(value).value)
            medians.append(nanoseconds.median())
    if len(medians) < 2:
        return
    span_years = (max(medians) - min(medians)) / (365.25 * 24 * 3600 * 1e9)
    if span_years > 1.0:
        raise ValueError(
            "eICU event kinds use inconsistent relative-time origins "
            f"(median span {span_years:.1f} years)"
        )


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_EVENT_COLUMNS)


def canonicalize_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema/dtypes and reject rows with invalid event identity."""

    if frame.empty:
        return _empty_event_frame()
    output = frame.copy()
    for column in CANONICAL_EVENT_COLUMNS:
        if column not in output:
            output[column] = None
    output = output[list(CANONICAL_EVENT_COLUMNS)]
    output["event_time"] = pd.to_datetime(output["event_time"], errors="coerce")
    output["end_time"] = pd.to_datetime(output["end_time"], errors="coerce")
    output["variable"] = output["variable"].map(canonical_variable)
    output["patient_id"] = output["patient_id"].astype(str)
    output["encounter_id"] = output["encounter_id"].astype(str)
    output["value"] = pd.to_numeric(output["value"], errors="coerce")
    for column in ("rate", "dose"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["quality"] = output["quality"].fillna(EventQuality.OBSERVED.value)
    output["metadata"] = output["metadata"].map(
        lambda value: (
            value
            if isinstance(value, str)
            else json.dumps(value or {}, sort_keys=True)
        )
    )
    valid_identity = (
        output["event_time"].notna()
        & output["patient_id"].ne("")
        & output["encounter_id"].ne("")
        & output["variable"].ne("")
    )
    output = output.loc[valid_identity].copy()
    return output.sort_values(
        ["encounter_id", "event_time", "kind", "variable", "source_id"],
        kind="stable",
    ).reset_index(drop=True)


def _measurement_events(
    measurements: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    database: str,
    source_table: str,
    time_column: str,
    time_basis: str,
) -> pd.DataFrame:
    if measurements.empty:
        return _empty_event_frame()
    lookup = meta.set_index("stay_id", drop=False)
    stay = pd.to_numeric(measurements["stay_id"], errors="coerce")
    usable = stay.notna() & stay.astype("Int64").isin(lookup.index)
    frame = measurements.loc[usable].copy()
    stay = frame["stay_id"].astype("int64")
    frame["patient_id"] = stay.map(lookup["subject_id"]).astype(str)
    frame["encounter_id"] = stay.astype(str)
    frame["event_time"] = pd.to_datetime(frame[time_column], errors="coerce")
    frame["end_time"] = pd.NaT
    frame["variable"] = frame["var"].map(canonical_variable)
    frame["kind"] = EventKind.MEASUREMENT.value
    frame["value"] = pd.to_numeric(frame["valuenum"], errors="coerce")
    frame["unit"] = frame["variable"].map(CANONICAL_UNITS)
    frame["database"] = database
    frame["source_table"] = source_table
    frame["source_id"] = [
        f"{source_table}:{stay_id}:{offset}"
        for stay_id, offset in zip(stay, frame.index)
    ]
    frame["hospital_id"] = stay.map(lookup["hospitalid"]).astype(str)
    frame["care_unit"] = stay.map(lookup["careunit"])
    frame["quality"] = frame.get(
        "quality", pd.Series(EventQuality.OBSERVED.value, index=frame.index)
    ).fillna(EventQuality.OBSERVED.value)
    frame["route"] = None
    frame["rate"] = np.nan
    frame["rate_unit"] = None
    frame["dose"] = np.nan
    frame["dose_unit"] = None
    offsets = frame.get("offset", pd.Series(np.nan, index=frame.index))
    intervals = frame.get(
        "collection_interval_hr", pd.Series(np.nan, index=frame.index)
    )
    frame["metadata"] = [
        {
            "time_basis": time_basis,
            "raw_offset": offset,
            **(
                {"collection_interval_hr": float(interval)}
                if np.isfinite(interval)
                else {}
            ),
        }
        for offset, interval in zip(offsets, intervals)
    ]
    return canonicalize_event_frame(frame)


def _treatment_events(
    evidence: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    database: str,
) -> pd.DataFrame:
    if evidence.empty:
        return _empty_event_frame()
    lookup = meta.set_index("stay_id", drop=False)
    frame = evidence.copy()
    frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce")
    frame = frame[
        frame["stay_id"].notna()
        & frame["stay_id"].astype("Int64").isin(lookup.index)
    ].copy()
    if frame.empty:
        return _empty_event_frame()
    stay = frame["stay_id"].astype("int64")
    source = frame.get(
        "source", pd.Series("unknown_treatment", index=frame.index)
    ).astype(str)
    frame["patient_id"] = stay.map(lookup["subject_id"]).astype(str)
    frame["encounter_id"] = stay.astype(str)
    frame["event_time"] = pd.to_datetime(frame["starttime"], errors="coerce")
    frame["end_time"] = pd.to_datetime(frame.get("endtime"), errors="coerce")
    frame["variable"] = frame["action"].map(canonical_variable)
    frame["kind"] = np.where(
        source.str.contains("procedure|treatment", case=False, regex=True),
        EventKind.PROCEDURE.value,
        np.where(
            source.str.contains("infusion|inputevent", case=False, regex=True),
            EventKind.INFUSION.value,
            EventKind.MEDICATION.value,
        ),
    )
    frame["value"] = np.nan
    frame["unit"] = None
    frame["database"] = database
    frame["source_table"] = source
    frame["source_id"] = [
        f"{source_value}:{stay_id}:{offset}"
        for source_value, stay_id, offset in zip(source, stay, frame.index)
    ]
    frame["hospital_id"] = stay.map(lookup["hospitalid"]).astype(str)
    frame["care_unit"] = stay.map(lookup["careunit"])
    frame["quality"] = np.where(
        frame.get(
            "dose_observed", pd.Series(False, index=frame.index)
        ).fillna(False),
        EventQuality.OBSERVED.value,
        EventQuality.DERIVED.value,
    )
    frame["route"] = frame.get("routeadmin")
    frame["rate"] = pd.to_numeric(
        frame.get(
            "rate_like",
            frame.get("drugrate", frame.get("infusionrate")),
        ),
        errors="coerce",
    )
    frame["rate_unit"] = frame.get("rate_unit")
    frame["dose"] = pd.to_numeric(
        frame.get(
            "amount_like",
            frame.get("dose_amount", frame.get("drugamount")),
        ),
        errors="coerce",
    )
    frame["dose_unit"] = frame.get(
        "dose_unit", frame.get("amount_unit")
    )
    frame["metadata"] = [
        {
            "original_label": label,
            "evidence_kind": evidence_kind,
            "future_treatment": False,
        }
        for label, evidence_kind in zip(
            frame.get("original_label", pd.Series("", index=frame.index)),
            frame.get("evidence_kind", pd.Series("", index=frame.index)),
        )
    ]
    return canonicalize_event_frame(frame)


def extract_mimiciv_events(
    data_root: Path,
    *,
    max_stays: int | None = 1000,
    include_treatments: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract a bounded MIMIC-IV event ledger and encounter provenance."""

    from mimiciv_observation_transition_extract import (
        LAB_ITEMIDS,
        VITAL_ITEMIDS,
        con,
        pull_measurements,
        pull_stay_meta,
        pull_treatment_evidence,
        register_stays,
    )

    connection = con()
    meta = pull_stay_meta(connection, Path(data_root), max_stays)
    meta = meta.rename(
        columns={
            "first_careunit": "careunit",
        }
    )
    meta["hospitalid"] = "mimiciv_single_center"
    register_stays(connection, meta)
    measurements = pull_measurements(connection, Path(data_root))
    vital_variables = set(VITAL_ITEMIDS) - {"temperature_f"}
    lab_variables = set(LAB_ITEMIDS)
    measurement_parts = []
    for source_table, keep in (
        ("chartevents", measurements["var"].isin(vital_variables)),
        ("outputevents", measurements["var"].eq("urine_output")),
        (
            "labevents",
            measurements["var"].isin(lab_variables)
            & ~measurements["var"].isin(vital_variables),
        ),
    ):
        measurement_parts.append(
            _measurement_events(
                measurements.loc[keep],
                meta,
                database="mimiciv",
                source_table=source_table,
                time_column="charttime",
                time_basis="absolute_charttime",
            )
        )
    parts = measurement_parts
    if include_treatments:
        treatment = pull_treatment_evidence(connection, Path(data_root))
        parts.append(_treatment_events(treatment, meta, database="mimiciv"))
    events = canonicalize_event_frame(
        pd.concat(parts, ignore_index=True, sort=False)
    )
    return events, meta.reset_index(drop=True)


def _harmonized_measurement_events(measurements: pd.DataFrame, stays: pd.DataFrame, *, database: str,
                                   hospitalid: str, max_stays: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical measurement events from an October-harmonized transition extractor's long measurements.

    ``stays`` needs subject_id, stay_id and first_careunit; stay ids are mapped to
    integers when the source uses strings (encounter ids stay unique).
    """

    stays = stays.drop_duplicates("stay_id").reset_index(drop=True)
    if max_stays is not None:
        stays = stays.iloc[: int(max_stays)]
    codes = {str(value): index + 1 for index, value in enumerate(stays["stay_id"].astype(str))}
    meta = pd.DataFrame({"subject_id": stays["subject_id"].astype(str), "stay_id": stays["stay_id"].astype(str).map(codes),
                         "careunit": stays["first_careunit"].astype(str), "hospitalid": hospitalid})
    measurements = measurements[measurements["stay_id"].astype(str).isin(codes)].copy()
    measurements["stay_id"] = measurements["stay_id"].astype(str).map(codes)
    events = _measurement_events(measurements, meta, database=database, source_table=f"{database}_harmonized_measurements",
                                 time_column="charttime", time_basis="absolute_charttime")
    return canonicalize_event_frame(events), meta


def extract_nwicu_events(data_root: Path, *, max_stays: int | None = 1000, include_treatments: bool = False):
    """NWICU measurement events with the October harmonized definitions (no treatment records exist)."""

    import mimiciv_observation_transition_extract as contract
    from nwicu_observation_transition_extract import load_labs, load_stays, load_vitals

    stays = load_stays(Path(data_root), 1.0)
    if max_stays is not None:
        stays = stays.iloc[: int(max_stays)]
    measurements = contract.clean_measurements(pd.concat([load_labs(Path(data_root), stays), load_vitals(Path(data_root), stays)],
                                                         ignore_index=True))
    return _harmonized_measurement_events(measurements, stays, database="nwicu", hospitalid="nwicu", max_stays=max_stays)


def extract_zigong_events(data_root: Path, *, max_stays: int | None = 1000, include_treatments: bool = False):
    """Zigong laboratory events (nursing chart excluded: day-level timestamps), harmonized definitions."""

    import mimiciv_observation_transition_extract as contract
    from zigong_observation_transition_extract import icu_stays, load_labs

    stays = icu_stays(Path(data_root), 1.0)
    if max_stays is not None:
        stays = stays.iloc[: int(max_stays)]
    measurements = contract.clean_measurements(load_labs(Path(data_root), stays))
    return _harmonized_measurement_events(measurements, stays, database="zigong", hospitalid="zigong", max_stays=max_stays)

def _merged_eicu_treatment_evidence(
    data_root: Path,
    stay_ids: set[int],
) -> pd.DataFrame:
    """Scan raw treatment tables once using the union action contract."""

    from eicu_body_system_configs import (
        BODY_SYSTEM_CONFIGS,
        BodySystemDiseaseConfig,
    )
    from eicu_body_system_transition_extract import (
        infusion_evidence,
        medication_evidence,
        respiratory_evidence,
        treatment_evidence,
        transfusion_intake_output_evidence,
    )

    action_terms: dict[str, set[str]] = {}
    for config in BODY_SYSTEM_CONFIGS.values():
        for action, terms in config.action_terms.items():
            action_terms.setdefault(action, set()).update(terms)
    union_config = BodySystemDiseaseConfig(
        name="whole_body_event_ledger",
        display_name="Whole-body union treatment contract",
        body_system="whole_body",
        diagnosis_patterns=(),
        targets=(),
        state_vars=(),
        action_terms={
            action: tuple(sorted(terms))
            for action, terms in sorted(action_terms.items())
        },
    )
    parts = []
    for extractor in (
        medication_evidence,
        infusion_evidence,
        treatment_evidence,
        respiratory_evidence,
        transfusion_intake_output_evidence,
    ):
        frame = extractor(Path(data_root), stay_ids, union_config)
        if not frame.empty:
            parts.append(frame)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True, sort=False)
    subset = [
        column
        for column in ("stay_id", "starttime", "endtime", "action", "source")
        if column in combined
    ]
    return combined.drop_duplicates(subset=subset).reset_index(drop=True)


def extract_eicu_events(
    data_root: Path,
    *,
    max_stays: int | None = 1000,
    include_treatments: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract a bounded eICU event ledger using real relative event times."""

    from eicu_sepsis_transition_extract import (
        read_labs,
        read_patient_meta,
        read_urine_output,
        read_vitals,
    )

    meta = read_patient_meta(Path(data_root)).reset_index(drop=True)
    meta = meta.sort_values("stay_id")
    if max_stays is not None:
        meta = meta.head(int(max_stays)).copy()
    meta["careunit"] = meta.get("unittype")
    stay_ids = set(meta["stay_id"].astype(int))
    measurement_parts = []
    for source_table, reader in (
        ("lab", read_labs),
        ("vital", read_vitals),
        ("intakeOutput", read_urine_output),
    ):
        measurements = reader(Path(data_root), stay_ids)
        if measurements.empty:
            continue
        measurements = measurements.copy()
        measurements["event_time"] = eicu_relative_timestamp(
            measurements["offset"]
        )
        measurement_parts.append(
            _measurement_events(
                measurements,
                meta,
                database="eicu",
                source_table=source_table,
                time_column="event_time",
                time_basis="unit_admission_relative_minutes",
            )
        )
    parts = measurement_parts
    if include_treatments:
        evidence = _merged_eicu_treatment_evidence(Path(data_root), stay_ids)
        parts.append(_treatment_events(evidence, meta, database="eicu"))
    _assert_eicu_event_time_alignment(parts)
    events = canonicalize_event_frame(
        pd.concat(parts, ignore_index=True, sort=False)
        if parts
        else _empty_event_frame()
    )
    return events, meta.reset_index(drop=True)


def write_event_dataset(
    events: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write one parquet per database/kind for bounded-memory downstream use."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for (database, kind), group in events.groupby(
        ["database", "kind"], sort=True, dropna=False
    ):
        path = output_dir / f"{database}_{kind}.parquet"
        group.reset_index(drop=True).to_parquet(path, index=False)
        paths.append(path)
    return paths


def read_event_dataset(paths: Iterable[Path] | Path) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        paths = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
    frames = [pd.read_parquet(Path(path)) for path in paths]
    return canonicalize_event_frame(
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else _empty_event_frame()
    )


def event_frame_to_ledger(frame: pd.DataFrame) -> EventLedger:
    """Materialize a bounded event frame as the indexed causal ledger."""

    records = []
    for record in canonicalize_event_frame(frame).to_dict("records"):
        payload = dict(record)
        for column in (
            "end_time",
            "unit",
            "source_id",
            "hospital_id",
            "care_unit",
            "route",
            "rate_unit",
            "dose_unit",
        ):
            value = payload.get(column)
            if value is None or pd.isna(value):
                payload[column] = None
        for column in ("value", "rate", "dose"):
            value = payload.get(column)
            if value is None or pd.isna(value):
                payload[column] = None
        metadata = payload.get("metadata")
        if isinstance(metadata, str):
            try:
                payload["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                payload["metadata"] = {"raw": metadata}
        records.append(CanonicalEvent.from_mapping(payload))
    return EventLedger(records)


__all__ = [
    "CANONICAL_EVENT_COLUMNS",
    "CANONICAL_UNITS",
    "EICU_RELATIVE_TIME_ORIGIN",
    "canonicalize_event_frame",
    "eicu_relative_timestamp",
    "extract_eicu_events",
    "extract_mimiciv_events",
    "event_frame_to_ledger",
    "read_event_dataset",
    "write_event_dataset",
]
