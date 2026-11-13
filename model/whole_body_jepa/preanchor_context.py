"""Canonical, anchor-safe bedside and observed-care context features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd

from osler_jepa.treatment_units import (
    ACTION_DOSE_DIMENSIONS,
    ACTION_RATE_DIMENSIONS,
    canonical_rate,
)


CONTEXT_CATEGORIES = (
    "hemodynamic",
    "respiratory_support",
    "infection",
    "renal_support",
    "neurologic",
    "fluid_resuscitation",
)
CONTEXT_WINDOWS_MINUTES = (60.0, 360.0, 1440.0)
TREATMENT_ACTIONS = (
    "fluids",
    "vasopressor",
    "diuretics",
    "renal_replacement",
    "nephrotoxin",
)

EICU_EVENT_TABLES = (
    ("treatment.csv.gz", ("treatmentoffset",), ("treatmentstring",)),
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

EVENT_PATTERNS = {
    "hemodynamic": re.compile(r"shock|hypotens|vasopress|pressor|hemodynamic", re.I),
    "respiratory_support": re.compile(
        r"intubat|ventilat|airway|oxygen|hypoxi|respiratory distress", re.I
    ),
    "infection": re.compile(r"sepsis|septic|infection|fever|antibiotic", re.I),
    "renal_support": re.compile(r"dialysis|renal replacement|oliguri|anuri", re.I),
    "neurologic": re.compile(r"altered mental|confus|deliri|sedat|gcs|unresponsive", re.I),
    "fluid_resuscitation": re.compile(
        r"fluid bolus|volume resusc|crystalloid|colloid", re.I
    ),
}

MIMIC_NEURO_ITEMIDS = (220739, 223900, 223901, 226104, 228096)
MIMIC_RESPIRATORY_ITEMIDS = (
    220339,
    223835,
    223849,
    223905,
    223990,
    223991,
    223992,
    224699,
    224700,
    226732,
    229314,
    229323,
    229841,
)


def event_feature_names() -> tuple[str, ...]:
    names = []
    for category in CONTEXT_CATEGORIES:
        names.extend(f"context_{category}_count_{int(minutes // 60)}h" for minutes in CONTEXT_WINDOWS_MINUTES)
        names.append(f"context_{category}_age_24h")
    return tuple(names)


def treatment_feature_names() -> tuple[str, ...]:
    basic = tuple(
        f"context_treatment_{action}_{suffix}"
        for action in TREATMENT_ACTIONS
        for suffix in ("present", "log_count", "dose_observed")
    )
    precise = []
    for action in TREATMENT_ACTIONS:
        for dimension in ACTION_DOSE_DIMENSIONS.get(action, ()):
            precise.append(f"context_treatment_{action}_dose_{dimension}_log1p")
        for dimension in ACTION_RATE_DIMENSIONS.get(action, ()):
            precise.extend(
                (
                    f"context_treatment_{action}_mean_rate_{dimension}_log1p",
                    f"context_treatment_{action}_current_rate_{dimension}_log1p",
                )
            )
            if dimension.endswith("_per_hour"):
                delivered = dimension.removesuffix("_per_hour")
                precise.append(
                    f"context_treatment_{action}_delivered_{delivered}_log1p"
                )
    return basic + tuple(precise)


def summarize_event_offsets(
    stay_ids: Sequence[object],
    anchor_offsets_minutes: Sequence[float],
    events: Mapping[str, Mapping[int, np.ndarray]],
) -> np.ndarray:
    """Summarize only events at or before each anchor into a fixed-width matrix."""

    stay_ids = np.asarray(stay_ids)
    anchors = np.asarray(anchor_offsets_minutes, dtype=np.float64)
    if len(stay_ids) != len(anchors):
        raise ValueError("stay IDs and anchor offsets are misaligned")
    output = np.zeros((len(stay_ids), len(event_feature_names())), dtype=np.float32)
    for row, (stay_value, anchor) in enumerate(zip(stay_ids, anchors)):
        if not np.isfinite(anchor):
            continue
        stay_id = int(stay_value)
        for category_index, category in enumerate(CONTEXT_CATEGORIES):
            offsets = np.asarray(events.get(category, {}).get(stay_id, ()), dtype=np.float64)
            offsets = np.sort(offsets[np.isfinite(offsets)])
            stop = int(np.searchsorted(offsets, anchor, side="right"))
            available = offsets[:stop]
            base = category_index * 4
            for width_index, minutes in enumerate(CONTEXT_WINDOWS_MINUTES):
                start = int(np.searchsorted(available, anchor - minutes, side="left"))
                output[row, base + width_index] = np.log1p(len(available) - start)
            output[row, base + 3] = (
                min(1.0, max(0.0, (anchor - available[-1]) / 1440.0))
                if len(available)
                else 1.0
            )
    return output


def treatment_history_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Build common factual-treatment features; future ``act_*`` is never read."""

    output = np.zeros((len(frame), len(treatment_feature_names())), dtype=np.float32)
    for action_index, action in enumerate(TREATMENT_ACTIONS):
        present = pd.to_numeric(frame.get(f"hist_{action}", 0.0), errors="coerce")
        count = pd.to_numeric(frame.get(f"hist_{action}_evidence_count", 0.0), errors="coerce")
        dose = pd.to_numeric(frame.get(f"hist_{action}_dose_observed", 0.0), errors="coerce")
        if np.isscalar(present):
            present = pd.Series(present, index=frame.index)
        if np.isscalar(count):
            count = pd.Series(count, index=frame.index)
        if np.isscalar(dose):
            dose = pd.Series(dose, index=frame.index)
        base = action_index * 3
        output[:, base] = present.fillna(0.0).gt(0.0).to_numpy(dtype=np.float32)
        output[:, base + 1] = np.log1p(count.fillna(0.0).clip(lower=0.0)).to_numpy(dtype=np.float32)
        output[:, base + 2] = dose.fillna(0.0).gt(0.0).to_numpy(dtype=np.float32)
    offset = len(TREATMENT_ACTIONS) * 3
    for action in TREATMENT_ACTIONS:
        for dimension in ACTION_DOSE_DIMENSIONS.get(action, ()):
            values = pd.to_numeric(
                frame.get(f"hist_{action}_dose_{dimension}_sum", 0.0), errors="coerce"
            )
            if np.isscalar(values):
                values = pd.Series(values, index=frame.index)
            output[:, offset] = np.log1p(values.fillna(0.0).clip(lower=0.0))
            offset += 1
        for dimension in ACTION_RATE_DIMENSIONS.get(action, ()):
            for source in (
                f"hist_{action}_rate_{dimension}_time_weighted_mean",
                f"hist_{action}_current_rate_{dimension}",
            ):
                values = pd.to_numeric(frame.get(source, 0.0), errors="coerce")
                if np.isscalar(values):
                    values = pd.Series(values, index=frame.index)
                output[:, offset] = np.log1p(values.fillna(0.0).clip(lower=0.0))
                offset += 1
            if dimension.endswith("_per_hour"):
                delivered = dimension.removesuffix("_per_hour")
                values = pd.to_numeric(
                    frame.get(f"hist_{action}_delivered_{delivered}_sum", 0.0),
                    errors="coerce",
                )
                if np.isscalar(values):
                    values = pd.Series(values, index=frame.index)
                output[:, offset] = np.log1p(values.fillna(0.0).clip(lower=0.0))
                offset += 1
    return output


def precise_treatment_source_columns() -> tuple[str, ...]:
    """Tabular columns consumed by :func:`treatment_history_matrix`."""

    columns = []
    for action in TREATMENT_ACTIONS:
        for dimension in ACTION_DOSE_DIMENSIONS.get(action, ()):
            columns.append(f"hist_{action}_dose_{dimension}_sum")
        for dimension in ACTION_RATE_DIMENSIONS.get(action, ()):
            columns.extend(
                (
                    f"hist_{action}_rate_{dimension}_time_weighted_mean",
                    f"hist_{action}_current_rate_{dimension}",
                )
            )
            if dimension.endswith("_per_hour"):
                delivered = dimension.removesuffix("_per_hour")
                columns.append(f"hist_{action}_delivered_{delivered}_sum")
    return tuple(columns)


def _rate_with_zero(value: object, unit: object) -> tuple[float, str | None]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan"), None
    if not np.isfinite(number) or number < 0.0:
        return float("nan"), None
    scale, dimension = canonical_rate(1.0, unit)
    if dimension is None:
        return float("nan"), None
    return number * scale, dimension


def load_eicu_infusion_rates(raw_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    """Load unit-normalized eICU infusion snapshots for selected stays."""

    from extractors.eicu_aki_transition_extract import classify_aki_action

    columns = (
        "patientunitstayid",
        "infusionoffset",
        "drugname",
        "drugrate",
        "infusionrate",
    )
    records = []
    for chunk in pd.read_csv(
        raw_root / "infusionDrug.csv.gz",
        compression="gzip",
        usecols=list(columns),
        chunksize=500_000,
        low_memory=False,
    ):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)].copy()
        if chunk.empty:
            continue
        chunk["infusionoffset"] = pd.to_numeric(chunk["infusionoffset"], errors="coerce")
        chunk = chunk[np.isfinite(chunk["infusionoffset"])]
        for row in chunk.itertuples(index=False):
            label = str(row.drugname or "")
            actions = classify_aki_action(label)
            if not actions:
                continue
            match = re.search(r"\(([^()]*)\)\s*$", label)
            unit = match.group(1) if match else ""
            rate, dimension = _rate_with_zero(row.drugrate, unit)
            if dimension is None and "fluids" in actions:
                rate, dimension = _rate_with_zero(row.infusionrate, "ml/hr")
            if dimension is None:
                continue
            for action in actions:
                if dimension not in ACTION_RATE_DIMENSIONS.get(action, ()):
                    continue
                records.append(
                    {
                        "stay_id": int(row.patientunitstayid),
                        "action": action,
                        "label": label,
                        "start": float(row.infusionoffset),
                        "rate": float(rate),
                        "dimension": dimension,
                    }
                )
    if not records:
        return pd.DataFrame(columns=("stay_id", "action", "start", "end", "rate", "dimension"))
    frame = pd.DataFrame.from_records(records).sort_values(
        ["stay_id", "action", "label", "start"], kind="stable"
    )
    return frame.reset_index(drop=True)


def attach_eicu_infusion_history(
    frame: pd.DataFrame,
    anchor_offsets_minutes: Sequence[float],
    infusions: pd.DataFrame,
    *,
    lookback_minutes: float = 360.0,
) -> pd.DataFrame:
    """Attach dimension-safe dose/rate history without using post-anchor flow."""

    output = frame.copy()
    for column in precise_treatment_source_columns():
        output[column] = 0.0
    if infusions.empty:
        return output
    grouped = {
        (int(stay), str(action), str(dimension)): group
        for (stay, action, dimension), group in infusions.groupby(
            ["stay_id", "action", "dimension"], sort=False
        )
    }
    for row_index, (stay_value, anchor_value) in enumerate(
        zip(output["_raw_stay_id"], anchor_offsets_minutes)
    ):
        stay = int(stay_value)
        anchor = float(anchor_value)
        for action in TREATMENT_ACTIONS:
            for dimension in ACTION_RATE_DIMENSIONS.get(action, ()):
                events = grouped.get((stay, action, dimension))
                if events is None:
                    continue
                # Snapshot validity must depend only on records visible now.
                events = events.loc[events["start"] <= anchor].copy()
                if events.empty:
                    continue
                if "label" not in events:
                    events["label"] = action
                events = events.sort_values(["label", "start"], kind="stable").drop_duplicates(
                    ["label", "start"], keep="last"
                )
                next_start = events.groupby("label")["start"].shift(-1)
                events["end"] = np.minimum(next_start.fillna(np.inf), events["start"] + 60.0)
                overlap = np.maximum(
                    0.0,
                    np.minimum(events["end"].to_numpy(dtype=float), anchor)
                    - np.maximum(events["start"].to_numpy(dtype=float), anchor - lookback_minutes),
                )
                valid = overlap > 0.0
                if not valid.any():
                    continue
                rates = events["rate"].to_numpy(dtype=float)
                mean = float(np.sum(rates[valid] * overlap[valid]) / np.sum(overlap[valid]))
                active = (
                    (events["start"].to_numpy(dtype=float) <= anchor)
                    & (events["end"].to_numpy(dtype=float) >= anchor)
                )
                active &= events["end"].to_numpy(dtype=float) > anchor
                current = float(rates[active].sum()) if active.any() else 0.0
                output.iat[
                    row_index,
                    output.columns.get_loc(
                        f"hist_{action}_rate_{dimension}_time_weighted_mean"
                    ),
                ] = mean
                output.iat[
                    row_index,
                    output.columns.get_loc(f"hist_{action}_current_rate_{dimension}"),
                ] = current
                if dimension.endswith("_per_hour"):
                    delivered = dimension.removesuffix("_per_hour")
                    output.iat[
                        row_index,
                        output.columns.get_loc(f"hist_{action}_delivered_{delivered}_sum"),
                    ] = float(np.sum(rates[valid] * overlap[valid] / 60.0))
    return output


def build_preanchor_context(
    frame: pd.DataFrame,
    anchor_offsets_minutes: Sequence[float],
    events: Mapping[str, Mapping[int, np.ndarray]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    event = summarize_event_offsets(frame["_raw_stay_id"], anchor_offsets_minutes, events)
    treatment = treatment_history_matrix(frame)
    return np.concatenate((event, treatment), axis=1), event_feature_names() + treatment_feature_names()


def _finalize_events(
    collected: Mapping[str, Mapping[int, Sequence[float]]],
) -> dict[str, dict[int, np.ndarray]]:
    return {
        category: {
            int(stay_id): np.sort(np.asarray(offsets, dtype=np.float64))
            for stay_id, offsets in by_stay.items()
        }
        for category, by_stay in collected.items()
    }


def load_eicu_event_offsets(
    raw_root: Path,
    stay_ids: set[int],
) -> dict[str, dict[int, np.ndarray]]:
    """Read timestamped eICU assessments and care intent for selected stays."""

    collected: dict[str, dict[int, list[float]]] = {
        category: {} for category in CONTEXT_CATEGORIES
    }
    for filename, offset_columns, text_columns in EICU_EVENT_TABLES:
        path = raw_root / filename
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=["patientunitstayid", *offset_columns, *text_columns],
            chunksize=250_000,
            low_memory=False,
        ):
            chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)].copy()
            if chunk.empty:
                continue
            offsets = chunk[list(offset_columns)].apply(pd.to_numeric, errors="coerce").max(
                axis=1, skipna=True
            )
            valid = np.isfinite(offsets)
            chunk = chunk.loc[valid]
            offsets = offsets.loc[valid]
            text = chunk[list(text_columns)].fillna("").astype(str).agg(" ".join, axis=1)
            for category, pattern in EVENT_PATTERNS.items():
                matched = text.str.contains(pattern)
                for stay_id, values in offsets[matched].groupby(
                    chunk.loc[matched, "patientunitstayid"]
                ):
                    collected[category].setdefault(int(stay_id), []).extend(
                        values.astype(float).tolist()
                    )
    return _finalize_events(collected)


def load_mimic_stays(raw_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    """Load ICU admission provenance needed to align hospital orders to anchors."""

    stays = pd.read_csv(
        raw_root / "icu" / "icustays.csv.gz",
        compression="gzip",
        usecols=["stay_id", "hadm_id", "intime", "outtime"],
        low_memory=False,
    )
    stays = stays[stays["stay_id"].isin(stay_ids)].copy()
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
    return stays.dropna(subset=["intime", "outtime"])


def load_mimic_event_offsets(
    raw_root: Path,
    stays: pd.DataFrame,
) -> dict[str, dict[int, np.ndarray]]:
    """Read MIMIC bedside observations and POE intent before selected anchors."""

    import duckdb

    if stays.empty:
        return {category: {} for category in CONTEXT_CATEGORIES}
    connection = duckdb.connect()
    connection.register("selected_stays", stays)
    poe_path = str(raw_root / "hosp" / "poe.csv.gz").replace("'", "''")
    chart_path = str(raw_root / "icu" / "chartevents.csv.gz").replace("'", "''")
    poe = connection.execute(
        f"""
        SELECT s.stay_id::BIGINT stay_id,
               epoch(CAST(p.ordertime AS TIMESTAMP) - s.intime) / 60.0 event_offset,
               CASE
                 WHEN lower(p.order_type) = 'critical care'
                   AND lower(p.order_subtype) LIKE '%hemodynamic%' THEN 'hemodynamic'
                 WHEN lower(p.order_type) = 'general care'
                   AND (lower(p.order_subtype) LIKE '%vitals/monitoring%'
                     OR lower(p.order_subtype) LIKE '%telemetry%') THEN 'hemodynamic'
                 WHEN lower(p.order_type) = 'respiratory' THEN 'respiratory_support'
                 WHEN lower(p.order_type) = 'hemodialysis'
                   OR lower(p.order_subtype) = 'dialysis'
                   OR lower(p.order_subtype) = 'nephrology' THEN 'renal_support'
                 WHEN lower(p.order_type) = 'neurology'
                   OR lower(p.order_subtype) LIKE '%intra-cranial%'
                   OR lower(p.order_subtype) IN ('neurology', 'psychiatry') THEN 'neurologic'
                 WHEN lower(p.order_subtype) = 'infectious disease' THEN 'infection'
                 WHEN lower(p.order_type) = 'iv therapy'
                   AND lower(p.order_subtype) = 'iv fluids' THEN 'fluid_resuscitation'
               END category
        FROM read_csv_auto('{poe_path}', header=true) p
        JOIN selected_stays s ON p.hadm_id = s.hadm_id
        WHERE lower(coalesce(p.transaction_type, '')) = 'new'
          AND CAST(p.ordertime AS TIMESTAMP) BETWEEN s.intime AND s.outtime
          AND category IS NOT NULL
        """
    ).df()
    neuro_items = ",".join(map(str, MIMIC_NEURO_ITEMIDS))
    respiratory_items = ",".join(map(str, MIMIC_RESPIRATORY_ITEMIDS))
    chart = connection.execute(
        f"""
        SELECT c.stay_id::BIGINT stay_id,
               epoch(greatest(CAST(c.charttime AS TIMESTAMP), CAST(c.storetime AS TIMESTAMP))
                     - s.intime) / 60.0 event_offset,
               CASE
                 WHEN c.itemid IN ({neuro_items}) THEN 'neurologic'
                 WHEN c.itemid IN ({respiratory_items}) THEN 'respiratory_support'
               END category
        FROM read_csv_auto('{chart_path}', header=true) c
        JOIN selected_stays s ON c.stay_id = s.stay_id
        WHERE c.itemid IN ({neuro_items},{respiratory_items})
          AND CAST(c.charttime AS TIMESTAMP) BETWEEN s.intime AND s.outtime
          AND c.storetime IS NOT NULL
        """
    ).df()
    connection.close()
    events = pd.concat((poe, chart), ignore_index=True)
    collected: dict[str, dict[int, list[float]]] = {
        category: {} for category in CONTEXT_CATEGORIES
    }
    events = events[np.isfinite(pd.to_numeric(events["event_offset"], errors="coerce"))]
    for (category, stay_id), group in events.groupby(["category", "stay_id"]):
        collected[str(category)][int(stay_id)] = group["event_offset"].astype(float).tolist()
    return _finalize_events(collected)


__all__ = [
    "CONTEXT_CATEGORIES",
    "CONTEXT_WINDOWS_MINUTES",
    "TREATMENT_ACTIONS",
    "build_preanchor_context",
    "event_feature_names",
    "load_eicu_event_offsets",
    "load_eicu_infusion_rates",
    "load_mimic_event_offsets",
    "load_mimic_stays",
    "attach_eicu_infusion_history",
    "precise_treatment_source_columns",
    "summarize_event_offsets",
    "treatment_feature_names",
    "treatment_history_matrix",
]
