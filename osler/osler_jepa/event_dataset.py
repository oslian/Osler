"""Build JEPA-ready factual examples from the canonical event ledger."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .event_ledger import (
    DEFAULT_MEASUREMENT_AGE_HOURS,
    DEFAULT_TARGET_LABEL_POLICIES,
    EventKind,
    EventLedger,
    TargetLabelPolicy,
    canonical_timestamp,
    canonical_variable,
)


def _treatment_summary(context: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate only treatment facts already known at the anchor."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in context:
        key = canonical_variable(str(event.get("variable", "unknown")))
        groups[key].append(event)
    output: dict[str, float] = {}
    for variable, events in groups.items():
        output[f"hist_{variable}_evidence_count"] = float(len(events))
        rates = [
            float(event["rate"])
            for event in events
            if event.get("rate") is not None
            and np.isfinite(float(event["rate"]))
        ]
        doses = [
            float(event["dose"])
            for event in events
            if event.get("dose") is not None
            and np.isfinite(float(event["dose"]))
        ]
        output[f"hist_{variable}_last_rate"] = rates[-1] if rates else np.nan
        output[f"hist_{variable}_last_dose"] = doses[-1] if doses else np.nan
        output[f"_observed_hist_{variable}_rate"] = float(bool(rates))
        output[f"_observed_hist_{variable}_dose"] = float(bool(doses))
        output[f"_treatment_observed_hist_{variable}"] = 1.0
    return output


def build_event_examples(
    ledger: EventLedger,
    anchors: pd.DataFrame,
    variables: Sequence[str],
    horizons: Sequence[float],
    *,
    policies: Mapping[str, TargetLabelPolicy] | None = None,
    max_age_hours: float | Mapping[str, float] = DEFAULT_MEASUREMENT_AGE_HOURS,
) -> pd.DataFrame:
    """Create one strict factual training row per anchor and horizon.

    Required anchor columns are ``encounter_id`` and ``anchor_time``. Optional
    ``patient_id``, ``hospital_id``, ``care_unit`` and ``database`` columns are
    copied only as provenance/splitting metadata. They are not physiological
    features.
    """

    required = {"encounter_id", "anchor_time"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(f"anchors missing required columns: {sorted(missing)}")
    variables = tuple(canonical_variable(value) for value in variables)
    policies = dict(DEFAULT_TARGET_LABEL_POLICIES if policies is None else policies)
    rows = []
    for anchor_record in anchors.to_dict(orient="records"):
        encounter_id = str(anchor_record["encounter_id"])
        anchor_time = canonical_timestamp(anchor_record["anchor_time"])
        view = ledger.anchor_view(
            encounter_id,
            anchor_time,
            variables,
            max_age_hours=max_age_hours,
        )
        treatment = _treatment_summary(view["observed_treatment_context"])
        base = {
            "patient_id": str(anchor_record.get("patient_id", "")),
            "subject_id": str(anchor_record.get("patient_id", "")),
            "encounter_id": encounter_id,
            "stay_id": encounter_id,
            "anchor_time": anchor_time,
            "t": anchor_time,
            "database": anchor_record.get("database"),
            "hospitalid": anchor_record.get("hospital_id"),
            "careunit": anchor_record.get("care_unit"),
            **treatment,
        }
        for variable in variables:
            base[f"{variable}_t"] = view["values"][variable]
            base[f"{variable}_age_hr"] = view["measurement_age_hours"][variable]
            base[f"_observed_{variable}"] = float(view["observed"][variable])
            provenance = view["provenance"][variable] or {}
            base[f"_source_table_{variable}"] = provenance.get("source_table")
            base[f"_source_database_{variable}"] = provenance.get("database")

        for horizon in horizons:
            row = dict(base)
            row["_source_horizon"] = float(horizon)
            row["horizon_hours"] = float(horizon)
            for variable in variables:
                label = ledger.future_label(
                    encounter_id,
                    variable,
                    anchor_time,
                    float(horizon),
                    policy=policies.get(variable),
                )
                row[f"future_{variable}"] = label["value"]
                row[f"future_event_time_{variable}"] = label["event_time"]
                row[f"future_delta_t_hr_{variable}"] = label[
                    "actual_delta_t_hours"
                ]
                row[f"future_label_quality_{variable}"] = label["quality"]
                row[f"future_label_weight_{variable}"] = label[
                    "training_weight"
                ]
                row[f"next_measurement_time_hr_{variable}"] = (
                    ledger.next_measurement_time(
                        encounter_id, variable, anchor_time
                    )
                )
            rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    frame.attrs["event_ledger_contract"] = {
        "schema": "whole_body_event_examples.v1",
        "measurement_and_treatment_events_are_timestamped": True,
        "future_labels_are_real_post_anchor_events": True,
        "future_delta_t_is_target_specific": True,
        "measurement_process_is_separate": True,
        "hospital_provenance_is_not_a_physiology_feature": True,
        "future_treatment_is_excluded": True,
        "causal_claim_allowed": False,
    }
    return frame


def ledger_readiness_report(
    frame: pd.DataFrame, variables: Sequence[str]
) -> dict[str, Any]:
    """Summarize support without emitting row-level patient data."""

    targets = {}
    for variable in variables:
        variable = canonical_variable(variable)
        weight = pd.to_numeric(
            frame.get(
                f"future_label_weight_{variable}",
                pd.Series(0.0, index=frame.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        delta = pd.to_numeric(
            frame.get(
                f"future_delta_t_hr_{variable}",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        )
        targets[variable] = {
            "rows": int(len(frame)),
            "strict_supervision_rows": int((weight > 0.0).sum()),
            "actual_delta_t_median": (
                float(delta[weight > 0.0].median())
                if bool((weight > 0.0).any())
                else None
            ),
        }
    return {
        "schema": "whole_body_event_ledger_readiness.v1",
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()) if "subject_id" in frame else 0,
        "encounters": int(frame["stay_id"].nunique()) if "stay_id" in frame else 0,
        "targets": targets,
        "row_level_data_included": False,
        "causal_claim_allowed": False,
    }


__all__ = ["build_event_examples", "ledger_readiness_report"]
