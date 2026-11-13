"""Patient timeline, watermark, per-variable state, and model input assembly.

The only path from the ledger to the model runs through ``build_model_input``,
which takes the delivered timeline filtered by ``effective_time <= cutoff``.
Nothing after the cutoff is ever serialised into the model CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import csv
from typing import Any

from .fhir import parse_instant, iso
from .repository import Repository
from .routing import variable_routing


@dataclass(frozen=True)
class ModelInput:
    cutoff: str
    watermark: str | None            # latest effective_time actually included
    csv_text: str
    rows: int
    resource_refs: tuple[str, ...]   # Observation/<id>/_history/<version>
    input_hash: str
    onset: str


def delivered_timeline(repo: Repository, session_id: str, epoch: int, cutoff: str | None = None) -> list[dict]:
    return repo.delivered_observations(session_id, epoch, cutoff)


def variable_state(timeline: list[dict], now: str | None) -> dict[str, dict[str, Any]]:
    """Latest value, age (relative to ``now``), and count per variable."""
    now_dt = parse_instant(now, "now") if now else None
    state: dict[str, dict[str, Any]] = {}
    for row in timeline:
        item = state.setdefault(row["variable"], {"count": 0, "latest": None})
        item["count"] += 1
        item["latest"] = row
    out = {}
    for variable, item in state.items():
        latest = item["latest"]
        age = None
        if now_dt is not None:
            age = (now_dt - parse_instant(latest["effective_time"], "effective_time")).total_seconds() / 3600.0
        out[variable] = {
            "variable": variable,
            "value": float(latest["value"]),
            "unit": latest["unit"],
            "effective_time": latest["effective_time"],
            "received_at": latest.get("delivered_at") or latest["received_at"],
            "resource_id": latest["resource_id"],
            "version_id": latest["version_id"],
            "age_hours": age,
            "count": item["count"],
            # the value as the source stated it, before unit conversion: a reference
            # range the observation supplies is written in this unit, not the canonical one
            "input_value": (None if latest.get("input_value") is None else float(latest["input_value"])),
            "input_unit": latest.get("input_unit"),
            # the mapping's own word for how canonical came from stated: "identity" means
            # the same number under another unit name, which is what lets a forecast in
            # the canonical unit be read against a range stated in the observation's unit
            "derivation": latest.get("derivation"),
        }
    return out


def _quantity_value(quantity: Any) -> float | None:
    if not isinstance(quantity, dict):
        return None
    value = quantity.get("value")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _quantity_unit(quantity: Any) -> str | None:
    if not isinstance(quantity, dict):
        return None
    unit = quantity.get("unit") or quantity.get("code")
    return unit if isinstance(unit, str) else None


def observation_statements(resource: dict | None) -> dict[str, Any]:
    """What an Observation says about its own value, passed through unchanged in meaning.

    Two FHIR elements can carry that statement: ``interpretation`` (the source's
    own coded reading, for example ``H`` or ``N`` from v3-ObservationInterpretation)
    and ``referenceRange`` (the interval the source compared the value against,
    in the unit the source used). Both are returned as lists, empty when the
    observation carries neither. Nothing here decides whether a value is normal:
    that decision is the observation's own statement or nobody's, and the page
    that reads these fields says "Range not supplied" when both lists are empty.
    """
    out: dict[str, Any] = {"interpretation": [], "reference_range": []}
    if not isinstance(resource, dict):
        return out
    for concept in resource.get("interpretation") or []:
        if not isinstance(concept, dict):
            continue
        # the concept's own text travels with each code: a fixture-derived reading
        # names itself there, and a reader of the state can tell it from a source's
        provenance = concept.get("text") if isinstance(concept.get("text"), str) else None
        for coding in concept.get("coding") or []:
            if isinstance(coding, dict) and isinstance(coding.get("code"), str):
                out["interpretation"].append({"system": coding.get("system"), "code": coding["code"],
                                              "display": coding.get("display"), "provenance": provenance})
    for item in resource.get("referenceRange") or []:
        if not isinstance(item, dict):
            continue
        low, high = item.get("low"), item.get("high")
        out["reference_range"].append({
            "low": _quantity_value(low), "low_unit": _quantity_unit(low),
            "high": _quantity_value(high), "high_unit": _quantity_unit(high),
            "text": item.get("text") if isinstance(item.get("text"), str) else None,
            # a range qualified by population or age applies to some patients only;
            # the reader of this field decides whether it applies, never this code
            "applies_to": bool(item.get("appliesTo")),
            "age": bool(item.get("age")),
        })
    return out


def attach_observation_statements(repo: Repository, source_system: str,
                                  variables: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Join each variable's latest raw Observation back on, for its own statements."""
    for state in variables.values():
        raw = repo.raw_by_reference(source_system, "Observation", state["resource_id"], state["version_id"])
        state.update(observation_statements(raw["resource"] if raw else None))
    return variables


def organ_state(timeline: list[dict], now: str | None) -> dict[str, dict[str, Any]]:
    """Latest delivered observation per canonical organ, for a snapshot after a refresh.

    Streamed events carry ``organ_ids`` so the browser never has to guess an organ
    from a variable name. A page that reloads mid-stream gets no events for what
    was already delivered, so the snapshot has to state the same thing, resolved
    through the same approved map: ``variable_routing()``. The browser stores these
    exactly as it stores an event's routing, which is why the lights that were on
    before the reload are the lights that come back.
    """
    routing = variable_routing()
    now_dt = parse_instant(now, "now") if now else None
    out: dict[str, dict[str, Any]] = {}
    for row in timeline:
        for organ in routing.organs_for(row["variable"]):
            item = out.setdefault(organ, {"organ": organ, "at": None, "sequence": 0,
                                          "variables": [], "count": 0, "age_hours": None})
            item["count"] += 1
            if row["variable"] not in item["variables"]:
                item["variables"].append(row["variable"])
            if item["at"] is None or row["effective_time"] >= item["at"]:
                item["at"] = row["effective_time"]
                item["sequence"] = int(row.get("source_sequence") or 0)
    for item in out.values():
        item["variables"].sort()
        if now_dt is not None and item["at"] is not None:
            item["age_hours"] = (now_dt - parse_instant(item["at"], "effective_time")).total_seconds() / 3600.0
    return out


def run_timeline(repo: Repository, run: dict) -> list[dict]:
    """The observations one model run actually consumed, in timeline order.

    This is the only admissible basis for any historical view. "The current timeline
    truncated at the cutoff" is not the same set: a resource delivered after the run
    but timestamped before its cutoff would slip in, and an amendment that superseded
    one of the run's inputs would replace it - either way the past would change under
    a view that says it is frozen. The run recorded the refs it consumed, so they are
    read back by ref.
    """
    return repo.observations_for_refs(run["session_id"], int(run["epoch"]), run.get("input_refs") or [])


def organ_state_for_run(repo: Repository, run: dict) -> dict[str, dict[str, Any]]:
    """Organ snapshot for one model run, aged against that run's own ``input_cutoff``."""
    return organ_state(run_timeline(repo, run), run["input_cutoff"])


def series(timeline: list[dict], variable: str) -> list[dict[str, Any]]:
    return [
        {"effective_time": r["effective_time"], "value": float(r["value"]), "unit": r["unit"],
         "resource_id": r["resource_id"], "version_id": r["version_id"], "status": r["observation_status"],
         "received_at": r.get("delivered_at") or r["received_at"], "derivation": r["derivation"]}
        for r in timeline if r["variable"] == variable
    ]


def build_model_input(timeline: list[dict], cutoff: str, onset: str) -> ModelInput:
    """Serialise the causal prefix (effective_time <= cutoff) as long-format CSV."""
    cutoff_dt = parse_instant(cutoff, "cutoff")
    onset_dt = parse_instant(onset, "onset")
    rows: list[tuple[float, str, float]] = []
    refs: list[str] = []
    watermark: datetime | None = None
    for row in timeline:
        at = parse_instant(row["effective_time"], "effective_time")
        if at > cutoff_dt:
            raise ValueError("timeline row after the input cutoff reached model input assembly")
        hours = (at - onset_dt).total_seconds() / 3600.0
        rows.append((round(hours, 6), row["variable"], float(row["value"])))
        refs.append(f"Observation/{row['resource_id']}/_history/{row['version_id']}")
        watermark = at if watermark is None or at > watermark else watermark
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["hours_since_onset", "variable", "value"])
    for hours, variable, value in rows:
        writer.writerow([f"{hours:.6f}", variable, f"{value:.6f}"])
    text = buffer.getvalue()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ModelInput(cutoff=cutoff, watermark=iso(watermark) if watermark else None, csv_text=text,
                      rows=len(rows), resource_refs=tuple(dict.fromkeys(refs)), input_hash=digest, onset=onset)
