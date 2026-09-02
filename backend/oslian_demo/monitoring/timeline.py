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
        }
    return out


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
