"""Scoring past forecasts against what was measured afterwards.

Every completed run forecast its cells at an anchor. Once the stream has moved
past a cell's forecast time, the forecast is compared with the measurement the
cells were fitted against: the target's value nearest to ``anchor + horizon``,
after the anchor and within the label tolerance (``min(2, max(0.25, horizon / 2))``
hours, the rule of the rows the cells were fitted on). Beside each forecast's
error stands the error of carrying the anchor's value forward, which is the
baseline every cell had to beat when it was validated.

A score is a fact about this stream, not about the model: one patient and a
handful of anchors. The document states its counts and says so.

A forecast is ``resolved`` when a qualifying measurement exists and no nearer one
can still arrive, ``unmeasured`` when the window closed without one, and
``pending`` while the stream has not yet passed the end of the window. Only the
newest completed run of each anchor is scored; an anchor that was re-run after a
late result is one forecast, not several.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from .fhir import iso, parse_instant
from .forecast_runtime import STATUS_MODEL
from .repository import Repository
from .timeline import delivered_timeline

TARGET_TOLERANCE_MAX_HOURS = 2.0


def label_tolerance_hours(horizon_hours: float) -> float:
    """The label window of the fitted rows: a short horizon must not pick up the current observation."""
    return min(TARGET_TOLERANCE_MAX_HOURS, max(0.25, float(horizon_hours) / 2.0))


def _latest_run_per_anchor(repo: Repository, session_id: str, epoch: int, patient_id: str) -> list[dict]:
    newest: dict[str, dict] = {}
    for run in repo.runs_for_patient(patient_id, limit=10_000):          # newest first
        if run["session_id"] == session_id and int(run["epoch"]) == epoch and run["status"] == "completed":
            newest.setdefault(run["input_cutoff"], run)
    return sorted(newest.values(), key=lambda run: run["input_cutoff"])


def score_forecasts(repo: Repository, session_id: str, epoch: int, patient_id: str) -> dict[str, Any]:
    timeline = delivered_timeline(repo, session_id, epoch)
    by_variable: dict[str, list[tuple]] = {}
    for row in timeline:
        by_variable.setdefault(row["variable"], []).append((parse_instant(row["effective_time"], "effective_time"), row))
    now = max((parse_instant(row["effective_time"], "effective_time") for row in timeline), default=None)
    items = []
    for run in _latest_run_per_anchor(repo, session_id, epoch, patient_id):
        anchor = parse_instant(run["input_cutoff"], "input_cutoff")
        for cell in repo.cells_for_run(run["run_id"]):
            if cell["status"] != STATUS_MODEL or cell["forecast_point"] is None:
                continue
            horizon = float(cell["horizon_hours"])
            due = anchor + timedelta(hours=horizon)
            tolerance = timedelta(hours=label_tolerance_hours(horizon))
            window = [(abs(at - due), at, row) for at, row in by_variable.get(cell["target"], [])
                      if at > anchor and abs(at - due) <= tolerance]
            item = {"run_id": run["run_id"], "cell": cell["cell"], "target": cell["target"], "unit": cell["unit"],
                    "horizon_hours": int(cell["horizon_hours"]), "anchor_time": run["input_cutoff"],
                    "forecast_time": iso(due), "tolerance_hours": label_tolerance_hours(horizon),
                    "forecast_point": float(cell["forecast_point"]), "anchor_value": cell["observed_value"]}
            if now is None or now < due + tolerance:
                items.append({**item, "state": "pending"})
            elif not window:
                items.append({**item, "state": "unmeasured"})
            else:
                _, at, row = min(window, key=lambda entry: (entry[0], entry[1]))
                value = float(row["value"])
                items.append({**item, "state": "resolved", "measured_value": value, "measured_time": row["effective_time"],
                              "reference": f"Observation/{row['resource_id']}/_history/{row['version_id']}",
                              "forecast_error": item["forecast_point"] - value,
                              "carried_forward_error": (None if cell["observed_value"] is None
                                                        else float(cell["observed_value"]) - value)})
    cells: dict[str, dict[str, Any]] = {}
    for item in items:
        entry = cells.setdefault(item["cell"], {"cell": item["cell"], "target": item["target"], "unit": item["unit"],
                                                "horizon_hours": item["horizon_hours"], "resolved": 0, "pending": 0,
                                                "unmeasured": 0, "_forecast": [], "_carried": []})
        entry[item["state"]] += 1
        if item["state"] == "resolved" and item["carried_forward_error"] is not None:
            entry["_forecast"].append(abs(item["forecast_error"]))
            entry["_carried"].append(abs(item["carried_forward_error"]))
    summary = []
    for entry in sorted(cells.values(), key=lambda e: (e["target"], e["horizon_hours"])):
        forecast, carried = entry.pop("_forecast"), entry.pop("_carried")
        entry["compared"] = len(forecast)
        entry["mean_abs_error_forecast"] = sum(forecast) / len(forecast) if forecast else None
        entry["mean_abs_error_carried_forward"] = sum(carried) / len(carried) if carried else None
        summary.append(entry)
    counts = {state: sum(1 for item in items if item["state"] == state) for state in ("resolved", "pending", "unmeasured")}
    return {"schema": "oslian.forecast-scores.v1", "patient_id": patient_id, "session_id": session_id, "epoch": epoch,
            "as_of": iso(now) if now else None, "anchors_scored": len({item["anchor_time"] for item in items}),
            "counts": counts, "cells": summary, "forecasts": items,
            "scope": "this stream only: one patient and a few anchors; not the model's accuracy at a hospital",
            "label_rule": "the target's value nearest to anchor + horizon, after the anchor, within min(2, max(0.25, horizon / 2)) hours",
            "baseline": "carried forward = the anchor's own value, repeated", "causal_claim_allowed": False}
