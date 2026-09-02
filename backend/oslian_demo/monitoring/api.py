"""Flask blueprint: REST session control + one-way SSE event stream.

Endpoints
    POST /api/demo-streams                      get-or-create the synthetic session (idempotent)
    GET  /api/demo-streams                      list sessions
    GET  /api/demo-streams/<id>                 session state
    POST /api/demo-streams/<id>/start|pause|reset   idempotent controls
    POST /api/demo-streams/<id>/speed           {"speed": 1|5|20}
    GET  /api/demo-streams/<id>/events          SSE (id: = sequence; Last-Event-ID or ?last_event_id=)
    GET  /api/patients/<id>/state               clocks, watermark, per-variable state, run head
    GET  /api/patients/<id>/timeline            delivered timeline (optionally up to a run's cutoff)
    GET  /api/patients/<id>/forecasts           cells for the live head or ?run_id=
    GET  /api/patients/<id>/forecast-history    every run, newest first
    GET  /api/patients/<id>/mapping-failures
    GET  /api/fhir/<type>/<id>/_history/<vid>   raw resource from the ledger (provenance drawer)
    GET  /api/model/capabilities | health | anatomy | mapping
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import time
from typing import Any

from flask import Blueprint, Response, jsonify, request

from ..config import AppConfig
from .anatomy import anatomy_document
from .eventlog import DELIVERY_LIVE, DELIVERY_REPLAY, StreamEventLogger
from .functions import functions_document
from .mapping import registry_document
from .repository import Repository
from .session import SPEEDS, SessionManager, StreamSession
from .source import SOURCES, make_source
from .timeline import (delivered_timeline, organ_state, organ_state_for_run, run_timeline, series,
                       variable_state)
from .wholebody import RuntimeConfig, WholeBodyRuntime

SSE_EVENT_TYPES = (
    "stream.state", "fhir.resource.accepted", "fhir.resource.rejected", "observation.normalized",
    "observation.unmapped", "model.run.started", "model.run.completed", "model.run.failed",
)
class StreamService:
    def __init__(self, db_path: str | Path | None = None, runtime: WholeBodyRuntime | None = None,
                 source_factory=None, logger: StreamEventLogger | None = None):
        self.repo = Repository(db_path or AppConfig.from_env().stream_db)
        self.runtime = runtime or WholeBodyRuntime(RuntimeConfig.from_env())
        self.logger = logger or StreamEventLogger.from_env(dict(os.environ))
        self.sessions = SessionManager(self.repo, self.runtime, source_factory or make_source, self.logger)

    def session_for_patient(self, patient_id: str, session_id: str | None) -> StreamSession | None:
        if session_id:
            session = self.sessions.get(session_id)
            return session if session and session.patient_id == patient_id else None
        candidates = [s for s in self.sessions.all() if s.patient_id == patient_id]
        return candidates[-1] if candidates else None

    def close(self) -> None:
        self.sessions.shutdown()
        self.repo.close()


def create_blueprint(service: StreamService) -> Blueprint:
    bp = Blueprint("stream", __name__)

    def _session_or_404(session_id: str):
        session = service.sessions.get(session_id)
        if session is None:
            return None, (jsonify({"error": "unknown session"}), 404)
        return session, None

    # ---- sessions -----------------------------------------------------------
    @bp.post("/api/demo-streams")
    def create_stream():
        body = request.get_json(silent=True) or {}
        source_key = str(body.get("source") or "synthetic")
        if source_key not in SOURCES:
            return jsonify({"error": f"unknown source; expected one of {sorted(SOURCES)}"}), 400
        session, created = service.sessions.get_or_create(source_key)
        return jsonify(session.state()), (201 if created else 200)

    @bp.get("/api/demo-streams/sources")
    def list_sources():
        return jsonify({"sources": [make_source(key).describe() for key in SOURCES]})

    @bp.post("/api/demo-streams/<session_id>/step")
    def step_stream(session_id: str):
        session, error = _session_or_404(session_id)
        return error or jsonify(session.step())

    @bp.get("/api/demo-streams")
    def list_streams():
        return jsonify({"sessions": [s.state() for s in service.sessions.all()]})

    @bp.get("/api/demo-streams/<session_id>")
    def get_stream(session_id: str):
        session, error = _session_or_404(session_id)
        return error or jsonify(session.state())

    @bp.post("/api/demo-streams/<session_id>/start")
    def start_stream(session_id: str):
        session, error = _session_or_404(session_id)
        return error or jsonify(session.start())

    @bp.post("/api/demo-streams/<session_id>/pause")
    def pause_stream(session_id: str):
        session, error = _session_or_404(session_id)
        return error or jsonify(session.pause())

    @bp.post("/api/demo-streams/<session_id>/reset")
    def reset_stream(session_id: str):
        session, error = _session_or_404(session_id)
        if error:
            return error
        state = session.reset()
        service.sessions.deliver_snapshot_for_epoch(session)
        return jsonify(state)

    @bp.post("/api/demo-streams/<session_id>/speed")
    def speed_stream(session_id: str):
        session, error = _session_or_404(session_id)
        if error:
            return error
        body = request.get_json(silent=True) or {}
        try:
            speed = float(body.get("speed"))
        except (TypeError, ValueError):
            return jsonify({"error": f"speed must be one of {list(SPEEDS)}"}), 400
        if speed not in SPEEDS:
            return jsonify({"error": f"speed must be one of {list(SPEEDS)}"}), 400
        return jsonify(session.set_speed(speed))

    @bp.get("/api/demo-streams/<session_id>/events")
    def stream_events(session_id: str):
        session, error = _session_or_404(session_id)
        if error:
            return error
        last = request.headers.get("Last-Event-ID") or request.args.get("last_event_id") or "0"
        try:
            last_sequence = int(last)
        except ValueError:
            last_sequence = 0
        return Response(_sse(service, session, last_sequence), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    # ---- patients -------------------------------------------------------------
    def _patient_context(patient_id: str):
        session = service.session_for_patient(patient_id, request.args.get("session_id"))
        if session is None:
            return None, None, (jsonify({"error": "unknown patient or session"}), 404)
        return session, service.repo.patient(patient_id), None

    @bp.get("/api/patients/<patient_id>/state")
    def patient_state(patient_id: str):
        session, patient, error = _patient_context(patient_id)
        if error:
            return error
        timeline = delivered_timeline(service.repo, session.session_id, session.epoch)
        latest_clinical = max((r["effective_time"] for r in timeline), default=None)
        latest_received = max((r.get("delivered_at") or r["received_at"] for r in timeline), default=None)
        live = service.repo.latest_completed_run(session.session_id, session.epoch)
        latest_run = service.repo.latest_run(session.session_id, session.epoch)
        encounter = service.repo.encounter_for_patient(patient_id)
        return jsonify({
            "patient": {"patient_id": patient_id, "display": _patient_display(patient), "synthetic": True,
                        "encounter_id": encounter["encounter_id"] if encounter else None,
                        "encounter_start": encounter["period_start"] if encounter else None},
            "session": session.state(),
            "clock": {
                "latest_clinical_event_time": latest_clinical,
                "latest_received_at": latest_received,
                "model_input_watermark": live["input_cutoff"] if live else None,
                "live_run_id": live["run_id"] if live else None,
                "latest_run_status": latest_run["status"] if latest_run else None,
                "latest_run_id": latest_run["run_id"] if latest_run else None,
            },
            "runtime": service.runtime.provenance(),
            "variables": variable_state(timeline, latest_clinical),
            # organ routing for what was already delivered, so a reload restores the
            # same organ states the events had produced; resolved by the same map
            "organs": organ_state(timeline, latest_clinical),
            "delivered_observations": len(timeline),
            "mapping_failures": len(service.repo.mapping_failures(patient_id)),
        })

    @bp.get("/api/patients/<patient_id>/timeline")
    def patient_timeline(patient_id: str):
        session, _, error = _patient_context(patient_id)
        if error:
            return error
        cutoff = None
        run_id = request.args.get("run_id")
        if run_id:
            run = service.repo.run(run_id)
            if run is None or run["patient_id"] != patient_id:
                return jsonify({"error": "unknown run"}), 404
            cutoff = run["input_cutoff"]
            # the run's own inputs, not the current timeline truncated at its cutoff:
            # the signals, the charts and the organs must all describe the same set
            timeline = run_timeline(service.repo, run)
        else:
            timeline = delivered_timeline(service.repo, session.session_id, session.epoch)
        variable = request.args.get("variable")
        if variable:
            return jsonify({"patient_id": patient_id, "variable": variable, "cutoff": cutoff,
                            "series": series(timeline, variable)})
        variables = sorted({r["variable"] for r in timeline})
        return jsonify({"patient_id": patient_id, "cutoff": cutoff, "count": len(timeline),
                        "series": {v: series(timeline, v) for v in variables},
                        "state": variable_state(timeline, cutoff or (timeline[-1]["effective_time"] if timeline else None))})

    @bp.get("/api/patients/<patient_id>/forecasts")
    def patient_forecasts(patient_id: str):
        session, _, error = _patient_context(patient_id)
        if error:
            return error
        run_id = request.args.get("run_id")
        if run_id:
            run = service.repo.run(run_id)
            if run is None or run["patient_id"] != patient_id:
                return jsonify({"error": "unknown run"}), 404
            anchor = "historical" if not _is_live_head(service, session, run) else "live_head"
        else:
            run = service.repo.latest_completed_run(session.session_id, session.epoch)
            anchor = "live_head"
        if run is None:
            return jsonify({"patient_id": patient_id, "run": None, "anchor": anchor, "cells": [],
                            "runtime": service.runtime.provenance(), "causal_claim_allowed": False})
        return jsonify({
            "patient_id": patient_id, "anchor": anchor, "run": run,
            "cells": service.repo.cells_for_run(run["run_id"]),
            # this run's own organ state, from the same observations the historical
            # timeline is built from, so the figure and the charts cannot disagree
            "organs": organ_state_for_run(service.repo, run),
            "runtime": service.runtime.provenance(),
            "interval_policy": "lower/upper are null: no patient-specific conformal artifact in this release",
            "persistence_policy": "persistence is a baseline fallback, not a model prediction",
            "causal_claim_allowed": False,
            "clinical_alerts_supported": False,
        })

    @bp.get("/api/patients/<patient_id>/forecast-history")
    def patient_forecast_history(patient_id: str):
        session, _, error = _patient_context(patient_id)
        if error:
            return error
        runs = service.repo.runs_for_patient(patient_id)
        live = service.repo.latest_completed_run(session.session_id, session.epoch)
        cell = request.args.get("cell")
        return jsonify({
            "patient_id": patient_id, "live_run_id": live["run_id"] if live else None,
            "current_epoch": session.epoch, "runs": runs,
            "cell_history": service.repo.cell_history(patient_id, cell) if cell else None,
        })

    @bp.get("/api/patients/<patient_id>/mapping-failures")
    def patient_mapping_failures(patient_id: str):
        return jsonify({"patient_id": patient_id, "failures": service.repo.mapping_failures(patient_id)})

    @bp.get("/api/fhir/<resource_type>/<resource_id>/_history/<version_id>")
    def raw_resource(resource_type: str, resource_id: str, version_id: str):
        source = request.args.get("source")
        raw = None
        for system in ([source] if source else [make_source(k).source_system for k in SOURCES]):
            raw = service.repo.raw_by_reference(system, resource_type, resource_id, version_id)
            if raw:
                break
        if raw is None:
            return jsonify({"error": "not in ledger"}), 404
        return jsonify({"raw_id": raw["raw_id"], "first_received_at": raw["first_received_at"],
                        "validation_status": raw["validation_status"], "resource": raw["resource"],
                        "canonical": service.repo.canonical_for_raw(int(raw["raw_id"]))})

    # ---- model ------------------------------------------------------------------
    @bp.get("/api/model/capabilities")
    def model_capabilities():
        return jsonify(service.runtime.capabilities())

    @bp.get("/api/model/health")
    def model_health():
        health = service.runtime.health()
        return jsonify(health), (200 if health["ready"] else 503)

    @bp.get("/api/model/anatomy")
    def model_anatomy():
        return jsonify(anatomy_document(service.runtime.contract["catalog"]["cells"]))

    @bp.get("/api/model/mapping")
    def model_mapping():
        return jsonify(registry_document())

    @bp.get("/api/model/functions")
    def model_functions():
        contract = service.runtime.contract
        return jsonify(functions_document(contract["catalog"]["cells"], contract["modules"]))

    return bp


def _is_live_head(service: StreamService, session: StreamSession, run: dict) -> bool:
    live = service.repo.latest_completed_run(session.session_id, session.epoch)
    return bool(live and live["run_id"] == run["run_id"])


def _patient_display(patient: dict | None) -> str:
    if not patient:
        return "Synthetic patient"
    names = patient["resource"].get("name") or []
    return names[0].get("text") if names and names[0].get("text") else patient["patient_id"]


def _sse(service: StreamService, session: StreamSession, last_sequence: int):
    bus = service.sessions.bus
    q = bus.subscribe(session.session_id)
    try:
        yield f": oslian synthetic stream {session.session_id}\nretry: 2000\n\n"
        sent = last_sequence
        # ledger catch-up after a reconnect: the same events, delivered a second time
        for event in service.repo.events_after(session.session_id, last_sequence):
            sent = event["sequence"]
            bus.logger.sse_send(bus.context(session.session_id), event, DELIVERY_REPLAY)
            yield _format(event)
        last_beat = time.monotonic()
        while True:
            try:
                event = q.get(timeout=1.0)
            except queue.Empty:
                if time.monotonic() - last_beat >= 15.0:
                    last_beat = time.monotonic()
                    yield ": keep-alive\n\n"          # transport padding: never logged
                continue
            if event["sequence"] <= sent:
                continue
            sent = event["sequence"]
            bus.logger.sse_send(bus.context(session.session_id), event, DELIVERY_LIVE)
            yield _format(event)
    finally:
        bus.unsubscribe(session.session_id, q)


def _format(event: dict[str, Any]) -> str:
    data = json.dumps({"event_id": event["event_id"], "emitted_at": event["emitted_at"], **event["payload"]})
    return f"id: {event['sequence']}\nevent: {event['event_type']}\ndata: {data}\n\n"
