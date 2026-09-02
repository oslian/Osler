"""Stream playback sessions, the event bus, and the coalesced run scheduler.

* Speed changes only the simulated arrival clock; clinical timestamps inside
  the FHIR resources are never touched.
* ``received_at`` is real wall-clock time of delivery.
* A reset increments the session epoch and rewinds the playback cursor. Ledger
  rows, deliveries, events and model runs from earlier epochs are retained.
* Model runs are scheduled only when a new canonical observation lands, and
  bursts are coalesced by a short wall-clock debounce; one batch, one run.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any
import uuid

from .fhir import now_iso, now_utc, iso
from .ingest import Ingestor
from .eventlog import LOG_OFF, StreamEventLogger
from .repository import Repository
from .routing import variable_routing
from .source import ClinicalDataSource, make_source
from .timeline import build_model_input, delivered_timeline, variable_state
from .wholebody import RuntimeUnavailable, WholeBodyRuntime

SPEEDS = (1.0, 5.0, 20.0)
STATUS_IDLE, STATUS_RUNNING, STATUS_PAUSED, STATUS_COMPLETED = "idle", "running", "paused", "completed"
RUN_DEBOUNCE_SECONDS = 0.4


class EventBus:
    def __init__(self, repo: Repository, logger: StreamEventLogger | None = None):
        self.repo = repo
        self.logger = logger or StreamEventLogger(mode=LOG_OFF)
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[queue.Queue]] = {}
        self._contexts: dict[str, dict[str, Any]] = {}

    def set_context(self, session_id: str, *, epoch: int, source: str) -> None:
        """Session-scoped fields the console prints on every line."""
        with self._lock:
            self._contexts[session_id] = {"session_id": session_id, "epoch": epoch, "source": source}

    def context(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._contexts.get(session_id) or {"session_id": session_id})

    def subscribe(self, session_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(session_id, []).append(q)
        return q

    def unsubscribe(self, session_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(session_id, [])
            if q in subs:
                subs.remove(q)

    def emit(self, session_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        emitted_at = now_iso()
        event_id = uuid.uuid4().hex
        sequence = self.repo.append_event(session_id, event_id, event_type, payload, emitted_at)
        event = {"sequence": sequence, "event_id": event_id, "event_type": event_type,
                 "payload": payload, "emitted_at": emitted_at}
        self.logger.event_emit(self.context(session_id), event)
        with self._lock:
            subscribers = list(self._subscribers.get(session_id, []))
        for q in subscribers:
            q.put(event)
        return event


@dataclass
class _Clock:
    base_elapsed: float = 0.0
    running_since: float | None = None   # time.monotonic()
    speed: float = 1.0

    def elapsed(self) -> float:
        if self.running_since is None:
            return self.base_elapsed
        return self.base_elapsed + (time.monotonic() - self.running_since) * self.speed

    def pause(self) -> None:
        self.base_elapsed = self.elapsed()
        self.running_since = None

    def resume(self) -> None:
        if self.running_since is None:
            self.running_since = time.monotonic()

    def set_speed(self, speed: float) -> None:
        running = self.running_since is not None
        self.pause()
        self.speed = speed
        if running:
            self.resume()


@dataclass
class _PendingRun:
    """The single waiting run, in constant space.

    Only one run is ever waiting, so the triggers that arrive while a run is in
    flight are a stream of ids with no bound. Keeping the list would let a long
    inference under a 20x feed grow memory in proportion to the feed, for data
    that answers one question: what set this run off. First, latest and count
    answer it. The audit record of what a run actually consumed is
    ``input_refs``, written from the delivered timeline when the run starts, and
    that is unaffected.
    """

    first: str | None = None
    latest: str | None = None
    count: int = 0

    def add(self, trigger_event_id: str) -> None:
        if self.first is None:
            self.first = trigger_event_id
        self.latest = trigger_event_id
        self.count += 1

    def take(self) -> dict[str, Any]:
        summary = {"first": self.first, "latest": self.latest, "count": self.count}
        self.clear()
        return summary

    def clear(self) -> None:
        self.first = None
        self.latest = None
        self.count = 0

    def __bool__(self) -> bool:
        return self.count > 0


class RunScheduler:
    """One inference at a time, one pending run, and never a stale prefix.

    The invariant is deliberately small: at most one run executing, and at most
    one run waiting. Triggers that arrive while a run is in flight are merged
    into that single waiting run rather than queued behind each other, so a 20x
    burst cannot build a backlog of forecasts that are obsolete before they are
    computed. The waiting run reads the delivered timeline when it *starts*, not
    when it was requested, so whatever it finally computes describes the newest
    delivered prefix. Nothing is dropped from the ledger by this: only redundant
    inference requests are merged.

    Consecutive runs are separated by ``debounce`` measured from the end of the
    previous run, which bounds how much of the worker a sustained burst can take
    while still guaranteeing a final run once the burst stops.

    Shutdown is explicit: after :meth:`stop` no further request is accepted, a
    run that has not started is cancelled, a run already executing is allowed to
    finish so its cells reach the ledger, and the thread ends.
    """

    def __init__(self, session: "StreamSession", debounce_seconds: float = RUN_DEBOUNCE_SECONDS):
        self.session = session
        self.debounce = debounce_seconds
        self._cv = threading.Condition()
        self._pending = _PendingRun()
        self._deadline: float | None = None
        self._stop = False
        self._busy = False
        self._thread = threading.Thread(target=self._loop, name=f"runs:{session.session_id}", daemon=True)
        self._thread.start()
        self.runs_started = 0
        self.runs_superseded = 0
        self.runs_cancelled = 0
        self.max_in_flight = 0
        self.max_pending_runs = 0

    def request(self, trigger_event_id: str) -> None:
        with self._cv:
            if self._stop:
                # closing: a new run must not be started, and the caller is not
                # left believing one was scheduled
                return
            self._pending.add(trigger_event_id)
            if self._deadline is None:
                self._deadline = time.monotonic() + self.debounce
            # pending *runs*, not pending triggers: the triggers all merge into one
            self.max_pending_runs = max(self.max_pending_runs, 1)
            self._cv.notify_all()

    def stats(self) -> dict[str, int]:
        with self._cv:
            return {"runs_started": self.runs_started, "runs_superseded": self.runs_superseded,
                    "runs_cancelled": self.runs_cancelled,
                    "in_flight": 1 if self._busy else 0,
                    "pending_runs": 1 if self._pending else 0,
                    "pending_triggers": self._pending.count,
                    "max_in_flight": self.max_in_flight,
                    "max_pending_runs": self.max_pending_runs,
                    "stopped": 1 if self._stop else 0,
                    "thread_alive": 1 if self._thread.is_alive() else 0}

    def idle(self) -> bool:
        with self._cv:
            return not self._pending and not self._busy

    def wait_idle(self, timeout: float = 30.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.idle():
                return True
            time.sleep(0.02)
        return False

    def stop(self, timeout: float = 15.0) -> bool:
        """Stop scheduling, cancel what has not started, and end the thread.

        A run already executing keeps going: its cells belong in the ledger and
        the repository must outlive it. A run that is only pending is cancelled
        outright - starting an inference during shutdown would delay exit for a
        forecast nobody will read. Idempotent, and safe to call from any thread
        except the scheduler's own.

        Returns True when the thread has ended.
        """
        with self._cv:
            first_call = not self._stop
            self._stop = True
            if self._pending:
                self.runs_cancelled += 1
                self._pending.clear()
            self._deadline = None
            self._cv.notify_all()
        thread = self._thread
        if first_call or thread.is_alive():
            if thread is not threading.current_thread():
                thread.join(timeout)
        return not thread.is_alive()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._stop and (not self._pending or time.monotonic() < (self._deadline or 0)):
                    timeout = None if not self._pending else max(0.0, (self._deadline or 0) - time.monotonic())
                    self._cv.wait(timeout)
                if self._stop:
                    return
                triggers = self._pending.take()
                self._deadline = None
                self._busy = True
                self.runs_started += 1
                self.max_in_flight = max(self.max_in_flight, 1)
            try:
                self.session.execute_run(triggers)
            finally:
                with self._cv:
                    self._busy = False
                    # space the next run from the end of this one, so a sustained
                    # burst cannot keep the worker permanently saturated
                    if self._pending and not self._stop:
                        self._deadline = time.monotonic() + self.debounce
                    self._cv.notify_all()


class StreamSession:
    def __init__(self, repo: Repository, bus: EventBus, source: ClinicalDataSource, runtime: WholeBodyRuntime,
                 record: dict[str, Any]):
        self.repo = repo
        self.bus = bus
        self.source = source
        self.runtime = runtime
        self.session_id = str(record["session_id"])
        self.patient_id = str(record["patient_id"])
        self.epoch = int(record["epoch"])
        self.cursor = int(record["cursor"])
        self.status = str(record["status"])
        self.clock = _Clock(base_elapsed=float(record["sim_elapsed_seconds"]), speed=float(record["speed"]))
        self.events = source.events()
        self.ingestor = Ingestor(repo, source.source_system)
        self.bus.set_context(self.session_id, epoch=self.epoch, source=source.source_system)
        self._lock = threading.RLock()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._last_run_input_hash: str | None = None
        self.scheduler = RunScheduler(self)
        encounter = repo.encounter_for_patient(self.patient_id)
        self.onset = encounter["period_start"] if encounter else None

    # ---- state ----------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        with self._lock:
            elapsed = self.clock.elapsed()
            next_event = self.events[self.cursor] if self.cursor < len(self.events) else None
            record = self.repo.session(self.session_id) or {}
            return {
                "last_event_sequence": int(record.get("last_event_sequence", 0)),
                "source": self.source.describe(),
                "session_id": self.session_id,
                "patient_id": self.patient_id,
                "source_system": self.source.source_system,
                "fixture_version": self.source.fixture_version,
                "status": self.status,
                "speed": self.clock.speed,
                "speeds": list(SPEEDS),
                "epoch": self.epoch,
                "cursor": self.cursor,
                "total_events": len(self.events),
                "remaining_events": len(self.events) - self.cursor,
                "sim_elapsed_seconds": round(elapsed, 3),
                "next_arrival_offset_seconds": next_event.arrival_offset_seconds if next_event else None,
                "synthetic": True,
            }

    def _persist(self, **extra: Any) -> None:
        self.repo.update_session(
            self.session_id, status=self.status, speed=self.clock.speed, epoch=self.epoch, cursor=self.cursor,
            sim_elapsed_seconds=self.clock.elapsed(), updated_at=now_iso(), **extra)

    def _emit_state(self, reason: str) -> None:
        self.bus.emit(self.session_id, "stream.state", {**self.state(), "reason": reason})

    # ---- controls (idempotent) -------------------------------------------------
    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.status == STATUS_RUNNING:
                return self.state()
            if self.status == STATUS_COMPLETED:
                return self.state()
            self.status = STATUS_RUNNING
            self.clock.resume()
            self._persist()
            self._ensure_thread()
        self._emit_state("start")
        return self.state()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            if self.status != STATUS_RUNNING:
                return self.state()
            self.clock.pause()
            self.status = STATUS_PAUSED
            self._persist()
        self._emit_state("pause")
        return self.state()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            if self.cursor == 0 and self.clock.elapsed() == 0.0 and self.status in {STATUS_IDLE, STATUS_PAUSED}:
                return self.state()
            self.clock.pause()
            self.clock.base_elapsed = 0.0
            self.epoch += 1
            self.cursor = 0
            self.status = STATUS_PAUSED
            self.bus.set_context(self.session_id, epoch=self.epoch, source=self.source.source_system)
            self._persist()
        self._emit_state("reset")
        return self.state()

    def set_speed(self, speed: float) -> dict[str, Any]:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        with self._lock:
            self.clock.set_speed(speed)
            self._persist()
        self._emit_state("speed")
        return self.state()

    def step(self) -> dict[str, Any]:
        """Deliver the next event now (works while idle/paused); playback resumes from its arrival time."""
        with self._lock:
            if self.cursor >= len(self.events):
                return self.state()
            first = self.events[self.cursor]
            # one step = one arrival group: every event sharing the next arrival offset
            # (a retrospective row is many resources; a synthetic event is one)
            group = [e for e in self.events[self.cursor:] if e.arrival_offset_seconds == first.arrival_offset_seconds]
            if self.status == STATUS_RUNNING:
                self.clock.pause()
            self.status = STATUS_PAUSED
            self.clock.base_elapsed = max(self.clock.base_elapsed, first.arrival_offset_seconds)
        for event in group:
            self._deliver(event)
        self._emit_state("step")
        return self.state()

    def stop(self) -> None:
        """Stop playback and the scheduler. Returns once the scheduler thread has ended."""
        self._stop = True
        self.scheduler.stop()

    # ---- playback thread ---------------------------------------------------------
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name=f"stream:{self.session_id}", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                if self.status != STATUS_RUNNING:
                    running = False
                else:
                    running = True
                    if self.cursor >= len(self.events):
                        self.status = STATUS_COMPLETED
                        self.clock.pause()
                        self._persist()
                        self._emit_state("completed")
                        continue
                    event = self.events[self.cursor]
                    due = self.clock.elapsed() >= event.arrival_offset_seconds
                    wait = 0.0 if due else min(0.05, (event.arrival_offset_seconds - self.clock.elapsed()) / max(self.clock.speed, 1e-6))
            if not running:
                time.sleep(0.05)
                continue
            if not due:
                time.sleep(max(wait, 0.005))
                continue
            self._deliver(event)

    def _deliver(self, event) -> None:
        received_at = now_iso()
        with self._lock:
            epoch = self.epoch
            event_id = f"{self.session_id}:{epoch}:{event.source_sequence}"
            result = self.ingestor.ingest_resource(event.resource, received_at=received_at, event_id=event_id)
            delivered_new = False
            if result.accepted and result.raw_id is not None:
                delivered_new = self.repo.record_delivery(self.session_id, epoch, result.raw_id, event.source_sequence,
                                                          event_id, received_at)
            self.cursor += 1
            self.repo.set_ingestion_cursor(self.session_id, epoch, self.source.source_system, event.source_sequence,
                                           received_at)
            self._persist()
        envelope = {
            "event_id": event_id,
            "source": self.source.source_system,
            "sequence": event.source_sequence,
            "received_at": received_at,
            "resource": event.resource,
        }
        if not result.accepted:
            self.bus.emit(self.session_id, "fhir.resource.rejected", {
                **envelope, "validation_issues": list(result.validation_issues)})
            return
        self.bus.emit(self.session_id, "fhir.resource.accepted", {
            **envelope,
            "raw_id": result.raw_id,
            "ledger_duplicate": result.ledger_duplicate,
            "delivery_duplicate": not delivered_new,
            "effective_time": result.effective_time,
            "scenario": event.scenario,
        })
        if not delivered_new:
            return
        if result.mapping_failure:
            self.bus.emit(self.session_id, "observation.unmapped", {
                "event_id": event_id, "resource_id": result.resource_id, "version_id": result.version_id,
                "reason": result.mapping_failure, "detail": result.mapping_detail,
                "effective_time": result.effective_time})
            return
        if result.resource_type != "Observation":
            return
        if not result.canonical:
            # e.g. entered-in-error version: the timeline drops the resource; re-run on the new prefix
            self.bus.emit(self.session_id, "observation.normalized", {
                "event_id": event_id, "resource_id": result.resource_id, "version_id": result.version_id,
                "effective_time": result.effective_time, "values": [], "superseding": True,
                "organ_ids": [], "functional_group_ids": []})
            self.scheduler.request(event_id)
            return
        routing = variable_routing()
        values = [{**row, "organ_ids": list(routing.organs_for(row["variable"])),
                   "functional_group_ids": list(routing.functions_for(row["variable"]))}
                  for row in result.canonical]
        self.bus.emit(self.session_id, "observation.normalized", {
            "event_id": event_id, "resource_id": result.resource_id, "version_id": result.version_id,
            "effective_time": result.effective_time, "received_at": received_at,
            "values": values, "superseding": result.version_id != "1",
            # union across this resource's values: what the motion renderer animates
            "organ_ids": sorted({o for v in values for o in v["organ_ids"]}),
            "functional_group_ids": sorted({f for v in values for f in v["functional_group_ids"]})})
        self.scheduler.request(event_id)

    # ---- model runs -----------------------------------------------------------------
    def execute_run(self, trigger_summary: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            epoch = self.epoch
            timeline_all = delivered_timeline(self.repo, self.session_id, epoch)
        if not timeline_all or not self.onset:
            return None
        cutoff = max(row["effective_time"] for row in timeline_all)
        timeline = [row for row in timeline_all if row["effective_time"] <= cutoff]
        model_input = build_model_input(timeline, cutoff, self.onset)
        # The delivered prefix is unchanged since the last completed run, so the
        # forecast would be identical. Skipping keeps the worker for prefixes that
        # actually moved; the newest completed run still describes the newest
        # delivered prefix, which is the property that matters.
        if model_input.input_hash == self._last_run_input_hash:
            self.scheduler.runs_superseded += 1
            return None
        state = variable_state(timeline, cutoff)
        run_id = uuid.uuid4().hex
        started_at = now_utc()
        provenance = self.runtime.provenance()
        record = {
            "run_id": run_id, "session_id": self.session_id, "epoch": epoch, "patient_id": self.patient_id,
            "status": "running", "runtime_mode": provenance["runtime_mode"],
            "model_commit": provenance["model_commit"], "registry_sha256": provenance["registry_sha256"],
            "release_date": provenance["release_date"], "input_cutoff": cutoff,
            "input_watermark": model_input.watermark, "input_resource_count": len(model_input.resource_refs),
            "input_hash": model_input.input_hash, "input_refs_json": __import__("json").dumps(list(model_input.resource_refs)),
            "trigger_event_ids_json": __import__("json").dumps(trigger_summary), "started_at": iso(started_at),
        }
        self.repo.create_run(record)
        self.bus.emit(self.session_id, "model.run.started", {
            "run_id": run_id, "epoch": epoch, "input_cutoff": cutoff, "input_watermark": model_input.watermark,
            "input_resource_count": len(model_input.resource_refs), "input_hash": model_input.input_hash,
            "triggers": trigger_summary, "runtime_mode": provenance["runtime_mode"],
            "model_commit": provenance["model_commit"], "started_at": record["started_at"]})
        try:
            result = self.runtime.run(model_input, state)
        except RuntimeUnavailable as exc:
            completed = now_utc()
            self.repo.finish_run(run_id, status="failed", completed_at=iso(completed),
                                 duration_ms=int((completed - started_at).total_seconds() * 1000),
                                 error=str(exc), summary=None)
            self.bus.emit(self.session_id, "model.run.failed", {
                "run_id": run_id, "epoch": epoch, "input_cutoff": cutoff, "error": str(exc),
                "runtime_mode": provenance["runtime_mode"], "completed_at": iso(completed)})
            return None
        except Exception as exc:  # noqa: BLE001 - never leave a run dangling
            completed = now_utc()
            message = f"{type(exc).__name__}: {exc}"
            self.repo.finish_run(run_id, status="failed", completed_at=iso(completed),
                                 duration_ms=int((completed - started_at).total_seconds() * 1000),
                                 error=message, summary=None)
            self.bus.emit(self.session_id, "model.run.failed", {
                "run_id": run_id, "epoch": epoch, "input_cutoff": cutoff, "error": message,
                "runtime_mode": provenance["runtime_mode"], "completed_at": iso(completed)})
            return None
        completed = now_utc()
        self.repo.insert_cells(run_id, result.cells)
        self.repo.finish_run(run_id, status="completed", completed_at=iso(completed),
                             duration_ms=int((completed - started_at).total_seconds() * 1000),
                             error=None, summary=result.summary)
        payload = {
            "run_id": run_id, "epoch": epoch, "input_cutoff": cutoff, "input_watermark": model_input.watermark,
            "input_resource_count": len(model_input.resource_refs), "input_hash": model_input.input_hash,
            "summary": result.summary, "runtime_mode": provenance["runtime_mode"],
            "runtime_attached": provenance["runtime_attached"], "model_commit": provenance["model_commit"],
            "registry_sha256": provenance["registry_sha256"], "duration_ms": result.duration_ms,
            "completed_at": iso(completed), "cells": result.cells,
        }
        self._last_run_input_hash = model_input.input_hash
        self.bus.emit(self.session_id, "model.run.completed", payload)
        return payload


class SessionManager:
    def __init__(self, repo: Repository, runtime: WholeBodyRuntime, source_factory=make_source,
                 logger: StreamEventLogger | None = None):
        self.repo = repo
        self.runtime = runtime
        self.bus = EventBus(repo, logger)
        self._source_factory = source_factory
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()
        for record in repo.sessions():
            if record["status"] == STATUS_RUNNING:
                # the playback thread did not survive a restart; resume is explicit
                repo.update_session(record["session_id"], status=STATUS_PAUSED, updated_at=now_iso())
                record["status"] = STATUS_PAUSED
            source = self._source_for_system(record["source_system"])
            if source is None:
                continue   # a session whose source no longer exists stays in the ledger but is not resurrected
            self._sessions[record["session_id"]] = StreamSession(repo, self.bus, source, runtime, record)

    def _source_for_system(self, source_system: str):
        for key in ("synthetic", "retrospective"):
            try:
                source = self._source_factory(key)
            except Exception:  # noqa: BLE001
                continue
            if source.source_system == source_system:
                return source
        return None

    def get(self, session_id: str) -> StreamSession | None:
        return self._sessions.get(session_id)

    def all(self) -> list[StreamSession]:
        return list(self._sessions.values())

    def get_or_create(self, source_key: str = "synthetic") -> tuple[StreamSession, bool]:
        """One session per (source, patient): POST /api/demo-streams is idempotent."""
        source = self._source_factory(source_key)
        with self._lock:
            existing = self.repo.session_for_patient(source.patient_id, source.source_system)
            if existing and existing["session_id"] in self._sessions:
                return self._sessions[existing["session_id"]], False
            ingestor = Ingestor(self.repo, source.source_system)
            bootstrap_id = f"bootstrap:{source.patient_id}"
            ingestor.ingest_bundle(source.initial_bundle(), received_at=now_iso(), event_id=bootstrap_id)
            session_id = uuid.uuid4().hex[:12]
            record = {
                "session_id": session_id, "patient_id": source.patient_id, "source_system": source.source_system,
                "fixture_version": source.fixture_version, "status": STATUS_IDLE, "speed": 1.0, "epoch": 1,
                "cursor": 0, "sim_elapsed_seconds": 0.0, "last_event_sequence": 0, "created_at": now_iso(),
                "updated_at": now_iso(),
            }
            self.repo.create_session(record)
            session = StreamSession(self.repo, self.bus, source, self.runtime, record)
            self._sessions[session_id] = session
            # snapshot bundle observations count as epoch-1 deliveries at sequence 0
            self._deliver_snapshot(session, source)
        session._emit_state("created")
        return session, True

    def _deliver_snapshot(self, session: StreamSession, source) -> None:
        received_at = now_iso()
        for resource in source.initial_bundle().get("entry", []):
            res = resource.get("resource", {})
            if res.get("resourceType") != "Observation":
                continue
            raw = self.repo.raw_by_reference(source.source_system, "Observation", res["id"], res["meta"]["versionId"])
            if raw:
                self.repo.record_delivery(session.session_id, session.epoch, int(raw["raw_id"]), 0,
                                          f"bootstrap:{source.patient_id}", received_at)

    def deliver_snapshot_for_epoch(self, session: StreamSession) -> None:
        self._deliver_snapshot(session, session.source)

    def shutdown(self) -> None:
        # session.stop() cancels a pending run and joins the scheduler thread, so
        # by the time it returns any run that was already executing has finished
        # writing and the repository can close under it safely. There is nothing
        # left to wait for, which is why there is no timeout here.
        for session in self._sessions.values():
            session.stop()
        self.runtime.close()
