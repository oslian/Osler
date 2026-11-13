"""Real-time console for everything the stream backend produces and sends.

Two stages are recorded, and only two:

``event_emit``
    one line every time :meth:`EventBus.emit` publishes an event.
``sse_send``
    one line every time the SSE generator hands a formatted event to the client,
    tagged ``delivery_mode=live`` for a fresh event and ``delivery_mode=replay``
    for one served from the ledger after a reconnect.

Keep-alive comments are transport padding, not events, and are never logged.

Redaction is structural, not a filter: each event type has an explicit field
extractor and nothing outside it is ever printed. The full FHIR resource is
included only when the payload level is ``full`` *and* the session's source is a
synthetic fixture; any other source is forced to a redacted summary regardless of
configuration, so tokens, authorization headers and real identifiers have no path
to the console.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from typing import Any, Callable, TextIO

from .source import RETROSPECTIVE_SOURCE_SYSTEM, SYNTHETIC_SOURCE_SYSTEM

LOG_EVENTS, LOG_OFF = "events", "off"
FORMAT_PRETTY, FORMAT_JSON = "pretty", "json"
PAYLOAD_SUMMARY, PAYLOAD_FULL = "summary", "full"
STAGE_EMIT, STAGE_SEND = "event_emit", "sse_send"
DELIVERY_LIVE, DELIVERY_REPLAY = "live", "replay"
PREFIX = "[OSLIAN_STREAM]"

_LOG_MODES = {LOG_EVENTS, LOG_OFF}
_FORMATS = {FORMAT_PRETTY, FORMAT_JSON}
_PAYLOADS = {PAYLOAD_SUMMARY, PAYLOAD_FULL}
# Only these sources may ever print a full payload; everything else is redacted.
SYNTHETIC_SOURCES = {SYNTHETIC_SOURCE_SYSTEM, RETROSPECTIVE_SOURCE_SYSTEM}


def _num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _ids(values: Any) -> str:
    return ",".join(values) if values else "none"


# The console is an observer, never a participant. Records are handed to a
# bounded queue and written by one background thread, so a terminal that stops
# reading - a PowerShell window in selection mode is the usual way - can never
# stall ingestion, the SSE generator or an HTTP response. The bound is what makes
# that true: an unbounded queue would simply move the stall into memory.
LOG_QUEUE_MAXSIZE = 4096


class StreamEventLogger:
    """Single writer for both stages. Never raises into, and never blocks, the stream."""

    def __init__(self, mode: str = LOG_EVENTS, log_format: str = FORMAT_PRETTY,
                 payload_level: str = PAYLOAD_SUMMARY, stream: TextIO | None = None) -> None:
        if mode not in _LOG_MODES:
            raise ValueError(f"OSLIAN_STREAM_LOG must be one of {sorted(_LOG_MODES)}")
        if log_format not in _FORMATS:
            raise ValueError(f"OSLIAN_STREAM_LOG_FORMAT must be one of {sorted(_FORMATS)}")
        if payload_level not in _PAYLOADS:
            raise ValueError(f"OSLIAN_STREAM_LOG_PAYLOAD must be one of {sorted(_PAYLOADS)}")
        self.mode = mode
        self.format = log_format
        self.payload_level = payload_level
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()
        # One lock over the whole accounting, so accepted / written / dropped are
        # always a consistent triple. Every accepted line is eventually settled -
        # written to the sink, or evicted and counted - and flush() waits on
        # exactly that, rather than on the queue looking momentarily empty.
        self._count_lock = threading.Lock()
        self._accepted = 0
        self._settled = 0
        self.dropped = 0
        self._unreported_drops = 0
        self._closing = False
        self._queue: queue.Queue[str | None] | None = None
        self._writer: threading.Thread | None = None
        if self.mode == LOG_EVENTS:
            self._queue = queue.Queue(maxsize=LOG_QUEUE_MAXSIZE)
            self._writer = threading.Thread(target=self._drain, name="oslian-log", daemon=True)
            self._writer.start()

    @classmethod
    def from_env(cls, env: dict[str, str], stream: TextIO | None = None) -> "StreamEventLogger":
        return cls(
            env.get("OSLIAN_STREAM_LOG", LOG_EVENTS).strip().lower(),
            env.get("OSLIAN_STREAM_LOG_FORMAT", FORMAT_PRETTY).strip().lower(),
            env.get("OSLIAN_STREAM_LOG_PAYLOAD", PAYLOAD_SUMMARY).strip().lower(),
            stream,
        )

    @property
    def enabled(self) -> bool:
        return self.mode == LOG_EVENTS

    # ---- stages ---------------------------------------------------------------
    def event_emit(self, context: dict[str, Any], event: dict[str, Any]) -> None:
        self._write(STAGE_EMIT, context, event, delivery_mode=None)

    def sse_send(self, context: dict[str, Any], event: dict[str, Any], delivery_mode: str) -> None:
        self._write(STAGE_SEND, context, event, delivery_mode=delivery_mode)

    # ---- record building ------------------------------------------------------
    def _write(self, stage: str, context: dict[str, Any], event: dict[str, Any],
               delivery_mode: str | None) -> None:
        if not self.enabled:
            return
        try:
            record = self._record(stage, context, event, delivery_mode)
            line = json.dumps(record, sort_keys=True) if self.format == FORMAT_JSON else self._pretty(record)
            self._offer(line)
        except Exception:                                  # noqa: BLE001 - logging never breaks the stream
            pass

    # ---- non-blocking hand-off ------------------------------------------------
    def _offer(self, line: str) -> None:
        """Queue a line without ever waiting for the terminal.

        When the console cannot keep up the oldest lines are discarded and
        counted, so the newest state is what survives and the caller returns at
        once. Dropping is announced on the next line that gets through, because
        a console that silently loses records is worse than one that says so.
        """
        q = self._queue
        if q is None:
            return
        with self._count_lock:
            if self._closing:
                # close() has already queued the sentinel. A line accepted now
                # would sit behind it and never be written, so it is refused
                # here and counted as a drop rather than lost in silence.
                self.dropped += 1
                return
            self._accepted += 1
        while True:
            try:
                q.put_nowait(line)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                    with self._count_lock:
                        self.dropped += 1
                        self._unreported_drops += 1
                        self._settled += 1     # evicted counts as settled
                except queue.Empty:                        # drained underneath us; retry
                    pass

    def _drain(self) -> None:
        q = self._queue
        assert q is not None
        while True:
            line = q.get()
            if line is None:
                return
            batch = [line]
            while len(batch) < 256:                        # one write per burst, not per event
                try:
                    nxt = q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    batch.append(None)                     # keep the sentinel; handled below
                    break
                batch.append(nxt)
            stopping = batch and batch[-1] is None
            if stopping:
                batch.pop()
            with self._count_lock:
                dropped, self._unreported_drops = self._unreported_drops, 0
            if dropped:
                batch.insert(0, f"{PREFIX} console_dropped lines={dropped} "
                                f"reason=terminal slower than the stream")
            try:
                with self._lock:
                    self._stream.write("".join(f"{item}\n" for item in batch))
                    self._stream.flush()
            except Exception:                              # noqa: BLE001
                pass
            finally:
                # settle every line taken from the queue, written or not: a sink
                # that raises must never leave flush() waiting for it
                with self._count_lock:
                    self._settled += len(batch) - (1 if dropped else 0)
            if stopping:
                return

    def flush(self, timeout: float = 2.0) -> bool:
        """Block until everything queued so far has reached the sink.

        The console is asynchronous by design, so anything that needs to *read*
        what was logged - a test, or an orderly shutdown - has to say when it
        wants to wait. Nothing on the stream path calls this.
        """
        if self._queue is None:
            return True
        with self._count_lock:
            target = self._accepted
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._count_lock:
                if self._settled >= target:
                    return True
            time.sleep(0.002)
        with self._count_lock:
            return self._settled >= target

    def close(self, timeout: float = 2.0) -> None:
        """Flush what is queued and stop the writer. Safe to call more than once."""
        q, writer = self._queue, self._writer
        if q is None or writer is None:
            return
        with self._count_lock:
            already_closing = self._closing
            self._closing = True
        if already_closing:
            # a concurrent close already queued the sentinel; wait for it rather
            # than queue a second one
            writer.join(timeout=timeout)
            return
        try:
            q.put_nowait(None)
        except queue.Full:                                 # make room for the sentinel
            try:
                q.get_nowait()
                with self._count_lock:
                    self.dropped += 1
                    self._unreported_drops += 1
                    self._settled += 1
                q.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        writer.join(timeout=timeout)
        self._writer = None

    def _record(self, stage: str, context: dict[str, Any], event: dict[str, Any],
                delivery_mode: str | None) -> dict[str, Any]:
        source = context.get("source") or "unknown"
        payload = event.get("payload") or {}
        record: dict[str, Any] = {
            "timestamp": event.get("emitted_at"),
            "stage": stage,
            "session": (context.get("session_id") or "")[:8],
            "epoch": context.get("epoch"),
            "seq": event.get("sequence"),
            "event_id": event.get("event_id"),
            "type": event.get("event_type"),
            "source": source,
        }
        if delivery_mode is not None:
            record["mode"] = delivery_mode
        record.update(_extract(event.get("event_type") or "", payload))
        if self.payload_level == PAYLOAD_FULL and source in SYNTHETIC_SOURCES:
            record["payload"] = payload
        return record

    def _pretty(self, record: dict[str, Any]) -> str:
        head = f"{record.get('timestamp')} {PREFIX} {record.get('stage')}"
        skip = {"timestamp", "stage", "payload"}
        body = " ".join(f"{k}={_render(v)}" for k, v in record.items() if k not in skip and v is not None)
        line = f"{head} {body}"
        if "payload" in record:
            line += " payload=" + json.dumps(record["payload"], sort_keys=True)
        return line


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return _ids(value)
    return str(value)


# ---- per-event-type field extractors ------------------------------------------
def _resource_fields(payload: dict[str, Any]) -> dict[str, Any]:
    resource = payload.get("resource") or {}
    coding = ((resource.get("code") or {}).get("coding") or [{}])[0]
    return {
        "resource": f"{resource.get('resourceType', '?')}/{resource.get('id', '?')}"
                    f" v{(resource.get('meta') or {}).get('versionId', '?')}",
        "code": coding.get("code"),
        "effective": payload.get("effective_time"),
        "received": payload.get("received_at"),
    }


def _accepted(payload: dict[str, Any]) -> dict[str, Any]:
    return {**_resource_fields(payload),
            "duplicate": bool(payload.get("ledger_duplicate") or payload.get("delivery_duplicate"))}


def _rejected(payload: dict[str, Any]) -> dict[str, Any]:
    issues = payload.get("validation_issues") or []
    return {**_resource_fields(payload), "rejected": True,
            "reason": ";".join(str(i) for i in issues) or "invalid"}


def _unmapped(payload: dict[str, Any]) -> dict[str, Any]:
    return {"resource": f"Observation/{payload.get('resource_id', '?')} v{payload.get('version_id', '?')}",
            "unmapped": True, "reason": payload.get("reason"), "detail": payload.get("detail"),
            "effective": payload.get("effective_time")}


def _normalized(payload: dict[str, Any]) -> dict[str, Any]:
    values = payload.get("values") or []
    first = values[0] if values else {}
    fields: dict[str, Any] = {
        "resource": f"Observation/{payload.get('resource_id', '?')} v{payload.get('version_id', '?')}",
        "variable": "|".join(str(v.get("variable")) for v in values) or "none",
        "value": "|".join(_num(v.get("value")) for v in values) or "—",
        "unit": first.get("unit"),
        "effective": payload.get("effective_time"),
        "received": payload.get("received_at"),
        "organs": _ids(payload.get("organ_ids")),
        "functions": _ids(payload.get("functional_group_ids")),
    }
    if payload.get("superseding"):
        fields["superseding"] = True
    return fields


def _run_started(payload: dict[str, Any]) -> dict[str, Any]:
    return {"run": payload.get("run_id"), "cutoff": payload.get("input_cutoff"),
            "resources": payload.get("input_resource_count")}


def _run_completed(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    return {"run": payload.get("run_id"), "cutoff": payload.get("input_cutoff"),
            "runtime": payload.get("runtime_mode"),
            "model": summary.get("model_cells"), "persistence": summary.get("persistence_cells"),
            "missing": summary.get("missing_cells"), "degraded": summary.get("degraded_cells"),
            "duration_ms": payload.get("duration_ms")}


def _run_failed(payload: dict[str, Any]) -> dict[str, Any]:
    return {"run": payload.get("run_id"), "failed": True, "reason": payload.get("error")}


def _state(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": payload.get("status"), "speed": payload.get("speed"),
            "cursor": payload.get("cursor"), "remaining": payload.get("remaining_events"),
            "reason": payload.get("reason")}


_EXTRACTORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "fhir.resource.accepted": _accepted,
    "fhir.resource.rejected": _rejected,
    "observation.unmapped": _unmapped,
    "observation.normalized": _normalized,
    "model.run.started": _run_started,
    "model.run.completed": _run_completed,
    "model.run.failed": _run_failed,
    "stream.state": _state,
}


def _extract(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    extractor = _EXTRACTORS.get(event_type)
    if extractor is None:
        return {}
    return {k: v for k, v in extractor(payload).items() if v is not None}
