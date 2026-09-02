"""Streaming event console: exactly one line per emit, one per SSE delivery, and no secrets."""

from collections import Counter
import io
import json
import os
import re
import time
import unittest

os.environ.setdefault("OSLIAN_STREAM_DB", ":memory:")
os.environ.setdefault("OSLIAN_WHOLEBODY_RUNTIME", "fixture")

from flask import Flask  # noqa: E402

from oslian_demo.monitoring.api import StreamService, create_blueprint  # noqa: E402
from oslian_demo.monitoring.eventlog import (  # noqa: E402
    FORMAT_JSON, LOG_OFF, PAYLOAD_FULL, PREFIX, StreamEventLogger,
)
from oslian_demo.monitoring.source import EpicFhirSource, SYNTHETIC_SOURCE_SYSTEM  # noqa: E402
from oslian_demo.monitoring.wholebody import RuntimeConfig, WholeBodyRuntime  # noqa: E402


def _service(logger):
    service = StreamService(":memory:", runtime=WholeBodyRuntime(RuntimeConfig("fixture", None, "python", None)),
                            logger=logger)
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(service))
    return service, app.test_client()


def _lines(sink, stage=None):
    out = [l for l in sink.getvalue().splitlines() if l.strip()]
    return [l for l in out if stage is None or f" {stage} " in l]


def _field(line, key):
    match = re.search(rf"(?:^| ){re.escape(key)}=(\S+)", line)
    return match.group(1) if match else None


class ConsoleStageTests(unittest.TestCase):
    def setUp(self):
        self.sink = io.StringIO()
        self.service, self.client = _service(StreamEventLogger(stream=self.sink))

    def tearDown(self):
        self.service.close()

    def test_every_bus_event_produces_exactly_one_emit_line(self):
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        for _ in range(6):
            self.client.post(f"/api/demo-streams/{sid}/step")
        time.sleep(1.5)                                   # let the debounced model run land
        emitted = self.service.repo.events_after(sid, 0)
        lines = _lines(self.sink, "event_emit")
        self.assertEqual(len(lines), len(emitted))
        self.assertEqual([_field(l, "seq") for l in lines], [str(e["sequence"]) for e in emitted])
        self.assertEqual(Counter(_field(l, "event_id") for l in lines),
                         Counter(e["event_id"] for e in emitted))

    def test_sse_delivery_logs_one_send_line_per_event_and_marks_replay(self):
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        for _ in range(3):
            self.client.post(f"/api/demo-streams/{sid}/step")
        time.sleep(2.0)                                    # let the debounced run land before snapshotting
        delivered = self.service.repo.events_after(sid, 0)
        self.sink.truncate(0), self.sink.seek(0)
        response = self.client.get(f"/api/demo-streams/{sid}/events", headers={"Last-Event-ID": "0"},
                                   buffered=False)
        stream = response.response
        seen = 0
        for _ in stream:                                   # preamble, ledger catch-up, then it blocks
            seen += 1
            if seen > len(delivered):
                break
        stream.close()
        sends = _lines(self.sink, "sse_send")
        self.assertEqual(len(sends), len(delivered))
        self.assertTrue(all(_field(l, "mode") == "replay" for l in sends))
        self.assertEqual([_field(l, "seq") for l in sends], [str(e["sequence"]) for e in delivered])
        self.assertNotIn("keep-alive", self.sink.getvalue())

    def test_sequence_is_continuous_at_twenty_times(self):
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        self.client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
        self.client.post(f"/api/demo-streams/{sid}/start")
        time.sleep(4.0)
        self.client.post(f"/api/demo-streams/{sid}/pause")
        time.sleep(1.0)
        sequences = [int(_field(l, "seq")) for l in _lines(self.sink, "event_emit")]
        self.assertGreater(len(sequences), 20, "20x should have produced a substantial burst")
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(sequences, list(range(sequences[0], sequences[0] + len(sequences))),
                         "no console line may be silently dropped at 20x")

    def test_observation_lines_carry_mapping_and_server_authored_routing(self):
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        for _ in range(3):
            self.client.post(f"/api/demo-streams/{sid}/step")
        normalized = [l for l in _lines(self.sink) if "type=observation.normalized" in l]
        self.assertTrue(normalized)
        line = normalized[0]
        for key in ("session", "epoch", "seq", "event_id", "type", "source", "resource",
                    "variable", "value", "unit", "effective", "received", "organs", "functions"):
            self.assertIsNotNone(_field(line, key), f"{key} missing from: {line}")
        heart_rate = next(l for l in normalized if _field(l, "variable") == "heart_rate")
        self.assertEqual(_field(heart_rate, "organs"), "blood_vessels,heart")
        self.assertEqual(_field(heart_rate, "functions"), "cardiovascular_perfusion")

    def test_model_run_lines_report_cutoff_and_cell_summary(self):
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        for _ in range(4):
            self.client.post(f"/api/demo-streams/{sid}/step")
        time.sleep(2.0)
        completed = [l for l in _lines(self.sink) if "type=model.run.completed" in l]
        self.assertTrue(completed, "a debounced run should have completed")
        line = completed[-1]
        for key in ("run", "cutoff", "runtime", "model", "persistence", "missing", "degraded", "duration_ms"):
            self.assertIsNotNone(_field(line, key), f"{key} missing from: {line}")


class ConsoleConfigTests(unittest.TestCase):
    def test_defaults_and_env_parsing(self):
        default = StreamEventLogger.from_env({})
        self.assertEqual((default.mode, default.format, default.payload_level), ("events", "pretty", "summary"))
        self.assertTrue(default.enabled)
        off = StreamEventLogger.from_env({"OSLIAN_STREAM_LOG": "off"})
        self.assertFalse(off.enabled)
        for bad in ({"OSLIAN_STREAM_LOG": "verbose"}, {"OSLIAN_STREAM_LOG_FORMAT": "xml"},
                    {"OSLIAN_STREAM_LOG_PAYLOAD": "raw"}):
            with self.assertRaises(ValueError):
                StreamEventLogger.from_env(bad)

    def test_off_writes_nothing(self):
        sink = io.StringIO()
        service, client = _service(StreamEventLogger(mode=LOG_OFF, stream=sink))
        try:
            sid = client.post("/api/demo-streams").get_json()["session_id"]
            client.post(f"/api/demo-streams/{sid}/step")
            self.assertEqual(sink.getvalue(), "")
        finally:
            service.close()

    def test_json_format_is_one_object_per_line(self):
        sink = io.StringIO()
        service, client = _service(StreamEventLogger(log_format=FORMAT_JSON, stream=sink))
        try:
            sid = client.post("/api/demo-streams").get_json()["session_id"]
            client.post(f"/api/demo-streams/{sid}/step")
            records = [json.loads(l) for l in sink.getvalue().splitlines() if l.strip()]
            self.assertTrue(records)
            for record in records:
                self.assertEqual(record["stage"], "event_emit")
                for key in ("timestamp", "session", "epoch", "seq", "event_id", "type", "source"):
                    self.assertIn(key, record)
        finally:
            service.close()

    def test_full_payload_only_for_a_synthetic_fixture_source(self):
        sink = io.StringIO()
        service, client = _service(StreamEventLogger(payload_level=PAYLOAD_FULL, stream=sink))
        try:
            sid = client.post("/api/demo-streams").get_json()["session_id"]
            client.post(f"/api/demo-streams/{sid}/step")
            self.assertIn("payload=", sink.getvalue())
        finally:
            service.close()

    def test_non_synthetic_source_is_forced_to_a_redacted_summary(self):
        sink = io.StringIO()
        logger = StreamEventLogger(payload_level=PAYLOAD_FULL, stream=sink)
        event = {"sequence": 7, "event_id": "abc", "event_type": "fhir.resource.accepted",
                 "emitted_at": "2026-03-14T08:00:00.000Z",
                 "payload": {"resource": {"resourceType": "Observation", "id": "x", "meta": {"versionId": "1"}},
                             "authorization": "Bearer sk-live-do-not-log", "effective_time": "t"}}
        logger.event_emit({"session_id": "s", "epoch": 1, "source": EpicFhirSource.source_system}, event)
        line = sink.getvalue()
        self.assertIn(PREFIX, line)
        self.assertNotIn("payload=", line)
        self.assertNotIn("Bearer", line)
        self.assertNotIn("sk-live-do-not-log", line)
        self.assertNotIn("authorization", line.lower())

    def test_logging_failure_never_breaks_the_stream(self):
        class Exploding(io.StringIO):
            def write(self, _):
                raise OSError("console gone")

        logger = StreamEventLogger(stream=Exploding())
        logger.event_emit({"session_id": "s", "epoch": 1, "source": SYNTHETIC_SOURCE_SYSTEM},
                          {"sequence": 1, "event_id": "e", "event_type": "stream.state",
                           "emitted_at": "t", "payload": {"status": "running"}})


class ConsoleSecretScanTests(unittest.TestCase):
    PATTERNS = {
        "bearer token": re.compile(r"Bearer\s+\S+", re.I),
        "authorization header": re.compile(r"authorization", re.I),
        "api key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
        "private key": re.compile(r"BEGIN (?:RSA |EC )?PRIVATE KEY"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "mrn": re.compile(r"\bMRN\b"),
        "windows path": re.compile(r"[A-Za-z]:\\\\?Users|conda_envs"),
    }

    def test_a_full_synthetic_run_never_prints_a_secret_or_identifier(self):
        sink = io.StringIO()
        service, client = _service(StreamEventLogger(payload_level=PAYLOAD_FULL, stream=sink))
        try:
            sid = client.post("/api/demo-streams").get_json()["session_id"]
            client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
            client.post(f"/api/demo-streams/{sid}/start")
            time.sleep(3.0)
            client.post(f"/api/demo-streams/{sid}/pause")
            time.sleep(1.0)
            text = sink.getvalue()
            self.assertTrue(text.strip())
            for name, pattern in self.PATTERNS.items():
                self.assertIsNone(pattern.search(text), f"{name} leaked into the console")
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
