"""The stream must stay answerable while the model is busy.

These are regressions for a measured failure, not for a suspicion. With the
release runtime attached and the synthetic stream at 20x, the demo became
unusable: ``/api/model/health`` had a p95 of 515 ms and a max of 672 ms because
it queued behind inference on the worker lock, and the console wrote and flushed
to the terminal synchronously on the ingestion thread and again on every SSE
delivery, so a terminal that stopped reading stopped the stream.

Each test below fails on the code that produced those numbers.
"""

import io
import os
import threading
import time
import unittest

os.environ.setdefault("OSLIAN_STREAM_DB", ":memory:")
os.environ.setdefault("OSLIAN_WHOLEBODY_RUNTIME", "fixture")

from flask import Flask  # noqa: E402

from oslian_demo.config import load_model_lock  # noqa: E402
from oslian_demo.monitoring.api import StreamService, create_blueprint  # noqa: E402
from oslian_demo.monitoring.eventlog import StreamEventLogger  # noqa: E402
from oslian_demo.monitoring.wholebody import (  # noqa: E402
    ReleaseWorker, RuntimeConfig, WholeBodyRuntime,
)

FIXTURE = RuntimeConfig("fixture", None, "python", None)


class SlowRuntime(WholeBodyRuntime):
    """A fixture runtime that takes as long as a real release inference."""

    def __init__(self, delay: float):
        super().__init__(FIXTURE)
        self.delay = delay
        self.calls = 0
        self.busy = threading.Event()

    def run(self, model_input, state):
        self.calls += 1
        self.busy.set()
        time.sleep(self.delay)
        return super().run(model_input, state)


def _app(runtime=None, logger=None):
    service = StreamService(":memory:", runtime=runtime or WholeBodyRuntime(FIXTURE), logger=logger)
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(service))
    return service, app.test_client()


def _timed(fn):
    t0 = time.monotonic()
    result = fn()
    return time.monotonic() - t0, result


class ReadsDoNotWaitForInferenceTests(unittest.TestCase):
    """The measured symptom: health blocked for the length of every inference."""

    def test_health_does_not_queue_behind_an_inference_in_flight(self):
        worker = ReleaseWorker(RuntimeConfig("release", None, "python", None), "deadbeef")
        worker.startup = {"ready": True, "checks": [], "commit": "deadbeef", "scikit_learn": "1.7.2"}

        class _Process:
            def poll(self):
                return None

        worker._process = _Process()
        holding = threading.Event()
        release = threading.Event()

        def inference():
            with worker._lock:          # exactly what predict() holds for its round trip
                holding.set()
                release.wait(5)

        thread = threading.Thread(target=inference, daemon=True)
        thread.start()
        self.assertTrue(holding.wait(2), "the fake inference never took the lock")
        try:
            elapsed, startup = _timed(worker.status)
            self.assertLess(elapsed, 0.5, "health read waited for the inference to finish")
            self.assertTrue(startup["ready"])
            self.assertEqual(startup["scikit_learn"], "1.7.2")
        finally:
            release.set()
            thread.join(timeout=5)

    def test_navigation_and_control_answer_while_a_run_is_executing(self):
        runtime = SlowRuntime(delay=1.5)
        service, client = _app(runtime)
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            client.post(f"/api/demo-streams/{sid}/step")
            self.assertTrue(runtime.busy.wait(10), "no run started")

            # while that run is in flight, everything a person can reach must answer
            health_s, health = _timed(lambda: client.get("/api/model/health"))
            caps_s, _ = _timed(lambda: client.get("/api/model/capabilities"))
            anatomy_s, _ = _timed(lambda: client.get("/api/model/anatomy"))
            state_s, _ = _timed(lambda: client.get(f"/api/demo-streams/{sid}"))
            pause_s, _ = _timed(lambda: client.post(f"/api/demo-streams/{sid}/pause"))

            self.assertIn(health.status_code, (200, 503))
            for name, seconds in (("health", health_s), ("capabilities", caps_s), ("anatomy", anatomy_s),
                                  ("session state", state_s), ("pause", pause_s)):
                self.assertLess(seconds, 0.5, f"{name} waited on the model ({seconds:.2f}s)")
        finally:
            service.close()

    def test_every_monitoring_view_can_be_driven_while_the_stream_runs(self):
        """The four Monitoring views read these documents; none may block on a run."""
        runtime = SlowRuntime(delay=1.0)
        service, client = _app(runtime)
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
            client.post(f"/api/demo-streams/{sid}/start")
            self.assertTrue(runtime.busy.wait(15), "no run started")
            pid = "syn-pat-0001"
            q = f"?session_id={sid}"
            views = {
                "anatomy": f"/api/model/anatomy",
                "functions": f"/api/model/functions",
                "all signals": f"/api/patients/{pid}/timeline{q}",
                "all forecast cells": f"/api/patients/{pid}/forecasts{q}",
                "organ state": f"/api/patients/{pid}/state{q}",
                "run history": f"/api/patients/{pid}/forecast-history{q}",
            }
            for name, url in views.items():
                seconds, response = _timed(lambda u=url: client.get(u))
                self.assertEqual(response.status_code, 200, name)
                self.assertLess(seconds, 1.0, f"{name} waited on the model ({seconds:.2f}s)")
        finally:
            client.post(f"/api/demo-streams/{sid}/pause")
            service.close()


class ConsoleIsNeverOnTheCriticalPathTests(unittest.TestCase):
    class BlockingSink(io.StringIO):
        """A terminal that has stopped reading - a selected PowerShell window."""

        def __init__(self):
            super().__init__()
            self.release = threading.Event()
            self.blocked = threading.Event()

        def write(self, text):
            self.blocked.set()
            self.release.wait(10)
            return super().write(text)

    def test_a_blocked_console_stalls_neither_ingestion_nor_http(self):
        sink = self.BlockingSink()
        logger = StreamEventLogger(stream=sink)
        service, client = _app(logger=logger)
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            self.assertTrue(sink.blocked.wait(5), "the console writer never reached the sink")

            # the console is stuck inside write(); the stream must not care
            for _ in range(20):
                seconds, response = _timed(lambda: client.post(f"/api/demo-streams/{sid}/step"))
                self.assertEqual(response.status_code, 200)
                self.assertLess(seconds, 1.0, f"ingestion waited on the console ({seconds:.2f}s)")
            seconds, response = _timed(lambda: client.get(f"/api/demo-streams/{sid}"))
            self.assertEqual(response.status_code, 200)
            self.assertLess(seconds, 0.5, "an HTTP read waited on the console")

            delivered = client.get(f"/api/patients/syn-pat-0001/state?session_id={sid}").get_json()
            self.assertGreater(delivered["delivered_observations"], 0,
                               "observations were lost while the console was blocked")
        finally:
            sink.release.set()
            service.close()

    def test_the_console_queue_is_bounded_and_drops_oldest_with_a_count(self):
        sink = self.BlockingSink()
        logger = StreamEventLogger(stream=sink)
        try:
            event = {"sequence": 1, "event_id": "e", "event_type": "stream.state",
                     "emitted_at": "t", "payload": {"status": "running"}}
            context = {"session_id": "s", "epoch": 1, "source": "synthetic-fhir-r4"}
            from oslian_demo.monitoring.eventlog import LOG_QUEUE_MAXSIZE
            logger.event_emit(context, event)                  # gets the writer into write()
            self.assertTrue(sink.blocked.wait(5), "the console writer never reached the sink")
            for i in range(LOG_QUEUE_MAXSIZE + 500):
                logger.event_emit(context, {**event, "sequence": i})
            self.assertLessEqual(logger._queue.qsize(), LOG_QUEUE_MAXSIZE,
                                 "the console queue grew without bound")
            self.assertGreater(logger.dropped, 0, "overflow was not counted")
        finally:
            sink.release.set()
            logger.close()

    def test_close_flushes_what_is_queued(self):
        sink = io.StringIO()
        logger = StreamEventLogger(stream=sink)
        context = {"session_id": "s", "epoch": 1, "source": "synthetic-fhir-r4"}
        for i in range(50):
            logger.event_emit(context, {"sequence": i, "event_id": f"e{i}", "event_type": "stream.state",
                                        "emitted_at": "t", "payload": {"status": "running"}})
        logger.close()
        self.assertEqual(len([l for l in sink.getvalue().splitlines() if l.strip()]), 50)


class RunSchedulerIsBoundedTests(unittest.TestCase):
    def test_one_inference_in_flight_one_run_pending_and_the_last_run_is_the_newest_prefix(self):
        runtime = SlowRuntime(delay=0.5)
        service, client = _app(runtime)
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            session = service.sessions.get(sid)
            client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
            client.post(f"/api/demo-streams/{sid}/start")

            peak = {"in_flight": 0, "pending_runs": 0}
            end = time.time() + 60
            while time.time() < end:
                stats = session.scheduler.stats()
                peak["in_flight"] = max(peak["in_flight"], stats["in_flight"])
                peak["pending_runs"] = max(peak["pending_runs"], stats["pending_runs"])
                if client.get(f"/api/demo-streams/{sid}").get_json()["status"] == "completed":
                    break
                time.sleep(0.02)

            session.scheduler.wait_idle(timeout=60)
            stats = session.scheduler.stats()
            self.assertEqual(stats["max_in_flight"], 1, "more than one inference ran at once")
            self.assertEqual(stats["max_pending_runs"], 1, "runs queued up instead of merging")
            self.assertLessEqual(peak["in_flight"], 1)
            self.assertLessEqual(peak["pending_runs"], 1)

            # the run that finished last must describe the newest delivered prefix
            state = client.get(f"/api/demo-streams/{sid}").get_json()
            self.assertEqual(state["status"], "completed")
            timeline = client.get(f"/api/patients/syn-pat-0001/timeline?session_id={sid}").get_json()
            history = client.get(f"/api/patients/syn-pat-0001/forecast-history?session_id={sid}").get_json()
            completed = [r for r in history["runs"] if r["status"] == "completed" and r["epoch"] == state["epoch"]]
            self.assertTrue(completed)
            self.assertEqual(completed[0]["input_resource_count"], timeline["count"],
                             "the newest run does not cover the newest delivered prefix")
            # merging must not have cost inference: far fewer runs than delivered events
            self.assertLess(len(completed), state["cursor"],
                            "a run was started for every event instead of merging the burst")
        finally:
            service.close()

    def test_an_unchanged_prefix_does_not_start_another_inference(self):
        runtime = SlowRuntime(delay=0.05)
        service, client = _app(runtime)
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            session = service.sessions.get(sid)
            client.post(f"/api/demo-streams/{sid}/step")
            session.scheduler.wait_idle(timeout=30)
            after_first = runtime.calls
            self.assertGreater(after_first, 0)

            # ask again with nothing newly delivered
            session.scheduler.request("synthetic-repeat")
            session.scheduler.wait_idle(timeout=30)
            self.assertEqual(runtime.calls, after_first,
                             "the model ran again on an unchanged delivered prefix")
            self.assertGreater(session.scheduler.stats()["runs_superseded"], 0)
        finally:
            service.close()


class ForecastStatesAreDistinguishableTests(unittest.TestCase):
    """Model, carried-forward and observation-only must each be identifiable."""

    def test_the_registry_splits_into_model_capable_and_carried_forward_cells(self):
        runtime = WholeBodyRuntime(FIXTURE)
        cells = runtime.contract["catalog"]["cells"]
        model_capable = [c for c in cells if c["can_move"]]
        carried_forward = [c for c in cells if not c["can_move"]]
        contract = load_model_lock()["runtime_contract"]
        self.assertEqual(len(cells), contract["catalog_cells"])
        self.assertEqual(len(model_capable), contract["model_capable_cells"])
        self.assertEqual(len(carried_forward), contract["persistence_fallback_cells"])
        for cell in carried_forward:
            self.assertEqual(cell["source_model"], "patient_persistence_anchor",
                             "a carried-forward cell must never claim a model as its source")

    def test_a_signal_outside_the_registry_has_no_forecast_cell_at_all(self):
        runtime = WholeBodyRuntime(FIXTURE)
        targets = {c["target"] for c in runtime.contract["catalog"]["cells"]}
        variables = set(runtime.contract["variables"])
        observation_only = variables - targets
        self.assertTrue(observation_only,
                        "the fixture must keep at least one observation-only signal for this state")
        for variable in observation_only:
            self.assertFalse([c for c in runtime.contract["catalog"]["cells"] if c["target"] == variable])


if __name__ == "__main__":
    unittest.main()
