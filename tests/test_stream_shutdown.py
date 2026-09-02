"""Shutdown must be quick, complete, and repeatable.

Two defects motivated these. ``RunScheduler.stop()`` used to set a flag and let
the thread return while leaving ``_pending`` populated, so ``idle()`` never
became true again and ``SessionManager.shutdown()`` spent its full 10 s timeout
waiting for a state that could not arrive - a plain Ctrl+C cost ten seconds.
And ``_pending`` was a list that every trigger appended to, so a long inference
under a 20x feed grew memory in proportion to the feed.

Each test here fails on that code.
"""

import io
import os
import threading
import time
import unittest

os.environ.setdefault("OSLIAN_STREAM_DB", ":memory:")
os.environ.setdefault("OSLIAN_WHOLEBODY_RUNTIME", "fixture")

from flask import Flask  # noqa: E402

from oslian_demo.monitoring.api import StreamService, create_blueprint  # noqa: E402
from oslian_demo.monitoring.eventlog import LOG_QUEUE_MAXSIZE, StreamEventLogger  # noqa: E402
from oslian_demo.monitoring.session import RUN_DEBOUNCE_SECONDS  # noqa: E402
from oslian_demo.monitoring.wholebody import RuntimeConfig, WholeBodyRuntime  # noqa: E402

FIXTURE = RuntimeConfig("fixture", None, "python", None)


class SlowRuntime(WholeBodyRuntime):
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


def _stream_threads():
    return [t for t in threading.enumerate()
            if t.is_alive() and (t.name.startswith("runs:") or t.name == "oslian-log"
                                 or t.name.startswith("stream:"))]


class SchedulerShutdownTests(unittest.TestCase):
    def test_stop_while_a_run_is_only_pending_cancels_it_and_returns_at_once(self):
        runtime = SlowRuntime(delay=0.05)
        service, client = _app(runtime)
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            session = service.sessions.get(sid)
            session.scheduler.wait_idle(timeout=20)
            calls_before = runtime.calls

            # request inside the debounce window, then stop before it can start
            session.scheduler.request("trigger-during-debounce")
            self.assertTrue(session.scheduler.stats()["pending_runs"], "nothing was pending")

            started = time.monotonic()
            ended = session.scheduler.stop(timeout=5)
            elapsed = time.monotonic() - started

            self.assertTrue(ended, "the scheduler thread did not end")
            self.assertLess(elapsed, RUN_DEBOUNCE_SECONDS + 1.0,
                            f"stop waited for the pending run ({elapsed:.2f}s)")
            self.assertTrue(session.scheduler.idle(), "idle() is still false after stop")
            self.assertEqual(session.scheduler.stats()["pending_runs"], 0)
            self.assertEqual(runtime.calls, calls_before,
                             "a cancelled pending run started an inference anyway")
        finally:
            service.close()

    def test_stop_lets_an_executing_inference_finish_and_persist(self):
        runtime = SlowRuntime(delay=1.2)
        service, client = _app(runtime)
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            session = service.sessions.get(sid)
            client.post(f"/api/demo-streams/{sid}/step")
            self.assertTrue(runtime.busy.wait(20), "no run started")

            started = time.monotonic()
            ended = session.scheduler.stop(timeout=20)
            elapsed = time.monotonic() - started

            self.assertTrue(ended, "the scheduler thread did not end")
            self.assertGreater(elapsed, 0.2, "stop returned before the inference could finish")
            self.assertTrue(session.scheduler.idle())
            history = client.get(f"/api/patients/syn-pat-0001/forecast-history?session_id={sid}").get_json()
            completed = [r for r in history["runs"] if r["status"] == "completed"]
            self.assertTrue(completed, "the in-flight run was lost instead of persisted")
        finally:
            service.close()

    def test_a_request_after_stop_is_refused(self):
        runtime = SlowRuntime(delay=0.05)
        service, client = _app(runtime)
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            session = service.sessions.get(sid)
            session.scheduler.wait_idle(timeout=20)
            session.scheduler.stop(timeout=5)
            calls_before = runtime.calls

            session.scheduler.request("after-stop")
            time.sleep(RUN_DEBOUNCE_SECONDS + 0.3)

            self.assertEqual(session.scheduler.stats()["pending_runs"], 0,
                             "a request was accepted after stop")
            self.assertTrue(session.scheduler.idle())
            self.assertEqual(runtime.calls, calls_before, "an inference started after stop")
        finally:
            service.close()

    def test_stop_is_idempotent(self):
        service, client = _app()
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            session = service.sessions.get(sid)
            for _ in range(3):
                self.assertTrue(session.scheduler.stop(timeout=5))
                self.assertTrue(session.scheduler.idle())
        finally:
            service.close()
            service.close()          # the service itself must survive a second close

    def test_shutdown_is_prompt_and_leaves_no_live_threads(self):
        before = {t.ident for t in _stream_threads()}
        sink = io.StringIO()
        runtime = SlowRuntime(delay=0.05)
        service, client = _app(runtime, logger=StreamEventLogger(stream=sink))
        sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
        client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
        client.post(f"/api/demo-streams/{sid}/start")
        time.sleep(1.5)                                   # let the burst build a pending run

        started = time.monotonic()
        service.close()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 5.0, f"shutdown took {elapsed:.2f}s; the old code idled for 10s")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and {t.ident for t in _stream_threads()} - before:
            time.sleep(0.05)
        leftover = [t.name for t in _stream_threads() if t.ident not in before]
        self.assertEqual(leftover, [], f"threads outlived shutdown: {leftover}")


class PendingStateIsBoundedTests(unittest.TestCase):
    def test_triggers_collapse_into_a_fixed_size_summary(self):
        service, client = _app()
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            session = service.sessions.get(sid)
            scheduler = session.scheduler
            scheduler.stop(timeout=5)                     # freeze the loop; only the state under test moves

            pending = scheduler._pending
            for i in range(50_000):
                pending.add(f"trigger-{i}")

            self.assertEqual(pending.count, 50_000)
            self.assertEqual(pending.first, "trigger-0")
            self.assertEqual(pending.latest, "trigger-49999")
            # constant space: three fields, whatever the trigger volume
            self.assertEqual(set(vars(pending)), {"first", "latest", "count"})
            summary = pending.take()
            self.assertEqual(summary, {"first": "trigger-0", "latest": "trigger-49999", "count": 50_000})
            self.assertFalse(pending, "take() must leave the pending state empty")
        finally:
            service.close()

    def test_a_run_records_the_bounded_summary_not_a_list_of_ids(self):
        service, client = _app()
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            session = service.sessions.get(sid)
            client.post(f"/api/demo-streams/{sid}/step")
            session.scheduler.wait_idle(timeout=30)
            history = client.get(f"/api/patients/syn-pat-0001/forecast-history?session_id={sid}").get_json()
            run = history["runs"][0]
            self.assertIsInstance(run["triggers"], dict)
            self.assertEqual(set(run["triggers"]), {"first", "latest", "count"})
            self.assertGreaterEqual(run["triggers"]["count"], 1)
            # the audit record of what was consumed is unchanged
            self.assertGreater(run["input_resource_count"], 0)
        finally:
            service.close()

    def test_pending_stays_at_one_run_through_a_sustained_burst(self):
        runtime = SlowRuntime(delay=0.3)
        service, client = _app(runtime)
        try:
            sid = client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]
            session = service.sessions.get(sid)
            client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
            client.post(f"/api/demo-streams/{sid}/start")
            peak_triggers = 0
            end = time.time() + 45
            while time.time() < end:
                stats = session.scheduler.stats()
                peak_triggers = max(peak_triggers, stats["pending_triggers"])
                self.assertLessEqual(stats["in_flight"], 1)
                self.assertLessEqual(stats["pending_runs"], 1)
                if client.get(f"/api/demo-streams/{sid}").get_json()["status"] == "completed":
                    break
                time.sleep(0.02)
            session.scheduler.wait_idle(timeout=30)
            self.assertEqual(session.scheduler.stats()["max_pending_runs"], 1)
            # triggers are counted, never accumulated as objects
            self.assertGreater(peak_triggers, 0, "the burst never produced a pending trigger")
        finally:
            service.close()


class ConsoleAccountingTests(unittest.TestCase):
    def test_flush_waits_for_every_accepted_line_including_batches(self):
        sink = io.StringIO()
        logger = StreamEventLogger(stream=sink)
        try:
            context = {"session_id": "s", "epoch": 1, "source": "synthetic-fhir-r4"}
            for i in range(1000):
                logger.event_emit(context, {"sequence": i, "event_id": f"e{i}", "event_type": "stream.state",
                                            "emitted_at": "t", "payload": {"status": "running"}})
            self.assertTrue(logger.flush(timeout=5), "flush gave up before the sink caught up")
            written = [l for l in sink.getvalue().splitlines() if l.strip()]
            self.assertEqual(len(written), 1000, "flush returned with lines still unwritten")
        finally:
            logger.close()

    def test_a_sink_that_raises_still_settles_the_batch(self):
        class Exploding(io.StringIO):
            def write(self, _):
                raise OSError("console gone")

        logger = StreamEventLogger(stream=Exploding())
        try:
            context = {"session_id": "s", "epoch": 1, "source": "synthetic-fhir-r4"}
            for i in range(20):
                logger.event_emit(context, {"sequence": i, "event_id": "e", "event_type": "stream.state",
                                            "emitted_at": "t", "payload": {"status": "running"}})
            self.assertTrue(logger.flush(timeout=3),
                            "flush hung on lines the sink refused to accept")
        finally:
            logger.close()

    def test_a_line_offered_after_close_is_counted_not_queued_behind_the_sentinel(self):
        sink = io.StringIO()
        logger = StreamEventLogger(stream=sink)
        context = {"session_id": "s", "epoch": 1, "source": "synthetic-fhir-r4"}
        logger.event_emit(context, {"sequence": 1, "event_id": "e", "event_type": "stream.state",
                                    "emitted_at": "t", "payload": {"status": "running"}})
        logger.close()
        dropped_before = logger.dropped
        logger.event_emit(context, {"sequence": 2, "event_id": "e2", "event_type": "stream.state",
                                    "emitted_at": "t", "payload": {"status": "running"}})
        self.assertEqual(logger.dropped, dropped_before + 1,
                         "a post-close line was neither written nor counted")
        self.assertNotIn("seq=2", sink.getvalue())

    def test_close_is_idempotent_and_concurrent_closes_are_safe(self):
        sink = io.StringIO()
        logger = StreamEventLogger(stream=sink)
        writer = logger._writer
        context = {"session_id": "s", "epoch": 1, "source": "synthetic-fhir-r4"}
        for i in range(200):
            logger.event_emit(context, {"sequence": i, "event_id": "e", "event_type": "stream.state",
                                        "emitted_at": "t", "payload": {"status": "running"}})
        errors = []

        def closer():
            try:
                logger.close(timeout=5)
            except Exception as exc:                      # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=closer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertFalse([t for t in threads if t.is_alive()], "a concurrent close deadlocked")
        logger.close()                                    # once more, after the fact
        # this logger's writer specifically: other tests in this process have
        # their own, and a global name scan would fail on theirs
        self.assertFalse(writer.is_alive(), "the writer outlived close")
        self.assertTrue(logger.flush(timeout=1), "accounting was left unsettled by close")

    def test_a_full_queue_never_blocks_the_caller(self):
        class Blocking(io.StringIO):
            def __init__(self):
                super().__init__()
                self.release = threading.Event()

            def write(self, text):
                self.release.wait(10)
                return super().write(text)

        sink = Blocking()
        logger = StreamEventLogger(stream=sink)
        try:
            context = {"session_id": "s", "epoch": 1, "source": "synthetic-fhir-r4"}
            started = time.monotonic()
            for i in range(LOG_QUEUE_MAXSIZE + 2000):
                logger.event_emit(context, {"sequence": i, "event_id": "e", "event_type": "stream.state",
                                            "emitted_at": "t", "payload": {"status": "running"}})
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 5.0, f"offering lines blocked for {elapsed:.2f}s")
            self.assertLessEqual(logger._queue.qsize(), LOG_QUEUE_MAXSIZE)
            self.assertGreater(logger.dropped, 0, "overflow was not counted")
        finally:
            sink.release.set()
            logger.close()


if __name__ == "__main__":
    unittest.main()
