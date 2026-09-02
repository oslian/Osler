"""HTTP + SSE contract: session controls, event chain, anchors, reconnect, medication-safety isolation."""

from collections import Counter
import os
import time
import unittest

os.environ.setdefault("OSLIAN_STREAM_DB", ":memory:")
os.environ.setdefault("OSLIAN_WHOLEBODY_RUNTIME", "fixture")

from flask import Flask  # noqa: E402

from oslian_demo.monitoring.api import StreamService, create_blueprint  # noqa: E402
from oslian_demo.monitoring.timeline import delivered_timeline  # noqa: E402
from oslian_demo.monitoring.wholebody import RuntimeConfig, WholeBodyRuntime  # noqa: E402


def _app(runtime=None):
    service = StreamService(":memory:", runtime=runtime or WholeBodyRuntime(RuntimeConfig("fixture", None, "python", None)))
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(service))
    return service, app.test_client()


def _wait(client, sid, status="completed", timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        state = client.get(f"/api/demo-streams/{sid}").get_json()
        if state["status"] == status:
            return state
        time.sleep(0.2)
    raise AssertionError(f"session did not reach {status}")


class StreamApiTests(unittest.TestCase):
    def setUp(self):
        self.service, self.client = _app()

    def tearDown(self):
        self.service.close()

    def test_create_is_idempotent_and_controls_are_idempotent(self):
        first = self.client.post("/api/demo-streams")
        second = self.client.post("/api/demo-streams")
        self.assertEqual((first.status_code, second.status_code), (201, 200))
        sid = first.get_json()["session_id"]
        self.assertEqual(second.get_json()["session_id"], sid)
        self.assertTrue(first.get_json()["synthetic"])
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/pause").get_json()["status"], "idle")
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/start").get_json()["status"], "running")
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/start").get_json()["status"], "running")
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/pause").get_json()["status"], "paused")
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/pause").get_json()["status"], "paused")
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 7}).status_code, 400)
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 5}).get_json()["speed"], 5.0)
        epoch = self.client.post(f"/api/demo-streams/{sid}/reset").get_json()["epoch"]
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/reset").get_json()["epoch"], epoch)
        self.assertEqual(self.client.get("/api/demo-streams/nope").status_code, 404)

    def test_observation_flows_from_fhir_to_run_history_and_sse(self):
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        pid = self.client.get(f"/api/demo-streams/{sid}").get_json()["patient_id"]
        self.client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
        self.client.post(f"/api/demo-streams/{sid}/start")
        _wait(self.client, sid)
        self.service.sessions.get(sid).scheduler.wait_idle(15)
        events = self.service.repo.events_after(sid, 0, 10000)
        kinds = Counter(e["event_type"] for e in events)
        for needed in ("stream.state", "fhir.resource.accepted", "observation.normalized", "model.run.started",
                       "model.run.completed"):
            self.assertGreater(kinds[needed], 0, needed)
        self.assertEqual(kinds["model.run.failed"], 0)
        self.assertGreaterEqual(kinds["model.run.completed"], 5, "bursts coalesce but runs keep flowing")
        self.assertLess(kinds["model.run.completed"], kinds["observation.normalized"], "not one run per frame")

        # one specific Observation: accepted -> normalized -> included in a run -> visible in forecast history
        accepted = next(e for e in events if e["event_type"] == "fhir.resource.accepted"
                        and e["payload"]["resource"]["id"] == "syn-obs-creatinine-h06")
        resource = accepted["payload"]["resource"]
        self.assertEqual(resource["resourceType"], "Observation")
        self.assertEqual(accepted["payload"]["source"], "synthetic-fhir-r4")
        normalized = next(e for e in events if e["event_type"] == "observation.normalized"
                          and e["payload"]["resource_id"] == "syn-obs-creatinine-h06")
        self.assertEqual(normalized["payload"]["values"][0]["variable"], "creatinine")
        self.assertEqual(normalized["payload"]["values"][0]["unit"], "mg/dL")
        completed = [e for e in events if e["event_type"] == "model.run.completed"]
        later = [e for e in completed if e["payload"]["input_cutoff"] >= normalized["payload"]["effective_time"]]
        self.assertTrue(later)
        # a run's input is frozen when it starts, so the observation is in the first run that started after it
        runs = [self.service.repo.run(e["payload"]["run_id"]) for e in later]
        run = next(r for r in runs if "Observation/syn-obs-creatinine-h06/_history/1" in r["input_refs"])
        self.assertTrue(all("Observation/syn-obs-creatinine-h06/_history/1" in r["input_refs"]
                            for r in runs if r["started_at"] >= run["started_at"]), "once in, never dropped")
        self.assertEqual(run["model_commit"], "5a49b3969d69145efa8ef66d0c6e80d24daef5d2")
        history = self.client.get(f"/api/patients/{pid}/forecast-history").get_json()
        self.assertIn(run["run_id"], {r["run_id"] for r in history["runs"]})
        cell_history = self.client.get(f"/api/patients/{pid}/forecast-history?cell=creatinine@6h").get_json()["cell_history"]
        self.assertTrue(cell_history)

        # duplicate / amended / out-of-order / unmapped surfaced as data, never as model output
        accepted_all = [e["payload"] for e in events if e["event_type"] == "fhir.resource.accepted"]
        self.assertEqual(sum(1 for p in accepted_all if p["delivery_duplicate"]), 1)
        self.assertEqual(sum(1 for p in accepted_all if p["resource"]["id"] == "syn-obs-potassium-h12"), 2)
        unmapped = {e["payload"]["reason"] for e in events if e["event_type"] == "observation.unmapped"}
        self.assertEqual(unmapped, {"unmapped_code", "missing_unit", "excluded_status", "data_absent"})
        state = self.client.get(f"/api/patients/{pid}/state").get_json()
        self.assertEqual(state["variables"]["potassium"]["count"], 6)
        self.assertEqual(state["variables"]["temperature"]["count"], 8)
        self.assertAlmostEqual(state["variables"]["temperature"]["value"], 37.3, delta=0.6)
        self.assertEqual(state["clock"]["model_input_watermark"], state["clock"]["latest_clinical_event_time"])
        self.assertEqual(state["runtime"]["runtime_mode"], "fixture")
        self.assertFalse(state["runtime"]["runtime_attached"])

        # live head vs historical anchor
        live = self.client.get(f"/api/patients/{pid}/forecasts").get_json()
        self.assertEqual(live["anchor"], "live_head")
        self.assertEqual(len(live["cells"]), 41)
        self.assertTrue(all(c["status"] == "degraded" and c["lower"] is None and c["upper"] is None for c in live["cells"]))
        old_run = history["runs"][-1]["run_id"]
        old = self.client.get(f"/api/patients/{pid}/forecasts?run_id={old_run}").get_json()
        self.assertEqual(old["anchor"], "historical")
        timeline = self.client.get(f"/api/patients/{pid}/timeline?run_id={old_run}").get_json()
        self.assertTrue(all(pt["effective_time"] <= timeline["cutoff"] for s in timeline["series"].values() for pt in s))
        raw = self.client.get("/api/fhir/Observation/syn-obs-potassium-h12/_history/2").get_json()
        self.assertEqual(raw["resource"]["status"], "amended")
        self.assertEqual(self.client.get("/api/fhir/Observation/nope/_history/1").status_code, 404)

    def test_sse_replays_from_last_event_id(self):
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        self.client.post(f"/api/demo-streams/{sid}/start")
        self.client.post(f"/api/demo-streams/{sid}/pause")
        events = self.service.repo.events_after(sid, 0)
        self.assertGreaterEqual(len(events), 3)
        response = self.client.get(f"/api/demo-streams/{sid}/events", headers={"Last-Event-ID": str(events[-2]["sequence"])})
        self.assertEqual(response.mimetype, "text/event-stream")
        chunks = iter(response.response)
        self.assertTrue(next(chunks).startswith(b": oslian synthetic stream"))
        replay = next(chunks).decode()
        self.assertTrue(replay.startswith(f"id: {events[-1]['sequence']}\nevent: stream.state\n"))
        response.close()
        query = self.client.get(f"/api/demo-streams/{sid}/events?last_event_id={events[-2]['sequence']}")
        it = iter(query.response)
        next(it)
        self.assertIn(f"id: {events[-1]['sequence']}", next(it).decode())
        query.close()

    def test_model_endpoints(self):
        health = self.client.get("/api/model/health")
        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.get_json()["status"], "runtime_not_attached")
        cap = self.client.get("/api/model/capabilities").get_json()
        self.assertEqual(cap["catalog_cells"], 41)
        self.assertFalse(cap["causal_claim_allowed"])
        anatomy = self.client.get("/api/model/anatomy").get_json()
        self.assertEqual(len(anatomy["organs"]), 14)
        mapping = self.client.get("/api/model/mapping").get_json()
        self.assertTrue(mapping["rule_count"] > 50)


    def test_snapshot_reports_the_same_organ_routing_the_events_carried(self):
        """A page that reloads mid-stream has no events for what was already delivered,
        so /state has to state the organ routing itself - through the same map."""
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        pid = self.client.get(f"/api/demo-streams/{sid}").get_json()["patient_id"]
        self.client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
        self.client.post(f"/api/demo-streams/{sid}/start")
        _wait(self.client, sid)
        self.service.sessions.get(sid).scheduler.wait_idle(15)

        state = self.client.get(f"/api/patients/{pid}/state?session_id={sid}").get_json()
        organs = state["organs"]
        self.assertTrue(organs)
        from_events = {}
        for event in self.service.repo.events_after(sid, 0, 10000):
            if event["event_type"] != "observation.normalized":
                continue
            for organ in event["payload"].get("organ_ids", []):
                at = event["payload"].get("effective_time")
                if at and (organ not in from_events or at > from_events[organ]):
                    from_events[organ] = at
        self.assertEqual(set(organs), set(from_events), "snapshot and events must agree on which organs")
        for organ, at in from_events.items():
            self.assertEqual(organs[organ]["at"], at, organ)
            self.assertGreater(organs[organ]["count"], 0)
            self.assertTrue(organs[organ]["variables"])

        # the organ block always describes the same delivered timeline the variable
        # block does, in this epoch and any later one
        from oslian_demo.monitoring.routing import variable_routing
        routing = variable_routing()
        for snapshot in (state, self.client.post(f"/api/demo-streams/{sid}/reset")
                         and self.client.get(f"/api/patients/{pid}/state?session_id={sid}").get_json()):
            expected = {o for v in snapshot["variables"] for o in routing.organs_for(v)}
            self.assertEqual(set(snapshot["organs"]), expected)


    def test_each_historical_run_carries_its_own_organ_snapshot(self):
        """Anchoring a run must show that run's organs, not the live head's."""
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        pid = self.client.get(f"/api/demo-streams/{sid}").get_json()["patient_id"]
        self.client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
        self.client.post(f"/api/demo-streams/{sid}/start")
        _wait(self.client, sid)
        self.service.sessions.get(sid).scheduler.wait_idle(15)

        history = self.client.get(f"/api/patients/{pid}/forecast-history?session_id={sid}").get_json()
        runs = [r for r in history["runs"] if r["status"] == "completed"]
        self.assertGreaterEqual(len(runs), 5)

        def organs_for(run_id=None):
            q = f"?session_id={sid}" + (f"&run_id={run_id}" if run_id else "")
            body = self.client.get(f"/api/patients/{pid}/forecasts{q}").get_json()
            return body["anchor"], body["organs"]

        early_run, late_run = runs[-1], runs[len(runs) // 2]
        early_anchor, early = organs_for(early_run["run_id"])
        late_anchor, late = organs_for(late_run["run_id"])
        live_anchor, live = organs_for()
        self.assertEqual((early_anchor, late_anchor, live_anchor), ("historical", "historical", "live_head"))

        # each snapshot is bounded by its own run, and they are genuinely different
        for run, organs in ((early_run, early), (late_run, late)):
            self.assertTrue(organs)
            for organ, item in organs.items():
                self.assertLessEqual(item["at"], run["input_cutoff"], organ)
                self.assertTrue(item["variables"])
                self.assertGreater(item["count"], 0)
        self.assertNotEqual(early, late)
        self.assertNotEqual(late, live)
        self.assertLess(sum(o["count"] for o in early.values()), sum(o["count"] for o in live.values()))

        # live -> historical -> live is reversible and deterministic
        self.assertEqual(organs_for()[1], live)
        self.assertEqual(organs_for(early_run["run_id"])[1], early)
        self.assertEqual(organs_for()[1], live)

        # organ routing is the server's, never re-derived
        from oslian_demo.monitoring.routing import variable_routing
        routing = variable_routing()
        for organ, item in live.items():
            for variable in item["variables"]:
                self.assertIn(organ, routing.organs_for(variable))


    def test_a_historical_timeline_is_the_runs_own_inputs(self):
        """Signals, charts and organs under an anchor all come from the run's input set."""
        sid = self.client.post("/api/demo-streams").get_json()["session_id"]
        pid = self.client.get(f"/api/demo-streams/{sid}").get_json()["patient_id"]
        self.client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
        self.client.post(f"/api/demo-streams/{sid}/start")
        _wait(self.client, sid)
        self.service.sessions.get(sid).scheduler.wait_idle(15)

        history = self.client.get(f"/api/patients/{pid}/forecast-history?session_id={sid}").get_json()
        runs = [r for r in history["runs"] if r["status"] == "completed"]
        run = runs[-1]                                    # the earliest run of this epoch
        q = f"?session_id={sid}&run_id={run['run_id']}"
        timeline = self.client.get(f"/api/patients/{pid}/timeline{q}").get_json()
        organs = self.client.get(f"/api/patients/{pid}/forecasts{q}").get_json()["organs"]

        # exactly the resources the run recorded, not the current timeline cut at its cutoff
        self.assertEqual(timeline["count"], run["input_resource_count"])
        naive = delivered_timeline(self.service.repo, run["session_id"], int(run["epoch"]), run["input_cutoff"])
        self.assertGreater(len(naive), timeline["count"],
                           "the fixture must contain a late arrival or an amendment for this to mean anything")

        # nothing after the cutoff, and the charts and the figure agree on the variables
        for variable, points in timeline["series"].items():
            for point in points:
                self.assertLessEqual(point["effective_time"], run["input_cutoff"], variable)
        charted = set(timeline["state"])
        for organ, item in organs.items():
            for variable in item["variables"]:
                self.assertIn(variable, charted, f"{organ}/{variable} lit but not in the historical signals")

        # re-entering the same run - a deep link, a reload - gives the same document
        self.assertEqual(self.client.get(f"/api/patients/{pid}/timeline{q}").get_json(), timeline)


class DemoAppIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from oslian_demo import create_app
        from oslian_demo.medication.routes import SAMPLE_CASES
        cls.client = create_app().test_client()
        cls.sample = SAMPLE_CASES[0]

    def test_old_replay_only_contract_is_gone(self):
        for path in ("/api/forecast/capabilities", "/api/monitoring/cases", "/api/health/models", "/api/forecast/fixtures"):
            self.assertEqual(self.client.get(path).status_code, 404, path)
        self.assertEqual(self.client.post("/api/forecast", json={}).status_code, 404)

    def test_medication_safety_ranking_is_unchanged_by_the_stream(self):
        payload = {"fields": self.sample, "use_openfda": False}
        before = self.client.post("/api/analyze", json=payload).get_json()["result"]["candidates"]
        self.client.post("/api/demo-streams")
        after = self.client.post("/api/analyze", json=payload).get_json()["result"]["candidates"]
        self.assertEqual([c["drug"] for c in before], [c["drug"] for c in after])
        self.assertTrue(before)


@unittest.skipUnless(os.environ.get("OSLIAN_WHOLEBODY_RUNTIME") == "release", "release runtime not configured")
class ReleaseRuntimeTests(unittest.TestCase):
    def test_release_worker_produces_model_and_persistence_cells(self):
        service, client = _app(WholeBodyRuntime(RuntimeConfig.from_env()))
        try:
            health = client.get("/api/model/health").get_json()
            self.assertTrue(health["ready"], health)
            sid = client.post("/api/demo-streams").get_json()["session_id"]
            pid = client.get(f"/api/demo-streams/{sid}").get_json()["patient_id"]
            client.post(f"/api/demo-streams/{sid}/speed", json={"speed": 20})
            client.post(f"/api/demo-streams/{sid}/start")
            _wait(client, sid, timeout=180)
            service.sessions.get(sid).scheduler.wait_idle(30)
            forecasts = client.get(f"/api/patients/{pid}/forecasts").get_json()
            statuses = Counter(c["status"] for c in forecasts["cells"])
            cap = client.get("/api/model/capabilities").get_json()
            self.assertEqual(statuses["model"], cap["model_cells"])
            self.assertEqual(statuses["persistence"], cap["persistence_cells"])
            self.assertTrue(all(c["lower"] is None and c["upper"] is None for c in forecasts["cells"]))
            self.assertEqual(forecasts["run"]["model_commit"], cap["model_commit"])
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
