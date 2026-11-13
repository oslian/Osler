"""Functional-group schema, retrospective replay source, step control, and mode isolation."""

from collections import Counter
import os
import time
import unittest

os.environ.setdefault("OSLIAN_STREAM_DB", ":memory:")
os.environ.setdefault("OSLIAN_WHOLEBODY_RUNTIME", "fixture")

from flask import Flask  # noqa: E402

from oslian_demo.monitoring import fhir, mapping  # noqa: E402
from oslian_demo.monitoring.api import StreamService, create_blueprint  # noqa: E402
from oslian_demo.monitoring.functions import functions_document, load_function_map  # noqa: E402
from oslian_demo.monitoring.source import RetrospectiveCaseSource, make_source  # noqa: E402
from oslian_demo.monitoring.wholebody import RuntimeConfig, WholeBodyRuntime, load_contract_fixture  # noqa: E402


def _app():
    service = StreamService(":memory:", runtime=WholeBodyRuntime(RuntimeConfig("fixture", None, "python", None)))
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(service))
    return service, app.test_client()


class FunctionalGroupTests(unittest.TestCase):
    def test_groups_partition_the_checkpoint_modules_and_variables(self):
        contract = load_contract_fixture()
        document = load_function_map(modules=contract["modules"])
        ids = [g["id"] for g in document["groups"]]
        self.assertEqual(set(ids), set(contract["modules"]))
        self.assertEqual(len(ids), 11)
        covered = [v for g in document["groups"] for v in g["module_variables"]]
        self.assertEqual(len(covered), 53)
        self.assertEqual(set(covered), set(mapping.CANONICAL_UNITS))
        renal = next(g for g in document["groups"] if g["id"] == "renal_urinary")
        self.assertEqual(renal["primary"], ["creatinine", "bun", "urine_output"])
        self.assertIn("potassium", renal["related"])
        self.assertNotIn("potassium", renal["module_variables"], "related indicators come from other modules")
        self.assertEqual(document["organ_primary_function"]["kidneys"], "renal_urinary")

    def test_functions_document_joins_cells_and_anatomy(self):
        contract = load_contract_fixture()
        document = functions_document(contract["catalog"]["cells"], contract["modules"])
        renal = next(g for g in document["groups"] if g["id"] == "renal_urinary")
        self.assertEqual([o["id"] for o in renal["organs"]], ["kidneys"])
        self.assertIn("creatinine@6h", {c["cell"] for c in renal["cells"]})
        self.assertIn("potassium@12h", {c["cell"] for c in renal["related_cells"]})
        total = sum(len(g["cells"]) for g in document["groups"])
        self.assertEqual(total, len(contract["catalog"]["cells"]), "every registry cell belongs to exactly one module")
        for group in document["groups"]:
            for word in ("kidney", "liver", "lung"):
                self.assertNotIn(word, group["id"], "functional groups are not organs")

    def test_functional_group_map_rejects_invalid_shapes(self):
        import json
        import tempfile
        from pathlib import Path
        bad = {"schema": "oslian.functional-group-map.v1", "groups": [
            {"id": "x", "label": "x", "module_variables": ["creatinine"], "primary": ["creatinine"], "secondary": [],
             "related": ["creatinine"], "organs": ["kidneys"]}], "organ_primary_function": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_function_map(path)


class RetrospectiveSourceTests(unittest.TestCase):
    def test_rows_become_valid_fhir_observations_at_measurement_time(self):
        source = RetrospectiveCaseSource()
        events = source.events()
        self.assertEqual(len(events), 38)
        for event in events:
            effective = fhir.validate_observation(event.resource)
            self.assertIsNotNone(effective.effective)
            outcome = mapping.map_observation(event.resource, effective.effective, effective.period_hours)
            self.assertTrue(outcome.mapped, event.resource["id"])
        # bicarbonate at row 2 carries age 5.92 h -> same measurement time as row 1 -> same resource id (idempotent ledger)
        bicarb = [e.resource["id"] for e in events if "bicarbonate" in e.resource["id"]]
        self.assertEqual(len(bicarb), 3)
        self.assertEqual(bicarb[0], bicarb[1])
        urine = next(e.resource for e in events if "urine_output" in e.resource["id"])
        self.assertIn("effectivePeriod", urine)
        describe = source.describe()
        self.assertTrue(describe["retrospective"])
        self.assertEqual(describe["time_axis"], "hours_since_admission")
        self.assertEqual(len(describe["rows"]), 3)
        self.assertEqual(source.initial_bundle()["type"], "searchset")
        self.assertEqual([e["resource"]["resourceType"] for e in source.initial_bundle()["entry"]], ["Patient", "Encounter"])

    def test_unknown_source_key_is_rejected(self):
        with self.assertRaises(KeyError):
            make_source("epic")


class RetrospectiveReplayApiTests(unittest.TestCase):
    def setUp(self):
        self.service, self.client = _app()

    def tearDown(self):
        self.service.close()

    def test_step_reveals_one_row_at_a_time_and_withholds_the_rest(self):
        self.assertEqual(self.client.post("/api/demo-streams", json={"source": "epic"}).status_code, 400)
        created = self.client.post("/api/demo-streams", json={"source": "retrospective"})
        self.assertEqual(created.status_code, 201)
        state = created.get_json()
        sid, pid = state["session_id"], state["patient_id"]
        self.assertEqual(state["source"]["mode"], "retrospective")
        self.assertEqual(state["total_events"], 38)
        # nothing revealed yet: no timeline, no run
        snapshot = self.client.get(f"/api/patients/{pid}/state?session_id={sid}").get_json()
        self.assertEqual(snapshot["variables"], {})
        step1 = self.client.post(f"/api/demo-streams/{sid}/step").get_json()
        self.assertEqual(step1["status"], "paused")
        self.assertEqual(step1["cursor"], 12, "row 1 = 12 resources delivered in one step")
        self.service.sessions.get(sid).scheduler.wait_idle(10)
        snapshot = self.client.get(f"/api/patients/{pid}/state?session_id={sid}").get_json()
        self.assertEqual(len(snapshot["variables"]), 12)
        self.assertNotIn("urine_output", snapshot["variables"], "row 2 is withheld")
        run = self.service.repo.latest_completed_run(sid, 1)
        self.assertIsNotNone(run)
        self.assertEqual(run["input_resource_count"], 12)
        step2 = self.client.post(f"/api/demo-streams/{sid}/step").get_json()
        self.assertEqual(step2["cursor"], 25)
        self.service.sessions.get(sid).scheduler.wait_idle(10)
        timeline = self.client.get(f"/api/patients/{pid}/timeline?session_id={sid}").get_json()
        self.assertIn("urine_output", timeline["series"])
        self.assertEqual(len(timeline["series"]["creatinine"]), 2)
        latest = self.service.repo.latest_completed_run(sid, 1)
        self.assertGreater(latest["input_cutoff"], run["input_cutoff"], "anchor moves forward with the revealed prefix")
        history = self.client.get(f"/api/patients/{pid}/forecast-history?session_id={sid}").get_json()
        self.assertGreaterEqual(len(history["runs"]), 2, "every anchor keeps its run")
        step3 = self.client.post(f"/api/demo-streams/{sid}/step").get_json()
        self.assertEqual(step3["cursor"], 38)
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/step").get_json()["cursor"], 38, "step at the end is a no-op")

    def test_modes_have_separate_sessions_and_do_not_mix(self):
        live = self.client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()
        retro = self.client.post("/api/demo-streams", json={"source": "retrospective"}).get_json()
        self.assertNotEqual(live["session_id"], retro["session_id"])
        self.assertNotEqual(live["patient_id"], retro["patient_id"])
        self.assertTrue(live["source"]["synthetic"])
        self.assertFalse(retro["source"]["synthetic"])
        self.assertTrue(retro["source"]["retrospective"])
        self.assertEqual(self.client.post("/api/demo-streams", json={"source": "retrospective"}).status_code, 200)
        sources = self.client.get("/api/demo-streams/sources").get_json()["sources"]
        self.assertEqual([s["source_key"] for s in sources], ["synthetic", "retrospective"])
        self.assertNotIn("epic", " ".join(s["provenance"].lower() for s in sources))
        functions = self.client.get("/api/model/functions").get_json()
        self.assertEqual(len(functions["groups"]), 11)

    def test_step_also_advances_the_live_stream_one_event(self):
        state = self.client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()
        sid = state["session_id"]
        first = self.client.post(f"/api/demo-streams/{sid}/step").get_json()
        self.assertEqual(first["cursor"], 1)
        self.assertEqual(first["status"], "paused")
        self.assertEqual(self.client.post(f"/api/demo-streams/{sid}/start").get_json()["status"], "running")
        time.sleep(0.3)
        stepped = self.client.post(f"/api/demo-streams/{sid}/step").get_json()
        self.assertEqual(stepped["status"], "paused", "stepping while running pauses the clock at the delivered event")
        events = self.service.repo.events_after(sid, 0)
        kinds = Counter(e["event_type"] for e in events)
        self.assertGreaterEqual(kinds["fhir.resource.accepted"], 2)


if __name__ == "__main__":
    unittest.main()
