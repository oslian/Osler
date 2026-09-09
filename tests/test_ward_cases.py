"""A row on the ward board opens the stay it names.

1. every ward stay has a case, so no row is a broken link
2. a case carries recorded measurements and nothing generated
3. an unknown case id fails closed rather than opening somebody else
4. asking for no case keeps opening the stay it always did
"""

import json
import tempfile
import unittest
from pathlib import Path

from oslian_demo.app import create_app
from oslian_demo.monitoring.api import StreamService
from oslian_demo.monitoring.build_retrospective_cases import CASE_VARIABLES
from oslian_demo.monitoring.source import (RETRO_CASE_PATHS, RetrospectiveCaseSource,
                                           is_known_source_key, make_source, split_source_key)
from oslian_demo.monitoring.ward import WARD_FIXTURE, build_payload


def ward_ids():
    document = json.loads(WARD_FIXTURE.read_text(encoding="utf-8"))
    return [p["id"] for p in document["ward"]["patients"]]


class CaseFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(RETRO_CASE_PATHS[1].read_text(encoding="utf-8"))
        cls.cases = {c["id"]: c for c in cls.document["cases"]}

    def test_every_ward_stay_has_a_case(self):
        # The board's whole point is that a row leads somewhere. A stay without a
        # case is a row that looks clickable and is not.
        missing = [i for i in ward_ids() if i not in self.cases]
        self.assertEqual(missing, [], f"ward rows with no case: {missing}")

    def test_the_document_declares_what_it_is(self):
        self.assertEqual(self.document["schema_version"], "oslian.retrospective-cases.v1")
        self.assertTrue(self.document["retrospective"])
        self.assertTrue(self.document["not_live"])
        self.assertIn("eICU", self.document["source_metadata"]["dataset"])
        # The omitted vitals are stated, so their absence is a scope and not a gap.
        self.assertIn("variable_scope", self.document)
        for absent in ("heart rate", "oxygen saturation", "respiratory rate"):
            self.assertIn(absent, self.document["variable_scope"].lower())

    def test_a_case_carries_only_codeable_measured_variables(self):
        allowed = set(CASE_VARIABLES) | {"hours_since_onset"}
        for case in self.cases.values():
            for event in case["observation_events"]:
                extra = set(event["observation"]) - allowed
                self.assertEqual(extra, set(), f"{case['id']} carries {sorted(extra)}")
                # An event with no value is a row that would emit no resource.
                self.assertGreater(len(event["observation"]), 1, case["id"])

    def test_a_case_carries_no_label_forecast_or_outcome(self):
        """Structural, not a word scan.

        These cases were built from the head-to-head anchor file, which also
        holds the labels, the persistence baseline and the forecast targets. None
        of that may travel with a case, and an allowlist catches a new key that a
        future builder change would otherwise leak silently -- the same guard the
        ward fixture carries.
        """
        allowed = {"id", "display_name", "title", "case_source", "care_unit",
                   "source_metadata", "case_context", "observation_events"}
        for case in self.cases.values():
            extra = set(case) - allowed
            self.assertEqual(extra, set(), f"{case['id']} carries {sorted(extra)}")
            self.assertEqual(set(case["case_context"]) - {
                "age", "gender", "unit_type", "source_admission_label"}, set(), case["id"])
        document_allowed = {"schema_version", "retrospective", "not_live", "note",
                            "variable_scope", "source_metadata", "cases"}
        self.assertEqual(set(self.document) - document_allowed, set())

    def test_no_case_carries_a_measurement_age(self):
        # A trajectory row is the measurement time, so an age would be zero for
        # every value -- and a field that is always zero reads as a recorded age.
        for case in self.cases.values():
            for event in case["observation_events"]:
                stale = [k for k in event["observation"] if k.endswith("_age_hr")]
                self.assertEqual(stale, [], case["id"])

    def test_the_demographics_are_the_ones_the_dataset_recorded(self):
        # eICU writes ">89" for the oldest patients; coercing that to a number
        # would invent a birth year the dataset deliberately withheld.
        with_context = [c for c in self.cases.values() if c["case_context"].get("age")]
        self.assertTrue(with_context)
        for case in with_context:
            self.assertIsInstance(case["case_context"]["age"], str)

    def test_the_case_observations_are_the_ones_the_board_read(self):
        """The board placed each row on a value; opening the row must show that
        same value, not a differently built copy of the stay."""
        payload = build_payload(display_cap=200)
        rows = {r["patientId"]: r for b in payload["bands"] for r in b["rows"]}
        checked = 0
        for patient_id, row in rows.items():
            board_series = row["series"]["creatinine"]
            if not board_series:
                continue
            case = self.cases[patient_id]
            case_series = [[e["observation"]["hours_since_onset"], e["observation"]["creatinine"]]
                           for e in case["observation_events"]
                           if "creatinine" in e["observation"]]
            self.assertEqual(case_series, board_series, patient_id)
            checked += 1
        self.assertGreater(checked, 20, "too few stays compared to prove anything")


class SourceSelectionTests(unittest.TestCase):
    def test_no_case_id_opens_the_stay_it_always_did(self):
        self.assertEqual(make_source("retrospective").patient_id, "eicu-demo-2677807")
        self.assertEqual(RetrospectiveCaseSource().patient_id, "eicu-demo-2677807")

    def test_a_case_id_opens_that_stay(self):
        target = ward_ids()[0]
        source = make_source(f"retrospective:{target}")
        self.assertEqual(source.patient_id, target)
        self.assertEqual(source.describe()["case_key"], f"retrospective:{target}")

    def test_an_unknown_case_id_fails_closed(self):
        # Never quietly open a different stay: a wrong id is a broken link.
        with self.assertRaises(KeyError):
            make_source("retrospective:eicu-demo-nobody")

    def test_only_the_retrospective_source_takes_a_case(self):
        with self.assertRaises(KeyError):
            make_source("synthetic:eicu-demo-1057910")
        self.assertFalse(is_known_source_key("synthetic:anything"))

    def test_the_key_splits_back_into_source_and_case(self):
        self.assertEqual(split_source_key("retrospective:x"), ("retrospective", "x"))
        self.assertEqual(split_source_key("retrospective"), ("retrospective", None))
        self.assertEqual(split_source_key("retrospective:"), ("retrospective", None))

    def test_every_case_converts_to_resources_the_registry_can_map(self):
        """Run the resources through the real mapper, not a code list.

        A case whose observations arrive unmapped would show a stay the ledger
        keeps but the page cannot chart -- which is exactly the failure the
        board's row would be blamed for.
        """
        from datetime import datetime
        from oslian_demo.monitoring.mapping import map_observation
        mapped = 0
        for case_id in ward_ids()[:5]:
            for event in make_source(f"retrospective:{case_id}").events():
                resource = event.resource
                period = resource.get("effectivePeriod")
                when = datetime.fromisoformat(
                    (period["end"] if period else resource["effectiveDateTime"]).replace("Z", "+00:00"))
                outcome = map_observation(resource, when, 1.0 if period else None)
                self.assertTrue(outcome.mapped,
                                f"{case_id}: {resource['code']['coding'][0]['code']} "
                                f"did not map ({outcome.failure} {outcome.detail})")
                mapped += 1
        self.assertGreater(mapped, 100)


class RestartTests(unittest.TestCase):
    """A session opened on one stay must come back as that stay."""

    def setUp(self):
        # An explicit ledger file for this test alone. A restart is a second
        # service over the same file; nothing another test module set in the
        # process environment (an in-memory ledger, say) can hand the second app
        # a different ledger, and nothing here touches the developer's own.
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.ledger = Path(tmp.name) / "ledger.sqlite3"

    def _client(self):
        service = StreamService(db_path=self.ledger)
        self.addCleanup(service.close)
        return create_app(stream_service=service).test_client()

    def test_a_restored_per_stay_session_keeps_its_own_case(self):
        # Forty cases share one source system. Rebuilding a stored session by
        # system alone hands every one of them the same case, so a stay opened
        # before a restart comes back as somebody else -- with the right patient
        # id on the record and another patient's observations behind it.
        target = ward_ids()[3]
        opened = self._client().post(
            "/api/demo-streams", json={"source": f"retrospective:{target}"}).get_json()
        self.assertEqual(opened["patient_id"], target)

        # A second app over the same ledger is what a restart looks like.
        restored = self._client()
        state = restored.get(f"/api/demo-streams/{opened['session_id']}").get_json()
        self.assertEqual(state["patient_id"], target)
        source = state["source"]
        self.assertEqual(source["case_id"], target, "restored onto the wrong stay")
        self.assertEqual(source["display_name"],
                         make_source(f"retrospective:{target}").describe()["display_name"])

    def test_the_default_retrospective_session_still_restores(self):
        opened = self._client().post(
            "/api/demo-streams", json={"source": "retrospective"}).get_json()
        state = self._client().get(
            f"/api/demo-streams/{opened['session_id']}").get_json()
        self.assertEqual(state["patient_id"], "eicu-demo-2677807")

    def test_the_synthetic_session_still_restores(self):
        opened = self._client().post(
            "/api/demo-streams", json={"source": "synthetic"}).get_json()
        state = self._client().get(
            f"/api/demo-streams/{opened['session_id']}").get_json()
        self.assertEqual(state["patient_id"], opened["patient_id"])


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_a_ward_stay_opens_its_own_session(self):
        target = ward_ids()[0]
        response = self.client.post("/api/demo-streams", json={"source": f"retrospective:{target}"})
        self.assertIn(response.status_code, (200, 201))
        self.assertEqual(response.get_json()["patient_id"], target)

    def test_two_stays_get_two_sessions(self):
        first, second = ward_ids()[0], ward_ids()[1]
        a = self.client.post("/api/demo-streams", json={"source": f"retrospective:{first}"}).get_json()
        b = self.client.post("/api/demo-streams", json={"source": f"retrospective:{second}"}).get_json()
        self.assertNotEqual(a["session_id"], b["session_id"])
        self.assertNotEqual(a["patient_id"], b["patient_id"])

    def test_reopening_a_stay_is_idempotent(self):
        target = ward_ids()[2]
        a = self.client.post("/api/demo-streams", json={"source": f"retrospective:{target}"}).get_json()
        b = self.client.post("/api/demo-streams", json={"source": f"retrospective:{target}"}).get_json()
        self.assertEqual(a["session_id"], b["session_id"])

    def test_an_unknown_case_is_404_not_500(self):
        response = self.client.post("/api/demo-streams", json={"source": "retrospective:eicu-demo-nobody"})
        self.assertEqual(response.status_code, 404)
        self.assertIn("nobody", response.get_json()["error"])

    def test_an_unknown_source_is_still_400(self):
        self.assertEqual(self.client.post("/api/demo-streams", json={"source": "nope"}).status_code, 400)

    def test_the_sources_listing_stays_two(self):
        # Forty stays do not become forty sources; the board is where one is picked.
        keys = [s["source_key"] for s in self.client.get("/api/demo-streams/sources").get_json()["sources"]]
        self.assertEqual(sorted(keys), ["retrospective", "synthetic"])


if __name__ == "__main__":
    unittest.main()
