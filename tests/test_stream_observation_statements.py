"""What an observation says about its own value travels with the state, untouched.

`Signals & forecasts` flags a row only on the observation's own statement: a
FHIR ``interpretation`` code or a ``referenceRange`` the observation supplies.
The server passes both through on the state and the historical timeline and
decides nothing itself; with neither present the lists are empty and the page
says the range was not supplied.
"""

import unittest

from flask import Flask

from oslian_demo.monitoring.api import StreamService, create_blueprint
from oslian_demo.monitoring.timeline import attach_observation_statements, observation_statements, variable_state
from oslian_demo.monitoring.wholebody import RuntimeConfig, WholeBodyRuntime

RESOURCE = {
    "resourceType": "Observation", "id": "obs-1", "status": "final",
    "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                    "code": "H", "display": "High"}]}],
    "referenceRange": [
        {"low": {"value": 0.6, "unit": "mg/dL", "code": "mg/dL"}, "high": {"value": 1.2, "unit": "mg/dL"},
         "text": "0.6-1.2 mg/dL"},
        {"low": {"value": 0.5, "unit": "mg/dL"}, "high": {"value": 1.0, "unit": "mg/dL"},
         "appliesTo": [{"text": "female"}]},
    ],
}


class ObservationStatementsTests(unittest.TestCase):
    def test_interpretation_and_reference_range_pass_through(self):
        got = observation_statements(RESOURCE)
        self.assertEqual(got["interpretation"], [{
            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "H", "display": "High",
            "provenance": None}])
        self.assertEqual(len(got["reference_range"]), 2)
        general, qualified = got["reference_range"]
        self.assertEqual((general["low"], general["low_unit"], general["high"], general["high_unit"]),
                         (0.6, "mg/dL", 1.2, "mg/dL"))
        self.assertEqual(general["text"], "0.6-1.2 mg/dL")
        self.assertFalse(general["applies_to"])
        # a range qualified by population is marked as such, not dropped and not chosen here
        self.assertTrue(qualified["applies_to"])

    def test_an_observation_with_neither_yields_empty_lists(self):
        self.assertEqual(observation_statements({"resourceType": "Observation", "id": "x"}),
                         {"interpretation": [], "reference_range": []})
        self.assertEqual(observation_statements(None), {"interpretation": [], "reference_range": []})

    def test_malformed_elements_are_ignored_not_invented(self):
        got = observation_statements({
            "interpretation": ["not-a-concept", {"coding": [{"display": "no code"}]}],
            "referenceRange": ["not-a-range", {"low": {"unit": "mg/dL"}, "high": "12"}],
        })
        self.assertEqual(got["interpretation"], [])
        self.assertEqual(got["reference_range"], [{"low": None, "low_unit": "mg/dL", "high": None, "high_unit": None,
                                                   "text": None, "applies_to": False, "age": False}])

    def test_attach_joins_each_variables_latest_raw_resource(self):
        class Repo:
            calls = []

            def raw_by_reference(self, source_system, resource_type, resource_id, version_id):
                self.calls.append((source_system, resource_type, resource_id, version_id))
                return {"resource": RESOURCE} if resource_id == "obs-1" else None

        variables = {"creatinine": {"resource_id": "obs-1", "version_id": "2"},
                     "sodium": {"resource_id": "obs-9", "version_id": "1"}}
        attach_observation_statements(Repo(), "src", variables)
        self.assertEqual(Repo.calls, [("src", "Observation", "obs-1", "2"), ("src", "Observation", "obs-9", "1")])
        self.assertEqual(variables["creatinine"]["interpretation"][0]["code"], "H")
        self.assertEqual(variables["sodium"], {"resource_id": "obs-9", "version_id": "1",
                                               "interpretation": [], "reference_range": []})

    def test_variable_state_carries_the_value_as_the_source_stated_it(self):
        # a supplied range is written in the source's unit, so the state has to carry
        # the value in that unit alongside the canonical one
        rows = [{"variable": "temperature", "value": 37.2, "unit": "Cel", "effective_time": "2026-03-14T06:00:00Z",
                 "received_at": "2026-03-14T06:00:01Z", "resource_id": "t1", "version_id": "1",
                 "input_value": 99.0, "input_unit": "[degF]"}]
        state = variable_state(rows, "2026-03-14T07:00:00Z")["temperature"]
        self.assertEqual((state["value"], state["unit"]), (37.2, "Cel"))
        self.assertEqual((state["input_value"], state["input_unit"]), (99.0, "[degF]"))


class StateRouteTests(unittest.TestCase):
    def setUp(self):
        self.service = StreamService(":memory:", runtime=WholeBodyRuntime(RuntimeConfig("fixture", None, "python", None)))
        app = Flask(__name__)
        app.register_blueprint(create_blueprint(self.service))
        self.client = app.test_client()

    def tearDown(self):
        self.service.close()

    def test_the_state_and_the_timeline_carry_the_statements_for_every_variable(self):
        session = self.client.post("/api/demo-streams").get_json()
        sid, pid = session["session_id"], session["patient_id"]
        for _ in range(6):
            self.client.post(f"/api/demo-streams/{sid}/step")
        state = self.client.get(f"/api/patients/{pid}/state?session_id={sid}").get_json()
        self.assertTrue(state["variables"])
        for variable in state["variables"].values():
            self.assertIsInstance(variable["interpretation"], list)
            self.assertIsInstance(variable["reference_range"], list)
            self.assertIn("input_unit", variable)
        timeline = self.client.get(f"/api/patients/{pid}/timeline?session_id={sid}").get_json()
        for variable in timeline["state"].values():
            self.assertIsInstance(variable["interpretation"], list)
            self.assertIsInstance(variable["reference_range"], list)


if __name__ == "__main__":
    unittest.main()
