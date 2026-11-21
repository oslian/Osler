"""Medication Safety may read a monitored patient's measured values, by request, and never a forecast."""

import copy
import os
import unittest

os.environ["OSLIAN_STREAM_LOG"] = "off"

from oslian_demo import create_app  # noqa: E402
from oslian_demo.config import AppConfig  # noqa: E402
from oslian_demo.medication import agent  # noqa: E402
from oslian_demo.monitoring import measured as measured_module  # noqa: E402
from oslian_demo.monitoring.api import StreamService  # noqa: E402
from oslian_demo.monitoring.forecast_runtime import ForecastRuntime, RuntimeConfig  # noqa: E402
from oslian_demo.monitoring.measured import MEASURED_FIELDS, measured_inputs  # noqa: E402

PATIENT = "syn-pat-0001"


class MeasuredInputTests(unittest.TestCase):
    def setUp(self):
        self.service = StreamService(":memory:", runtime=ForecastRuntime(RuntimeConfig("detached", "http://127.0.0.1:1")))
        self.client = create_app(AppConfig.from_env({}), self.service).test_client()
        sample = self.client.get("/api/cases").get_json()["cases"][0]
        self.fields = {k: v for k, v in sample.items() if k not in ("id", "title")}
        self.sid = self.client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()["session_id"]

    def tearDown(self):
        self.service.close()

    def step(self, count):
        for _ in range(count):
            self.client.post(f"/api/demo-streams/{self.sid}/step")

    def analyze(self, **extra):
        return self.client.post("/api/analyze", json={"fields": copy.deepcopy(self.fields), "use_openfda": False, **extra})

    def test_without_the_request_the_case_is_analysed_as_written(self):
        self.step(30)
        bundle = self.analyze().get_json()
        self.assertIsNone(bundle["measured_inputs"])
        self.assertEqual(bundle["fields"]["vitals"], self.fields["vitals"])

    def test_measured_values_replace_the_matching_case_values_with_provenance(self):
        self.step(30)                                    # through the 12:00Z results and one 14:00Z blood pressure
        bundle = self.analyze(stream_patient_id=PATIENT).get_json()
        measured = bundle["measured_inputs"]
        self.assertFalse(measured["forecasts_used"])
        self.assertEqual(measured["as_of"], "2026-03-14T14:00:00.000Z")
        used = {item["field"]: item for item in measured["used"]}
        self.assertEqual(set(used), {"heart_rate", "spo2", "potassium"})
        self.assertEqual((used["heart_rate"]["value"], used["heart_rate"]["replaced_case_value"]), (89.0, 112))
        self.assertEqual(used["heart_rate"]["reference"], "Observation/syn-obs-heart_rate-h06/_history/1")
        self.assertEqual(used["potassium"]["replaced_case_value"], None)
        self.assertEqual(bundle["fields"]["vitals"], {"sbp": 88, "heart_rate": 89.0, "spo2": 94.0},
                         "the systolic pressure is the case's own: the stream has no systolic variable")
        self.assertEqual(bundle["fields"]["labs"], {"potassium": 4.6})
        self.assertEqual(bundle["fields"]["egfr"], self.fields["egfr"], "kidney function is never taken from the stream")
        self.assertIn("Measured values · 3 from the monitored patient", [step["title"] for step in bundle["trace"]])

    def test_a_measured_value_can_change_a_patient_flag(self):
        self.fields["vitals"] = {"heart_rate": 135}
        before = self.analyze().get_json()["result"]["patient"]
        self.step(30)
        after = self.analyze(stream_patient_id=PATIENT).get_json()["result"]["patient"]
        flags = lambda patient: {flag if isinstance(flag, str) else flag["flag"] for flag in patient["flags"]}
        self.assertIn("tachycardia", flags(before))
        self.assertNotIn("tachycardia", flags(after), "the measured heart rate is 89")

    def test_a_value_older_than_the_limit_is_not_used(self):
        self.step(6)                                     # the 08:00Z vitals; potassium is still the 06:00Z baseline
        early = {item["field"]: item for item in measured_inputs(self.service, PATIENT)["used"]}
        self.assertEqual((early["potassium"]["effective_time"], early["potassium"]["age_hours"]), ("2026-03-14T06:00:00.000Z", 2.0))
        limit = measured_module.MAX_AGE_HOURS
        measured_module.MAX_AGE_HOURS = 1.0              # the vitals are fresh; the baseline potassium is 2 h old
        try:
            stale = measured_inputs(self.service, PATIENT)
            bundle = self.analyze(stream_patient_id=PATIENT).get_json()
        finally:
            measured_module.MAX_AGE_HOURS = limit
        self.assertEqual({item["field"] for item in stale["used"]}, {"heart_rate", "spo2"})
        self.assertEqual([(item["variable"], item["reason"]) for item in stale["not_used"]], [("potassium", "stale")])
        self.assertNotIn("potassium", bundle["fields"]["labs"])

    def test_an_unknown_patient_is_refused(self):
        self.assertEqual(self.analyze(stream_patient_id="nobody").status_code, 404)
        self.assertEqual(self.client.post("/api/analyze_stream", json={"fields": self.fields, "stream_patient_id": "nobody"}).status_code, 404)

    def test_the_streamed_route_carries_the_same_measured_inputs(self):
        self.step(30)
        response = self.client.post("/api/analyze_stream", json={"fields": self.fields, "stream_patient_id": PATIENT,
                                                                 "orchestrator": "deterministic"})
        import json
        final = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line][-1]
        self.assertEqual(final["type"], "final")
        self.assertEqual(len(final["bundle"]["measured_inputs"]["used"]), 3)

    def test_only_observed_variables_the_engine_already_reads_are_mapped(self):
        self.assertEqual(MEASURED_FIELDS, {"heart_rate": ("vitals", "heart_rate", "beats/min"),
                                           "o2sat": ("vitals", "spo2", "%"), "potassium": ("labs", "potassium", "mEq/L")})
        import inspect
        source = inspect.getsource(measured_module)
        for forbidden in ("forecast_point", "cells_for_run", "latest_completed_run", "ForecastRuntime"):
            self.assertNotIn(forbidden, source, "a forecast must never reach the medication engine")

    def test_apply_measured_leaves_the_case_untouched_and_other_fields_alone(self):
        fields = {"age": 60, "vitals": {"sbp": 100, "heart_rate": 120}, "labs": {}}
        measured = {"used": [{"group": "vitals", "field": "heart_rate", "value": 80.0}]}
        out = agent.apply_measured(fields, measured)
        self.assertEqual(out["vitals"], {"sbp": 100, "heart_rate": 80.0})
        self.assertEqual(fields["vitals"], {"sbp": 100, "heart_rate": 120}, "the caller's fields are not edited")
        self.assertEqual(measured["used"][0]["replaced_case_value"], 120)
        self.assertIs(agent.apply_measured(fields, None), fields)


if __name__ == "__main__":
    unittest.main()
