"""The demo's reference ranges: one registry, cited, written into the fixture, passed through.

Every interval the synthetic stream supplies is written in exactly one place
(``reference_ranges.py``) with its public source, and the fixture builder copies
it into the FHIR ``Observation.referenceRange`` of every laboratory observation
it applies to, with an ``interpretation`` derived from that same interval that
says so. The stream and the API pass both through untouched, so the page reads
only what the observation carries and never a number of its own.
"""

import json
import unittest

from flask import Flask

from oslian_demo.config import FIXTURE_DIR
from oslian_demo.monitoring import build_synthetic_fixture as builder
from oslian_demo.monitoring.api import StreamService, create_blueprint
from oslian_demo.monitoring.reference_ranges import (DEMO_REFERENCE_RANGES, INTERPRETATION_PROVENANCE,
                                                     NOT_SUPPLIED, V3_INTERPRETATION, provenance_document)
from oslian_demo.monitoring.wholebody import RuntimeConfig, WholeBodyRuntime

FIXTURE = FIXTURE_DIR / "synthetic_patient_v1.json"
LOINC_TO_VARIABLE = {"2951-2": "sodium", "2823-3": "potassium", "2075-0": "chloride", "1963-8": "bicarbonate",
                     "3094-0": "bun", "2524-7": "lactate", "6690-2": "wbc", "777-3": "platelets",
                     "2160-0": "creatinine", "718-7": "hemoglobin", "4544-3": "hematocrit", "2345-7": "glucose",
                     "14749-6": "glucose", "8867-4": "heart_rate", "9279-1": "respiratory_rate",
                     "59408-5": "o2sat", "8310-5": "temperature", "8478-0": "map", "9187-6": "urine_output"}


def _observations(document):
    return [e["resource"] for e in document["initial_bundle"]["entry"] if e["resource"]["resourceType"] == "Observation"] \
        + [e["resource"] for e in document["events"]]


class RegistryTests(unittest.TestCase):
    def test_every_interval_is_cited_and_well_formed(self):
        for variable, rr in DEMO_REFERENCE_RANGES.items():
            self.assertEqual(rr.variable, variable)
            self.assertLess(rr.low, rr.high, variable)
            self.assertTrue(rr.unit and rr.unit_code and rr.source_unit, variable)
            self.assertTrue(rr.source_url.startswith("https://medlineplus.gov/ency/article/"), variable)
            self.assertRegex(rr.source_review_date, r"^\d{4}-\d{2}-\d{2}$")
            self.assertTrue(rr.source_title and rr.applicable_age and rr.applicable_sex and rr.context, variable)
            self.assertTrue(rr.citable)
            self.assertEqual(rr.origin, "external_reference_cited_by_demo_fixture")
            prov = rr.provenance()
            for key in ("source_organization", "retrieval_date", "applicability", "unit_note"):
                self.assertTrue(prov[key], f"{variable}: {key}")

    def test_a_variable_is_either_supplied_or_deliberately_not_and_never_both(self):
        self.assertEqual(set(DEMO_REFERENCE_RANGES) & set(NOT_SUPPLIED), set())
        # every laboratory or vital sign the synthetic stream carries has a decision written down
        for variable in ("creatinine", "hemoglobin", "hematocrit", "glucose",
                         "heart_rate", "respiratory_rate", "o2sat", "temperature", "map", "urine_output"):
            self.assertIn(variable, NOT_SUPPLIED)
            self.assertTrue(NOT_SUPPLIED[variable])

    def test_interpretation_is_arithmetic_with_inclusive_limits(self):
        bun = DEMO_REFERENCE_RANGES["bun"]
        self.assertEqual(bun.interpret(5.9), "L")
        self.assertEqual(bun.interpret(6.0), "N")      # on the line is inside
        self.assertEqual(bun.interpret(20.0), "N")
        self.assertEqual(bun.interpret(20.1), "H")
        concept = bun.fhir_interpretation(26.0)
        self.assertEqual(concept["coding"][0]["system"], V3_INTERPRETATION)
        self.assertEqual(concept["coding"][0]["code"], "H")
        self.assertEqual(concept["text"], INTERPRETATION_PROVENANCE)

    def test_the_fhir_element_carries_the_unit_and_names_its_source(self):
        element = DEMO_REFERENCE_RANGES["sodium"].fhir_reference_range()
        self.assertEqual(element["low"], {"value": 135.0, "unit": "mmol/L", "system": "http://unitsofmeasure.org", "code": "mmol/L"})
        self.assertEqual(element["high"]["value"], 145.0)
        self.assertIn("MedlinePlus", element["text"])
        self.assertIn("not a clinical threshold", element["text"])
        # the source stated mEq/L; the identity to mmol/L is written down, not assumed
        self.assertIn("mEq/L", DEMO_REFERENCE_RANGES["sodium"].unit_note)


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.observations = _observations(cls.document)

    def _variable(self, resource):
        return LOINC_TO_VARIABLE.get(resource["code"]["coding"][0]["code"])

    def test_every_supplied_laboratory_value_carries_the_registry_interval_and_a_derived_reading(self):
        seen = set()
        for resource in self.observations:
            variable = self._variable(resource)
            if variable not in DEMO_REFERENCE_RANGES or "valueQuantity" not in resource:
                continue
            rr = DEMO_REFERENCE_RANGES[variable]
            self.assertEqual(resource["referenceRange"], [rr.fhir_reference_range()], resource["id"])
            self.assertEqual(resource["interpretation"], [rr.fhir_interpretation(resource["valueQuantity"]["value"])], resource["id"])
            self.assertEqual(resource["valueQuantity"]["unit"], rr.unit, resource["id"])
            seen.add(variable)
        self.assertEqual(seen, set(DEMO_REFERENCE_RANGES))

    def test_a_variable_with_no_adopted_interval_carries_neither_element(self):
        for resource in self.observations:
            variable = self._variable(resource)
            if variable in NOT_SUPPLIED or variable is None:
                self.assertNotIn("referenceRange", resource, resource["id"])
                self.assertNotIn("interpretation", resource, resource["id"])

    def test_the_amended_potassium_reads_high_then_normal(self):
        v1 = next(r for r in self.observations if r["id"] == "syn-obs-potassium-h12" and r["meta"]["versionId"] == "3")
        v2 = next(r for r in self.observations if r["id"] == "syn-obs-potassium-h12" and r["meta"]["versionId"] == "4")
        self.assertEqual(v1["interpretation"][0]["coding"][0]["code"], "H")
        self.assertEqual(v2["interpretation"][0]["coding"][0]["code"], "N")

    def test_the_stream_shows_above_below_and_within_at_its_last_values(self):
        # The demo has to make the states visible, not only possible: the last
        # laboratory values of the stream must include one above, one below and
        # one within its supplied interval.
        last = {}
        for resource in self.observations:
            variable = self._variable(resource)
            if variable in DEMO_REFERENCE_RANGES and "valueQuantity" in resource:
                key = resource["effectiveDateTime"], resource["meta"]["versionId"]
                if variable not in last or key > last[variable][0]:
                    last[variable] = (key, resource["interpretation"][0]["coding"][0]["code"])
        readings = {v: code for v, (_, code) in last.items()}
        self.assertEqual(readings["bun"], "H")
        self.assertEqual(readings["bicarbonate"], "L")
        self.assertEqual(readings["sodium"], "N")
        self.assertEqual(readings["chloride"], "N")

    def test_the_document_carries_the_provenance_block_and_says_the_reading_is_derived(self):
        self.assertEqual(self.document["reference_ranges"], provenance_document())
        self.assertEqual(self.document["reference_ranges"]["interpretation_provenance"], INTERPRETATION_PROVENANCE)
        self.assertTrue(any(INTERPRETATION_PROVENANCE in note for note in self.document["notes"]))

    def test_the_fixture_is_reproducible_from_the_builder_byte_for_byte(self):
        text = json.dumps(builder.build(), indent=1, ensure_ascii=False) + "\n"
        self.assertEqual(text, FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(builder.build(), builder.build())


class PassthroughTests(unittest.TestCase):
    def setUp(self):
        self.service = StreamService(":memory:", runtime=WholeBodyRuntime(RuntimeConfig("fixture", None, "python", None)))
        app = Flask(__name__)
        app.register_blueprint(create_blueprint(self.service))
        self.client = app.test_client()

    def tearDown(self):
        self.service.close()

    def test_the_state_carries_the_interval_the_reading_and_its_provenance(self):
        session = self.client.post("/api/demo-streams").get_json()
        sid, pid = session["session_id"], session["patient_id"]
        state = self.client.get(f"/api/patients/{pid}/state?session_id={sid}").get_json()
        bun = state["variables"]["bun"]
        self.assertEqual(bun["reference_range"], [{"low": 6.0, "low_unit": "mg/dL", "high": 20.0, "high_unit": "mg/dL",
                                                   "text": DEMO_REFERENCE_RANGES["bun"].fhir_reference_range()["text"],
                                                   "applies_to": False, "age": False}])
        self.assertEqual(bun["interpretation"][0]["code"], "H")
        self.assertEqual(bun["interpretation"][0]["provenance"], INTERPRETATION_PROVENANCE)
        self.assertEqual((bun["input_value"], bun["input_unit"], bun["derivation"]), (bun["value"], "mg/dL", "identity"))
        creatinine = state["variables"]["creatinine"]
        self.assertEqual((creatinine["reference_range"], creatinine["interpretation"]), ([], []))
        timeline = self.client.get(f"/api/patients/{pid}/timeline?session_id={sid}").get_json()
        self.assertEqual(timeline["state"]["bun"]["reference_range"], bun["reference_range"])

    def test_the_live_event_carries_the_statements_too(self):
        # A page that has already loaded learns of a new value from the event alone,
        # so the event has to say what the observation says about itself.
        session = self.client.post("/api/demo-streams").get_json()
        sid = session["session_id"]
        for _ in range(40):
            self.client.post(f"/api/demo-streams/{sid}/step")
        events = [e for e in self.service.repo.events_after(sid, 0, limit=5000) if e["event_type"] == "observation.normalized"]
        bun = [e for e in events if any(v["variable"] == "bun" for v in e["payload"]["values"])]
        self.assertTrue(bun, "no bun value delivered in forty steps")
        payload = bun[-1]["payload"]
        self.assertEqual(payload["reference_range"][0]["low"], 6.0)
        self.assertEqual(payload["interpretation"][0]["provenance"], INTERPRETATION_PROVENANCE)
        value = next(v for v in payload["values"] if v["variable"] == "bun")
        self.assertEqual((value["input_unit"], value["derivation"]), ("mg/dL", "identity"))
        vitals = [e for e in events if any(v["variable"] == "heart_rate" for v in e["payload"]["values"])]
        self.assertEqual((vitals[-1]["payload"]["reference_range"], vitals[-1]["payload"]["interpretation"]), ([], []))


if __name__ == "__main__":
    unittest.main()
