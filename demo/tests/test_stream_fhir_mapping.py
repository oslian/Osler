"""FHIR R4 fixture validation and code/unit mapping tests."""

from datetime import datetime, timezone
import json
import unittest

from oslian_demo.config import FIXTURE_DIR
from oslian_demo.monitoring import fhir, mapping
from oslian_demo.monitoring.source import MockFhirSource
from oslian_demo.monitoring.wholebody import load_contract_fixture

AT = datetime(2026, 3, 14, 8, 0, tzinfo=timezone.utc)


def _obs(code, value=None, unit=None, unit_system=mapping.UCUM, **extra):
    resource = {
        "resourceType": "Observation", "id": "t", "meta": {"versionId": "1", "lastUpdated": "2026-03-14T08:00:00Z"},
        "status": "final", "code": {"coding": [{"system": mapping.LOINC, "code": code}]},
        "subject": {"reference": "Patient/p"}, "effectiveDateTime": "2026-03-14T08:00:00Z",
    }
    if value is not None:
        quantity = {"value": value}
        if unit is not None:
            quantity["unit"] = unit
            if unit_system:
                quantity["system"] = unit_system
                quantity["code"] = unit
        resource["valueQuantity"] = quantity
    resource.update(extra)
    return resource


class FixtureValidationTests(unittest.TestCase):
    def setUp(self):
        self.source = MockFhirSource()

    def test_every_fixture_resource_is_structurally_valid_r4(self):
        resources = fhir.validate_bundle(self.source.initial_bundle())
        self.assertEqual(self.source.initial_bundle()["type"], "searchset")
        types = [r["resourceType"] for r in resources]
        self.assertEqual(types[:2], ["Patient", "Encounter"])
        for resource in resources:
            if resource["resourceType"] == "Patient":
                fhir.validate_patient(resource)
            elif resource["resourceType"] == "Encounter":
                fhir.validate_encounter(resource)
            else:
                fhir.validate_observation(resource)
        for event in self.source.events():
            effective = fhir.validate_observation(event.resource)
            self.assertIsNotNone(effective.effective)
            self.assertIn("versionId", event.resource["meta"])
            self.assertIn("issued", event.resource)

    def test_fixture_is_synthetic_and_free_of_identifiers(self):
        text = (FIXTURE_DIR / "synthetic_patient_v1.json").read_text(encoding="utf-8")
        document = json.loads(text)
        self.assertTrue(document["synthetic"])
        self.assertFalse(document["phi"])
        for resource in [e["resource"] for e in document["events"]] + [
            e["resource"] for e in document["initial_bundle"]["entry"]
        ]:
            self.assertTrue(resource["id"].startswith("syn-"), resource["id"])
        patient = document["initial_bundle"]["entry"][0]["resource"]
        self.assertNotIn("birthDate", patient)
        self.assertNotIn("address", patient)
        self.assertNotIn("telecom", patient)

    def test_fixture_covers_required_scenarios(self):
        scenarios = {e.scenario for e in self.source.events()}
        for needed in ("duplicate_delivery", "amended_v2", "out_of_order", "missing_ucum_unit",
                       "data_absent_reason", "unmapped_code", "entered_in_error_v2"):
            self.assertIn(needed, scenarios)

    def test_obs6_and_timezone_violations_are_rejected(self):
        with self.assertRaises(fhir.FhirValidationError) as ctx:
            fhir.validate_observation(_obs("8867-4", 80, "/min", dataAbsentReason={"coding": [{"code": "unknown"}]}))
        self.assertIn("obs-6", str(ctx.exception))
        with self.assertRaises(fhir.FhirValidationError):
            fhir.validate_observation(_obs("8867-4", 80, "/min", effectiveDateTime="2026-03-14T08:00:00"))
        with self.assertRaises(fhir.FhirValidationError):
            fhir.validate_observation(_obs("8867-4", 80, "/min", status="not-a-status"))

    def test_period_effective_time_is_period_end(self):
        resource = _obs("9187-6", 120, "mL")
        del resource["effectiveDateTime"]
        resource["effectivePeriod"] = {"start": "2026-03-14T06:00:00Z", "end": "2026-03-14T08:00:00Z"}
        effective = fhir.validate_observation(resource)
        self.assertEqual(effective.period_hours, 2.0)
        self.assertEqual(fhir.iso(effective.effective), "2026-03-14T08:00:00.000Z")


class MappingTests(unittest.TestCase):
    def test_identity_and_unit_conversions(self):
        cases = [
            (_obs("8867-4", 92, "/min"), "heart_rate", 92.0, "beats/min"),
            (_obs("8310-5", 98.6, "[degF]"), "temperature", 37.0, "degC"),
            (_obs("8310-5", 37.4, "Cel"), "temperature", 37.4, "degC"),
            (_obs("14749-6", 10.0, "mmol/L"), "glucose", 180.156, "mg/dL"),
            (_obs("14682-9", 88.42, "umol/L"), "creatinine", 1.0, "mg/dL"),
            (_obs("2823-3", 4.4, "mmol/L"), "potassium", 4.4, "mEq/L"),
            (_obs("718-7", 104, "g/L"), "hemoglobin", 10.4, "g/dL"),
            (_obs("777-3", 172, "10*3/uL"), "platelets", 172.0, "K/uL"),
        ]
        for resource, variable, expected, unit in cases:
            outcome = mapping.map_observation(resource, AT, None)
            self.assertTrue(outcome.mapped, (variable, outcome))
            value = outcome.values[0]
            self.assertEqual(value.variable, variable)
            self.assertEqual(value.unit, unit)
            self.assertAlmostEqual(value.value, expected, places=3)

    def test_urine_volume_over_period_becomes_rate(self):
        outcome = mapping.map_observation(_obs("9187-6", 130, "mL"), AT, 2.0)
        self.assertTrue(outcome.mapped)
        self.assertEqual(outcome.values[0].variable, "urine_output")
        self.assertAlmostEqual(outcome.values[0].value, 65.0)
        self.assertEqual(outcome.values[0].unit, "mL/hour")
        self.assertEqual(mapping.map_observation(_obs("9187-6", 130, "mL"), AT, None).failure, mapping.FAIL_SHAPE)

    def test_blood_pressure_panel_components_derive_map(self):
        resource = _obs("85354-9", component=[
            {"code": {"coding": [{"system": mapping.LOINC, "code": "8480-6"}]},
             "valueQuantity": {"value": 120, "unit": "mmHg", "system": mapping.UCUM, "code": "mm[Hg]"}},
            {"code": {"coding": [{"system": mapping.LOINC, "code": "8462-4"}]},
             "valueQuantity": {"value": 60, "unit": "mmHg", "system": mapping.UCUM, "code": "mm[Hg]"}},
        ])
        outcome = mapping.map_observation(resource, AT, None)
        self.assertTrue(outcome.mapped)
        self.assertEqual(outcome.values[0].variable, "map")
        self.assertAlmostEqual(outcome.values[0].value, 80.0)
        self.assertEqual(outcome.values[0].derivation, "map_from_sbp_dbp")

    def test_failures_are_explicit_and_never_guess_from_display(self):
        self.assertEqual(mapping.map_observation(_obs("8302-2", 171, "cm"), AT, None).failure, mapping.FAIL_UNMAPPED_CODE)
        self.assertEqual(mapping.map_observation(_obs("2345-7", 150, "mg/dL", unit_system=None), AT, None).failure,
                         mapping.FAIL_MISSING_UNIT)
        self.assertEqual(mapping.map_observation(_obs("2345-7", 8.0, "mmol/L"), AT, None).failure,
                         mapping.FAIL_UNMAPPED_UNIT)
        absent = _obs("2160-0", dataAbsentReason={"coding": [{"code": "not-performed"}]})
        self.assertEqual(mapping.map_observation(absent, AT, None).failure, mapping.FAIL_DATA_ABSENT)
        display_only = {
            "resourceType": "Observation", "status": "final",
            "code": {"coding": [{"system": "http://example.org/local", "code": "HR", "display": "Heart rate"}]},
        }
        self.assertEqual(mapping.map_observation(display_only, AT, None).failure, mapping.FAIL_UNMAPPED_CODE)
        self.assertEqual(mapping.map_observation(_obs("8867-4", 80, "/min", status="entered-in-error"), AT, None).failure,
                         mapping.FAIL_STATUS)

    def test_registry_is_consistent_with_release_variables(self):
        contract = load_contract_fixture()
        keys = [rule.key for rule in mapping.RULES]
        self.assertEqual(len(keys), len(set(keys)), "duplicate mapping keys")
        release_variables = set(contract["variables"])
        self.assertEqual(len(release_variables), 53)
        self.assertEqual(set(mapping.CANONICAL_UNITS), release_variables)
        for variable, unit in mapping.CANONICAL_UNITS.items():
            self.assertEqual(unit, contract["variable_units"][variable], variable)
        self.assertEqual(set(mapping.known_variables()), release_variables, "every release variable has a mapping rule")
        for rule in mapping.RULES:
            self.assertTrue(rule.display, rule.code)


if __name__ == "__main__":
    unittest.main()
