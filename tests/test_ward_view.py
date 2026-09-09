"""The ward fixture and the /api/ward route: provenance, fail-closed, no runtime."""

import json
import tempfile
import unittest
from pathlib import Path

from oslian_demo.app import create_app
from oslian_demo.monitoring import ward
from oslian_demo.monitoring.ranking import PLAUSIBLE_WEIGHT_KG
from oslian_demo.monitoring.ward import (MAX_DISPLAY_CAP, WARD_FIXTURE, build_payload,
                                        load_ward, to_snapshots)


def _rows(payload):
    """Every row on the board, whichever band it landed in."""
    return [r for b in payload["bands"] for r in b["rows"]] + payload["unrankable"]


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load_ward()

    def test_the_fixture_declares_what_it_is(self):
        self.assertEqual(self.document["schema_version"], "oslian.ward-fixture.v1")
        self.assertTrue(self.document["retrospective"])
        self.assertTrue(self.document["not_live"])
        self.assertFalse(self.document["safety_boundary"]["causal_claim_allowed"])
        self.assertFalse(self.document["safety_boundary"]["clinical_alert"])
        self.assertIn("eICU", self.document["source_metadata"]["dataset"])

    def test_no_patient_carries_an_observation_from_after_its_own_now(self):
        # The board must never be able to show a reader something the clock had
        # not reached; PatientSnapshot enforces it and this proves the data obeys.
        snapshots = to_snapshots(self.document)
        self.assertGreaterEqual(len(snapshots), 20)
        for snapshot in snapshots:
            for series in (snapshot.creatinine, snapshot.urine_output):
                for hour, _ in series:
                    self.assertLessEqual(hour, snapshot.now_hours + 1e-9)

    def test_one_row_per_patient(self):
        ids = [p["id"] for p in self.document["ward"]["patients"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_unscorable_path_is_present_in_the_demo_data(self):
        # Keeping a couple of patients with no creatinine is deliberate: the
        # "not scored" column is part of the contract and must be visible.
        payload = build_payload()
        self.assertTrue(payload["unrankable"])
        for row in payload["unrankable"]:
            self.assertIsNone(row["rank"])
            self.assertTrue(row["unrankableReason"])

    def test_a_patient_row_carries_observations_and_nothing_else(self):
        # Structural, not a word scan: the anchor file this was built from also
        # holds the head-to-head labels and the persistence baseline, and none of
        # that may travel with the board. An allowlist catches a new key that a
        # future builder change would otherwise leak silently.
        allowed = {"id", "display_name", "care_unit", "now_hours",
                   "creatinine", "urine_output", "treatments", "weight_kg",
                   "bun", "potassium", "sodium", "chloride", "bicarbonate"}
        for patient in self.document["ward"]["patients"]:
            extra = set(patient) - allowed
            self.assertEqual(extra, set(), f"{patient['id']} carries {sorted(extra)}")
            self.assertEqual(set(patient["treatments"]), {
                "hist_fluids", "hist_vasopressor", "hist_diuretics",
                "hist_renal_replacement", "hist_nephrotoxin"})

    def test_the_document_carries_no_label_or_forecast_section(self):
        allowed = {"schema_version", "retrospective", "not_live", "note",
                   "source_metadata", "ward", "safety_boundary"}
        self.assertEqual(set(self.document) - allowed, set())
        self.assertEqual(set(self.document["ward"]) - {"label", "variable_units", "patients"}, set())


class WeightTests(unittest.TestCase):
    """The weight column: joined at build time, and absent rather than guessed."""

    @classmethod
    def setUpClass(cls):
        cls.document = load_ward()
        cls.patients = cls.document["ward"]["patients"]

    def test_every_stay_carries_the_key_even_when_it_has_no_weight(self):
        # Written as null, not omitted: a reader of this fixture must be able to
        # tell "no weight on record" from "this builder predates the column".
        for patient in self.patients:
            self.assertIn("weight_kg", patient, patient["id"])

    def test_the_weights_present_are_ones_the_board_will_divide_by(self):
        low, high = PLAUSIBLE_WEIGHT_KG
        for patient in self.patients:
            if patient["weight_kg"] is not None:
                self.assertGreaterEqual(patient["weight_kg"], low, patient["id"])
                self.assertLessEqual(patient["weight_kg"], high, patient["id"])

    def test_the_demo_ward_shows_both_sides_of_the_fail_closed_path(self):
        # Like the unscored column, the "no weight" path is part of the contract
        # and has to be reachable in the demo rather than only in a unit test.
        stated = [p for p in self.patients if p["weight_kg"] is not None]
        withheld = [p for p in self.patients if p["weight_kg"] is None]
        self.assertTrue(stated)
        self.assertTrue(withheld, "no stay without a weight: the withheld path is invisible")

    def test_a_stay_without_a_weight_gets_no_rate_and_a_reason(self):
        payload = build_payload()
        for row in _rows(payload):
            if row["weightKg"] is None:
                self.assertIsNone(row["urineRatePerKg"], row["patientId"])
                self.assertIsNone(row["urineBand"], row["patientId"])
                if any("urine output" in r for r in row["reasons"]):
                    self.assertTrue(any("weight not recorded" in r for r in row["reasons"]),
                                    row["patientId"])

    def test_a_stay_with_a_weight_and_urine_gets_the_rate(self):
        payload = build_payload()
        rated = [r for r in _rows(payload) if r["urineRatePerKg"] is not None]
        self.assertTrue(rated, "no stay on the demo ward can state a rate per kg")
        for row in rated:
            self.assertTrue(row["urineBand"], row["patientId"])
            self.assertAlmostEqual(row["urineRatePerKg"],
                                   row["series"]["urine_output"][-1][1] / row["weightKg"])

    def test_the_units_line_says_what_the_weight_column_is(self):
        self.assertIn("weight_kg", self.document["ward"]["variable_units"])


class PayloadTests(unittest.TestCase):
    def test_provenance_travels_with_the_board(self):
        payload = build_payload()
        self.assertEqual(payload["schema"], "oslian.ward-attention-order.v3")
        self.assertTrue(payload["source"]["retrospective"])
        self.assertTrue(payload["source"]["notLive"])
        self.assertIn("license", payload["source"]["metadata"])
        self.assertFalse(payload["safetyBoundary"]["causal_claim_allowed"])

    def test_every_row_carries_its_care_unit(self):
        payload = build_payload()
        for row in _rows(payload):
            self.assertTrue(row["careUnit"], row["patientId"])

    def test_every_row_carries_the_series_the_rank_was_read_off(self):
        payload = build_payload()
        for row in _rows(payload):
            if not row["rankable"]:
                continue
            self.assertTrue(row["series"]["creatinine"], row["patientId"])
            self.assertIsNotNone(row["nowHours"])
            # nothing after the patient's own clock may reach the chart
            for hour, _ in row["series"]["creatinine"] + row["series"]["urine_output"]:
                self.assertLessEqual(hour, row["nowHours"] + 1e-9)

    def test_the_overview_names_every_band_that_has_anyone_in_it(self):
        payload = build_payload()
        for band in payload["bands"]:
            if band["count"]:
                self.assertIn(band["label"], payload["overview"], band["key"])

    def test_the_display_cap_truncates_but_reports_what_it_held_back(self):
        small = build_payload(display_cap=2)
        for band in small["bands"]:
            self.assertLessEqual(len(band["rows"]), 2)
            self.assertEqual(len(band["rows"]) + band["truncated"], band["count"])
        # Every scored stay is still counted somewhere.
        scored = sum(b["count"] for b in small["bands"])
        self.assertEqual(scored + len(small["unrankable"]), small["totalPatients"])

    def test_no_stay_is_dropped_from_the_board(self):
        payload = build_payload()
        counted = sum(b["count"] for b in payload["bands"]) + len(payload["unrankable"])
        self.assertEqual(counted, payload["totalPatients"])

    def test_a_bad_schema_fails_closed_rather_than_returning_an_empty_ward(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"schema_version": "something.else"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ward(path)

    def test_a_fixture_that_does_not_declare_itself_retrospective_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            document = json.loads(WARD_FIXTURE.read_text(encoding="utf-8"))
            document["not_live"] = False
            path = Path(tmp) / "live.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ward(path)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_the_route_serves_the_board(self):
        response = self.client.get("/api/ward")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["schema"], "oslian.ward-attention-order.v3")
        self.assertEqual(payload["scoreBasis"], "last_creatinine_any_age")
        self.assertEqual([b["key"] for b in payload["bands"]],
                         ["second_cut", "first_cut", "outside", "within"])
        self.assertFalse(payload["isClinicalAlert"])

    def test_a_non_numeric_cap_is_refused(self):
        self.assertEqual(self.client.get("/api/ward?cap=lots").status_code, 400)

    def test_an_out_of_range_cap_is_refused(self):
        self.assertEqual(self.client.get("/api/ward?cap=0").status_code, 400)
        self.assertEqual(self.client.get(f"/api/ward?cap={MAX_DISPLAY_CAP + 1}").status_code, 400)

    def test_a_missing_fixture_returns_503_not_an_empty_board(self):
        # An empty list reads as "nobody needs looking at". That must never be
        # what a missing file produces.
        app = create_app()
        app.register_blueprint(ward.create_blueprint(Path("/nonexistent/ward.json")),
                               name="ward_missing", url_prefix="/missing")
        response = app.test_client().get("/missing/api/ward")
        self.assertEqual(response.status_code, 503)
        self.assertIn("unavailable", response.get_json()["error"])

    def test_the_board_declares_that_the_rate_per_kg_orders_nothing(self):
        payload = self.client.get("/api/ward").get_json()
        self.assertIs(payload["urineRateOrdersNothing"], True)
        self.assertEqual(payload["orderWithinBandBasis"], "last_creatinine_any_age")

    def test_the_score_orders_each_band_even_though_it_no_longer_orders_the_board(self):
        # Before the bands read more than one analyte, reading the board top to
        # bottom was the creatinine ordering. It is not any more, deliberately: a
        # normal creatinine with a sodium of 110 sits above a creatinine of 2.9.
        # The measured claim is about the ordering, and the ordering lives inside
        # a band -- so that is what has to hold.
        payload = self.client.get("/api/ward?cap=200").get_json()
        for band in payload["bands"]:
            scores = [r["score"] for r in band["rows"]]
            self.assertEqual(scores, sorted(scores, reverse=True), band["key"])
        ranked = [r for b in payload["bands"] for r in b["rows"]]
        self.assertEqual([r["rank"] for r in ranked], sorted(r["rank"] for r in ranked))

    def test_the_two_bases_are_named_apart_in_the_payload(self):
        payload = self.client.get("/api/ward").get_json()
        self.assertEqual(payload["bandBasis"], "highest_tier_of_any_measured_analyte")
        self.assertEqual(payload["orderWithinBandBasis"], "last_creatinine_any_age")

    def test_the_cutpoint_table_travels_with_the_board_and_flags_its_own_sources(self):
        # A reader has to be able to check any cut, and to see which of them this
        # project cannot yet cite. An unpinned cut is used, never hidden.
        payload = self.client.get("/api/ward").get_json()
        table = {c["variable"]: c for c in payload["cutpoints"]}
        self.assertIn("creatinine", table)
        self.assertNotIn("calcium", table)
        for cut in table.values():
            self.assertTrue(cut["source"], cut["variable"])
            self.assertIsInstance(cut["sourcePinned"], bool, cut["variable"])
        self.assertFalse(table["bun"]["sourcePinned"])

    def test_a_stay_on_the_list_says_which_analyte_put_it_there(self):
        payload = self.client.get("/api/ward?cap=200").get_json()
        on_list = [r for b in payload["bands"] if b["key"] in ("second_cut", "first_cut")
                   for r in b["rows"]]
        self.assertTrue(on_list, "no stay on the demo ward reaches either upper band")
        for row in on_list:
            self.assertTrue(row["findings"], row["patientId"])
            # the band is exactly the worst tier among its findings
            self.assertEqual(row["band"], row["findings"][0]["tierKey"], row["patientId"])
            self.assertGreaterEqual(row["outsideCount"], 1)

    def test_the_demo_ward_shows_a_stay_lifted_by_something_other_than_creatinine(self):
        # The whole reason for reading more than one analyte. On the eICU demo
        # ward most of the list is there for a reason other than the creatinine,
        # and the demo has to show that rather than only assert it in a unit test.
        payload = self.client.get("/api/ward?cap=200").get_json()
        lifted = [r for b in payload["bands"] if b["key"] in ("second_cut", "first_cut")
                  for r in b["rows"]
                  if r["findings"][0]["variable"] != "creatinine"]
        self.assertTrue(lifted, "no stay is on the list for a non-creatinine reason")

    def test_the_ward_route_does_not_touch_the_wholebody_runtime(self):
        # The order is computed from observations alone. If this route ever grows
        # a runtime dependency it stops being answerable without a model contract.
        import inspect
        source = inspect.getsource(ward)
        for forbidden in ("wholebody", "WholeBodyRuntime", "forecast_point", "release"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
