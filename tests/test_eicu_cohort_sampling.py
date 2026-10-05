import unittest

import pandas as pd

from osler_jepa.event_adapters import select_eicu_encounters


class EICUCohortSamplingTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame([
            {"stay_id": h * 1000 + i, "subject_id": f"{h}:{i}", "hospitalid": h,
             "outcome": i % 2}
            for h in range(4) for i in range(40)
        ])

    def test_budget_covers_all_hospitals_instead_of_id_prefix(self):
        selected = select_eicu_encounters(self.meta, 20)
        self.assertEqual(selected.groupby("hospitalid").size().tolist(), [5] * 4)
        self.assertEqual(list(selected.columns), list(self.meta.columns))

    def test_selection_invariant_to_file_order_and_outcomes(self):
        expected = select_eicu_encounters(self.meta, 20).stay_id.tolist()
        shuffled = self.meta.sample(frac=1, random_state=19)
        shuffled["outcome"] = 1000
        self.assertEqual(select_eicu_encounters(shuffled, 20).stay_id.tolist(), expected)

    def test_distinct_patients_before_repeated_encounters(self):
        repeated = self.meta.copy()
        repeated["stay_id"] += 10000
        all_stays = pd.concat([self.meta, repeated], ignore_index=True)
        selected = select_eicu_encounters(all_stays, len(self.meta))
        self.assertEqual(selected.subject_id.nunique(), len(self.meta))
        self.assertEqual(len(select_eicu_encounters(all_stays, len(self.meta) + 10)), 170)

    def test_small_hospitals_and_no_budget(self):
        small = self.meta.loc[(self.meta.hospitalid != 0) | (self.meta.stay_id == 0)]
        selected = select_eicu_encounters(small, 20)
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected.hospitalid.nunique(), 4)
        self.assertEqual(len(select_eicu_encounters(small, None)), len(small))
        self.assertEqual(len(select_eicu_encounters(small, 1000)), len(small))

    def test_invalid_identifiers_and_budgets_fail_closed(self):
        for limit in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                select_eicu_encounters(self.meta, limit)
        for invalid in (self.meta.drop(columns="hospitalid"),
                        self.meta.assign(hospitalid=None),
                        pd.concat([self.meta, self.meta.iloc[:1]])):
            with self.assertRaises(ValueError):
                select_eicu_encounters(invalid, 20)

    def test_empty_metadata(self):
        self.assertTrue(select_eicu_encounters(self.meta.iloc[:0], 20).empty)

    def test_coverage_report_is_aggregate_only(self):
        from scripts.audit_masked_readout_ensemble import cohort_coverage

        frame = self.meta.assign(careunit="ICU")
        report = cohort_coverage(frame)
        self.assertEqual(report["hospitals"], 4)
        self.assertEqual(report["largest_hospital_encounter_fraction"], .25)
        self.assertNotIn("subject_id", report)
        self.assertNotIn("stay_id", report)

    def test_failure_report_distinguishes_harm_from_inconclusive(self):
        from scripts.compare_masked_readout_reports import failure_breakdown

        comparison = {"all": {"pass": True}, "paired": {"pass": True},
                      "patient_splits": [{"pass": True}],
                      "careunits": {
                          "harm": {"pass": False, "available": True, "ci95": [.1, .2]},
                          "uncertain": {"pass": False, "available": True, "ci95": [-.1, .2]},
                          "win": {"pass": True, "available": True, "ci95": [-.2, -.1]}}}
        task = {"status": "evaluated", "ensemble": {"comparisons": {"population": comparison}},
                "vs_trained_placebo": comparison}
        result = failure_breakdown(task)["population"]
        self.assertEqual(result["significantly_worse_care_units"], ["harm"])
        self.assertEqual(result["care_units_passed"], 1)
        self.assertEqual(result["patient_splits_passed"], 1)
        self.assertEqual(set(result["failed_care_units"]), {"harm", "uncertain"})


if __name__ == "__main__":
    unittest.main()
