import unittest

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import (
    failure_reasons,
    fit_continuous_recipe,
    group_equal_coverage,
    hospital_bootstrap,
    module_candidates,
    normalize_cohort,
)


class ExternalAuditTests(unittest.TestCase):
    def test_module_order_is_predeclared(self):
        self.assertEqual(module_candidates("glucose")[0], "endocrine_stress")
        self.assertEqual(module_candidates("creatinine")[0], "aki")
        self.assertEqual(module_candidates("hypoxemia_event")[0], "respiratory")

    def test_normalization_filters_quality_and_prefixes_ids(self):
        frame = pd.DataFrame({
            "subject_id": [1, 2, 3],
            "stay_id": [11, 22, 33],
            "hospitalid": [1, 2, 3],
            "unittype": ["MICU", "SICU", "CCU"],
            "t": pd.date_range("2020-01-01", periods=3, freq="h"),
            "glucose_t": [100.0, 110.0, 120.0],
            "glucose_tp6": [105.0, 115.0, 125.0],
            "future_label_quality_glucose": ["exact", "near", "missing"],
        })
        normalized, support = normalize_cohort(
            frame,
            database="eICU",
            target="glucose",
            horizon=6,
            max_rows=100,
        )
        self.assertEqual(len(normalized), 2)
        self.assertTrue(normalized["subject_id"].str.startswith("EICU:").all())
        self.assertEqual(support["quality_filter"], "exact_or_near_when_available")

    def test_nwicu_rows_keep_their_own_prefix_and_careunit(self):
        frame = pd.DataFrame({
            "subject_id": [1],
            "stay_id": [11],
            "hospitalid": ["NWICU"],
            "first_careunit": ["MICU"],
            "t": pd.date_range("2020-01-01", periods=1, freq="h"),
            "glucose_t": [100.0],
            "glucose_tp6": [105.0],
        })
        normalized, _ = normalize_cohort(
            frame, database="NWICU", target="glucose", horizon=6, max_rows=100
        )
        self.assertEqual(normalized["subject_id"].iloc[0], "NWICU:1")
        self.assertEqual(normalized["careunit"].iloc[0], "MICU")
        self.assertEqual(normalized["hospitalid"].iloc[0], "NWICU")

    def test_hospital_bootstrap_detects_consistent_improvement(self):
        hospitals = np.repeat(np.arange(12), 4)
        frame = pd.DataFrame({
            "hospitalid": hospitals.astype(str),
            "subject_id": np.arange(len(hospitals)).astype(str),
        })
        result = hospital_bootstrap(
            frame,
            np.arange(len(frame)),
            np.zeros(len(frame)),
            np.ones(len(frame)),
            seed=7,
        )
        self.assertTrue(result["pass"])

    def test_group_equal_coverage_does_not_weight_large_hospital_more(self):
        frame = pd.DataFrame({
            "hospitalid": ["a", "a", "a", "b"],
            "subject_id": ["1", "2", "3", "4"],
        })
        value = group_equal_coverage(
            frame, np.arange(4), np.array([1, 1, 1, 0]), "hospitalid"
        )
        self.assertEqual(value, 0.5)

    def test_interval_failure_is_explicit(self):
        metrics = {
            "point_pass": True,
            "interval_pass": False,
            "interval": {
                "patient_equal_coverage": 0.95,
                "hospital_equal_coverage": 0.85,
            },
        }
        self.assertEqual(
            failure_reasons(metrics, "continuous_value"),
            ["hospital_interval_miscalibration", "patient_interval_miscalibration"],
        )

    def test_persistence_recipe_predicts_current_value(self):
        frame = pd.DataFrame({
            "subject_id": [f"MIMIC:{i}" for i in range(40)] + ["EICU:1", "EICU:2"],
            "sodium_t": np.arange(42, dtype=float),
            "future_sodium": np.arange(42, dtype=float) + 1.0,
        })
        frame.loc[41, "sodium_t"] = np.nan
        fitted = fit_continuous_recipe(frame, np.arange(40), np.array([40, 41]), target="sodium",
                                       source="persistence", seed=0)
        np.testing.assert_array_equal(fitted["test_rows"], [40])
        np.testing.assert_array_equal(fitted["test_prediction"], [40.0])
        np.testing.assert_array_equal(fitted["comparator"], [40.0])
        self.assertEqual(fitted["comparator_name"], "persistence_matched_baseline")
        self.assertTrue(set(fitted["calibration_rows"]) <= set(range(40)))
        np.testing.assert_array_equal(fitted["calibration_prediction"], frame["sodium_t"].to_numpy()[fitted["calibration_rows"]])


if __name__ == "__main__":
    unittest.main()
