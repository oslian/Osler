import unittest

import numpy as np
import pandas as pd

from mimic_to_eicu_site_adaptive_rescue import (
    common_treatment_columns,
    hospital_cv_conformal_quantile,
    rescue_gate,
    split_adaptation_hospitals,
    split_adaptation_subjects,
)


class SiteAdaptiveRescueTests(unittest.TestCase):
    def test_hospital_split_keeps_ten_untouched(self):
        frame = pd.DataFrame({"hospitalid": np.arange(12).astype(str)})
        adaptation, evaluation = split_adaptation_hospitals(frame, 7)
        self.assertEqual(len(adaptation), 2)
        self.assertEqual(len(evaluation), 10)
        self.assertFalse(adaptation & evaluation)

    def test_subject_partitions_are_disjoint(self):
        frame = pd.DataFrame({
            "subject_id": np.arange(90).astype(str),
            "hospitalid": "a",
        })
        rows = np.arange(len(frame))
        selection, scale, conformal = split_adaptation_subjects(
            frame, rows, 11, source_selection=True
        )
        self.assertFalse(set(selection) & set(scale))
        self.assertFalse(set(selection) & set(conformal))
        self.assertFalse(set(scale) & set(conformal))
        self.assertEqual(len(selection) + len(scale) + len(conformal), 90)

    def test_treatment_schema_keeps_only_shared_supported_fields(self):
        mimic = pd.DataFrame({
            "hist_insulin": np.ones(40),
            "hist_insulin_dose_observed": np.ones(40),
            "hist_dextrose": np.ones(40),
        })
        eicu = pd.DataFrame({
            "hist_insulin": np.ones(40),
            "hist_insulin_dose_observed": np.zeros(40),
        })
        columns = common_treatment_columns(mimic, eicu, "glucose")
        self.assertEqual(columns, ["hist_insulin"])

    def test_interval_only_rescue_reuses_frozen_point_evidence(self):
        self.assertTrue(
            rescue_gate(source_rescue=False, point_pass=False, interval_pass=True)
        )

    def test_new_source_must_pass_both_gates(self):
        self.assertFalse(
            rescue_gate(source_rescue=True, point_pass=False, interval_pass=True)
        )
        self.assertTrue(
            rescue_gate(source_rescue=True, point_pass=True, interval_pass=True)
        )

    def test_hospital_cv_conformal_does_not_use_evaluation_labels(self):
        frame = pd.DataFrame({
            "subject_id": [f"p{i}" for i in range(120)],
            "hospitalid": [f"h{i % 5}" for i in range(120)],
            "careunit": ["ICU" if i % 2 else "CCU" for i in range(120)],
            "anchor_time": pd.date_range("2025-01-01", periods=120, freq="h"),
        })
        quantile, metadata = hospital_cv_conformal_quantile(
            frame, np.arange(120), np.linspace(0.01, 2.0, 120)
        )
        self.assertGreater(quantile, 0.0)
        self.assertFalse(metadata["test_labels_used"])
        self.assertEqual(metadata["adaptation_hospitals"], 5)


if __name__ == "__main__":
    unittest.main()
