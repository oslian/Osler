import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from scripts.calibrate_multisource_intervals import unused_patients
from whole_body_jepa.multisource_intervals import (
    BOUND_MODEL_FILES, audit_intervals, conformal_radius, coverage, domain_coverage, load_calibrator, patient_split,
    prediction_interval, representative_rows, sha256,
)


class MultisourceIntervalTests(unittest.TestCase):
    def test_finite_sample_order_statistic(self):
        self.assertEqual(conformal_radius(np.arange(100)), 90.0)
        with self.assertRaises(ValueError):
            conformal_radius([0, 1, 2])
        with self.assertRaises(ValueError):
            conformal_radius([0, float("nan")])

    def test_exclusion_is_patient_not_row_level(self):
        full = pd.DataFrame({"subject_id": ["a", "a", "b", "c"], "stay_id": [1, 2, 3, 4]})
        selected = full.iloc[[0]]
        self.assertEqual(unused_patients(full, selected)["subject_id"].tolist(), ["b", "c"])

    def test_seven_split_roles_do_not_split_patients(self):
        subjects = np.repeat(np.arange(200), 5)
        for seed in (7, 19, 31, 43, 59, 71, 83):
            cal, test = patient_split(subjects, seed)
            self.assertFalse(set(subjects[cal]) & set(subjects[test]))
            self.assertTrue(np.all(cal != test))

    def test_patient_anchor_selection_ignores_values(self):
        frame = pd.DataFrame({"subject_id": ["a", "a", "b"], "stay_id": [1, 1, 2],
                              "t": ["2020-01-01", "2020-01-02", "2020-01-01"], "residual": [0, 100, 50]})
        first = representative_rows(frame, 7)
        frame["residual"] = [100000, 0, 0]
        np.testing.assert_array_equal(first, representative_rows(frame, 7))
        self.assertEqual(len(first), 2)
        with self.assertRaises(ValueError):
            representative_rows(pd.concat([frame, frame.iloc[[0]]]), 7)

    def test_unknown_or_single_domain_never_passes(self):
        frame = pd.DataFrame({"subject_id": np.arange(200), "careunit": "unknown", "residual": 0.5})
        self.assertFalse(domain_coverage(frame, 1.0)["pass"])
        frame["careunit"] = "MICU"
        self.assertFalse(domain_coverage(frame, 1.0)["pass"])

    def test_coverage_reports_overcoverage_as_failure(self):
        frame = pd.DataFrame({"subject_id": np.arange(200), "residual": 0.5})
        self.assertFalse(coverage(frame, 1.0)["pass"])
        frame.loc[:19, "residual"] = 2.0
        self.assertTrue(coverage(frame, 1.0)["pass"])

    def test_empty_unused_pool_does_not_calibrate_on_external_labels(self):
        external = pd.DataFrame({"subject_id": np.arange(200).astype(str), "careunit": ["a", "b"] * 100,
                                 "t": pd.date_range("2020-01-01", periods=200), "residual": 0.5,
                                 "truth": 2.0, "current": 1.0})
        result = audit_intervals(external.iloc[:0], external, "test@1h", 1)
        self.assertIsNone(result["radius"])
        self.assertFalse(result["runtime_interval_eligible"])
        self.assertIn("insufficient_unused_mimic_calibration_patients", result["failure_reasons"])

    def test_binding_and_failed_cell_are_fail_closed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in BOUND_MODEL_FILES:
                (root / name).write_bytes(b"fixed")
            model = root / BOUND_MODEL_FILES[0]
            payload = {"schema": "whole_body_jepa.multisource_interval_calibrator.v1",
                       "release_binding": {name: sha256(root / name) for name in BOUND_MODEL_FILES},
                       "cells": {"a@1h": {"runtime_interval_eligible": True, "radius": 2.0},
                                 "b@1h": {"runtime_interval_eligible": False, "radius": 0.1}}}
            path = root / "calibrator.json"
            path.write_text(json.dumps(payload))
            loaded = load_calibrator(path, root)
            self.assertEqual(prediction_interval(loaded, "a@1h", 5)["lower"], 3)
            self.assertIsNone(prediction_interval(loaded, "b@1h", 5)["lower"])
            self.assertIsNone(prediction_interval(loaded, "unknown", 5)["lower"])
            self.assertIsNone(prediction_interval(loaded, "a@1h", None)["upper"])
            path.write_text(json.dumps({**payload, "release_binding": {}}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_calibrator(path, root)
            path.write_text(json.dumps(payload))
            model.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_calibrator(path, root)


if __name__ == "__main__":
    unittest.main()
