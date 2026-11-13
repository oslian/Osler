import unittest

import numpy as np
import pandas as pd

from external_value_conditional_interval_audit import (
    evaluability,
    interval_report,
    normalized_bin_widths,
    value_bin_edges,
)


class ValueConditionalIntervalTests(unittest.TestCase):
    def _frame(self, n):
        return pd.DataFrame({"subject_id": [f"MIMIC:{i}" for i in range(n)]})

    def test_bin_edges_come_from_calibration_values_only(self):
        edges = value_bin_edges(np.arange(100, dtype=float))
        self.assertEqual(len(edges), 4)
        self.assertTrue(np.all(np.diff(edges) > 0))

    def test_high_value_bin_gets_wider_interval(self):
        n = 400
        frame = self._frame(n)
        rows = np.arange(n)
        bins = np.repeat([0, 1], n // 2)
        residual = np.where(bins == 0, np.linspace(0, 1, n), np.linspace(0, 5, n))
        widths, table = normalized_bin_widths(frame, rows, residual, bins, np.array([0, 1]))
        self.assertLess(widths[0], widths[1])
        self.assertTrue(all(item["source"] == "bin" for item in table))

    def test_one_global_quantile_gives_nominal_patient_coverage(self):
        rng = np.random.default_rng(3)
        n = 3000
        frame = self._frame(n)
        bins = rng.integers(0, 5, size=n)
        residual = np.abs(rng.normal(size=n)) * (1 + bins)
        widths, _ = normalized_bin_widths(frame, np.arange(n), residual, bins, bins)
        self.assertAlmostEqual(float(np.mean(residual <= widths)), 0.90, delta=0.01)

    def test_sparse_bin_falls_back_to_overall_scale(self):
        n = 200
        frame = self._frame(n)
        bins = np.zeros(n, dtype=int)
        bins[:5] = 1
        _, table = normalized_bin_widths(frame, np.arange(n), np.linspace(0, 1, n), bins, np.array([1]))
        self.assertEqual(next(item for item in table if item["bin"] == 1)["source"], "overall_scale_fallback")


class GateAxisTests(unittest.TestCase):
    def test_nwicu_interval_gate_uses_careunit_not_hospital(self):
        frame = pd.DataFrame({
            "subject_id": [f"NWICU:{i}" for i in range(4)],
            "hospitalid": ["NWICU"] * 4,
            "careunit": ["MICU", "MICU", "SICU", "SICU"],
        })
        covered = np.array([True, True, True, False])
        report = interval_report(frame, np.arange(4), covered, np.ones(4), ("subject_id", "careunit"))
        self.assertEqual(report["gate_axes"], ["patient_equal_coverage", "careunit_equal_coverage"])
        self.assertAlmostEqual(report["careunit_equal_coverage"], 0.75)
        self.assertFalse(report["pass"])

    def test_evaluability_is_support_based_for_any_database(self):
        self.assertEqual(evaluability("SICdb", pd.DataFrame()), "insufficient_external_support")
        small = pd.DataFrame({"subject_id": [f"SICDB:{i}" for i in range(10)], "hospitalid": ["SICdb"] * 10})
        self.assertEqual(evaluability("SICdb", small), "insufficient_external_support")
        ok = pd.DataFrame({"subject_id": [f"HIRID:{i % 50}" for i in range(200)], "hospitalid": ["HiRID"] * 200})
        self.assertIsNone(evaluability("HiRID", ok))

    def test_single_site_databases_use_patient_and_careunit_axes(self):
        from external_value_conditional_interval_audit import GATE_AXES, EXTERNAL_PREFIX

        for database in ("NWICU", "SICdb", "HiRID"):
            self.assertEqual(GATE_AXES[database]["interval"], ("subject_id", "careunit"))
            self.assertEqual(GATE_AXES[database]["point"], ("patient",))
        self.assertEqual(EXTERNAL_PREFIX["SICdb"], "SICDB:")


if __name__ == "__main__":
    unittest.main()
