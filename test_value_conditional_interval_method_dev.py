import unittest

import numpy as np
import pandas as pd

from value_conditional_interval_method_dev import (
    candidate_widths,
    patient_row_weights,
    weighted_conformal_quantile,
)


class IntervalMethodDevTests(unittest.TestCase):
    def test_patient_row_weights_sum_to_one_per_patient(self):
        weights = patient_row_weights(np.array(["a", "a", "a", "b"]))
        np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3, 1.0])

    def test_weighted_quantile_ignores_downweighted_rows(self):
        scores = np.array([1.0, 2.0, 3.0, 100.0])
        weights = np.array([1.0, 1.0, 1.0, 1e-9])
        self.assertEqual(weighted_conformal_quantile(scores, weights, patients=1000, coverage=0.9), 3.0)

    def test_transient_bin_visitor_no_longer_inflates_bin_width(self):
        # 40 resident patients with 10 small residuals in bin 1, plus 40 visitors
        # with a single large residual there (their other 19 rows sit in bin 0).
        # Patient-equal weight of the visitor rows in bin 1 is 2/42 < 10%.
        subjects, bins, residual = [], [], []
        for patient in range(40):
            subjects += [f"r{patient}"] * 10
            bins += [1] * 10
            residual += [1.0] * 10
        for patient in range(40):
            subjects += [f"v{patient}"] * 20
            bins += [0] * 19 + [1]
            residual += [1.0] * 19 + [10.0]
        frame = pd.DataFrame({"subject_id": subjects})
        rows = np.arange(len(subjects))
        bins = np.array(bins)
        residual = np.array(residual)
        v1 = candidate_widths("mondrian_v1", frame, rows, residual, bins, np.array([1]))[0]
        fixed_weight = candidate_widths("mondrian_global_weight", frame, rows, residual, bins, np.array([1]))[0]
        self.assertEqual(v1, 10.0)
        self.assertEqual(fixed_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
