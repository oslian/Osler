import unittest

import numpy as np
import pandas as pd

from scripts.diagnose_source_group_residuals import arm_statistics, crossfit_offset, group_diagnostic, patient_half


class ResidualDiagnosisTests(unittest.TestCase):
    def test_exact_constant_bias_recovered_out_of_half(self):
        truth = np.arange(10.)
        prediction = truth+3
        np.testing.assert_allclose(crossfit_offset(truth, prediction, np.arange(10) % 2), truth)

    def test_held_labels_never_change_held_predictions(self):
        truth = np.arange(10.)
        halves = np.arange(10) % 2
        prediction = truth+np.arange(10.)**2
        before = crossfit_offset(truth, prediction, halves)
        truth[halves == 0] += 1000
        after = crossfit_offset(truth, prediction, halves)
        np.testing.assert_array_equal(before[halves == 0], after[halves == 0])

    def test_insufficient_halves_remain_missing(self):
        result = arm_statistics([1., 1.], [2., 2.], [0, 0])
        self.assertIsNone(result['crossfit_local_offset_mae'])
        self.assertIsNone(result['spearman'])
        self.assertIsNone(result['prediction_truth_iqr_ratio'])

    def test_centering_cannot_repair_reversed_ranking(self):
        y = np.arange(-5., 5.)
        result = arm_statistics(y, -y, np.arange(10) % 2)
        self.assertAlmostEqual(result['spearman'], -1.)
        self.assertGreater(result['crossfit_local_offset_mae'], 0)

    def test_hash_is_patient_specific_not_target_specific(self):
        self.assertEqual(patient_half('a'), patient_half('a'))
        self.assertTrue({patient_half(str(i)) for i in range(100)} == {0, 1})

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            crossfit_offset([1., np.nan], [1., 2.], [0, 1])

    def test_duplicate_patients_rejected(self):
        part = pd.DataFrame(dict(patient=['a', 'a'], truth=1., real=1., process=2., population=2., fit_scale=1.))
        with self.assertRaises(ValueError):
            group_diagnostic(part)


if __name__ == '__main__':
    unittest.main()
