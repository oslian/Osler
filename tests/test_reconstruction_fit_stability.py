import unittest

import numpy as np
import pandas as pd

from scripts.audit_reconstruction_fit_stability import align_target, decompose


class ReconstructionFitStabilityTests(unittest.TestCase):
    def test_exact_decomposition(self):
        r = decompose([[0, 2], [2, 4]], [1, 1])
        self.assertEqual(r['mean_fit_mse'], 3)
        self.assertEqual(r['between_fit_variance'], 1)
        self.assertEqual(r['mean_prediction_residual_mse'], 2)

    def test_identical_bad_predictions_are_not_instability(self):
        r = decompose([[3, 3], [3, 3]], [0, 0])
        self.assertEqual(r['between_fit_mse_fraction'], 0)
        self.assertEqual(r['mean_prediction_residual_mse'], 9)

    def test_perfect_prediction(self):
        self.assertIsNone(decompose([[0, 1]], [0, 1])['between_fit_mse_fraction'])

    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError):
            decompose([[np.nan]], [1])

    def test_alignment_and_no_silent_intersection(self):
        frame = pd.DataFrame(dict(subject_id=[1, 2], target=['x', 'x'], truth=[3, 4]))
        aligned, truth = align_target({7: frame, 19: frame.iloc[::-1]}, 'x')
        self.assertEqual(len(aligned), 2)
        np.testing.assert_array_equal(truth, [3, 4])
        with self.assertRaises(ValueError):
            align_target({7: frame, 19: frame.iloc[:1]}, 'x')
        changed = frame.copy()
        changed.loc[0, 'truth'] = 5
        with self.assertRaises(ValueError):
            align_target({7: frame, 19: changed}, 'x')
        with self.assertRaises(ValueError):
            align_target({7: pd.concat([frame, frame])}, 'x')


if __name__ == '__main__':
    unittest.main()
