import unittest
import numpy as np

from scripts.train_nonlinear_probe import PARAMETERS, fit_probes


class ProbeTrainingTests(unittest.TestCase):
    def test_fixed_iterations_and_query_independence(self):
        rng = np.random.default_rng(19)
        body = rng.normal(size=(50, 64))
        raw = rng.normal(size=(50, 265))
        y = body[:, 0]
        first, predictions = fit_probes(body, raw, y, np.ones(50), body[:4], raw[:4])
        second, _ = fit_probes(body, raw, y, np.ones(50), body[:4]+20, raw[:4]-10)
        self.assertFalse(PARAMETERS['early_stopping'])
        for key in first:
            self.assertEqual(first[key].n_iter_, 100)
            self.assertTrue(np.isfinite(predictions[key]).all())
            np.testing.assert_array_equal(first[key]._baseline_prediction, second[key]._baseline_prediction)
            for a, b in zip(first[key]._predictors, second[key]._predictors):
                np.testing.assert_array_equal(a[0].nodes, b[0].nodes)

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            fit_probes(np.zeros((3, 64)), np.zeros((3, 264)), np.ones(3), np.ones(3),
                       np.zeros((2, 64)), np.zeros((2, 265)))


if __name__ == '__main__':
    unittest.main()
