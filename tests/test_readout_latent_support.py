import unittest
import numpy as np
from scripts.audit_readout_latent_support import support_distances, strata


class SupportTests(unittest.TestCase):
    def test_query_cannot_change_fitted_threshold(self):
        rng = np.random.default_rng(19)
        train = rng.normal(size=(200, 5))
        query = rng.normal(size=(10, 5))
        distance, threshold = support_distances(train, query)
        augmented, other = support_distances(train, np.r_[query, np.full((1, 5), 100.)])
        self.assertEqual(threshold, other)
        np.testing.assert_allclose(distance, augmented[:-1])
        self.assertGreater(augmented[-1], threshold)

    def test_no_empty_stratum_mean(self):
        result = strata(np.array([1., -2.]), np.array([1., 2.]), 3.)
        self.assertEqual(result['inside']['mean_delta'], -.5)
        self.assertIsNone(result['outside']['mean_delta'])
