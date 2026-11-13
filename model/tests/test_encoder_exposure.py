import unittest
import numpy as np
from scripts.audit_encoder_exposure import replay


class ExposureTest(unittest.TestCase):
    def test_matches_training_rng(self):
        pools = [np.array([0, 2]), np.array([4, 8, 9])]
        rng = np.random.default_rng(19)
        expected = np.zeros(10, dtype=int)
        for _ in range(12):
            selected = [rng.choice(pools[i]) for i in rng.integers(len(pools),size=5)]
            np.add.at(expected,selected,1)
        np.testing.assert_array_equal(replay(pools,19,3,4,5),expected)
        self.assertEqual(expected.sum(),60)
        self.assertEqual(expected[[1,3,5,6,7]].sum(),0)
