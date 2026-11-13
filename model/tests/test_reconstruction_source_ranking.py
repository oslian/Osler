import unittest

import numpy as np

from scripts.audit_reconstruction_source_ranking import correlation


class SourceRankingTests(unittest.TestCase):
    def test_unsupported_and_constant(self):
        self.assertIsNone(correlation(np.array([1, 2]), np.array([2, 3])))
        self.assertIsNone(correlation(np.ones(3), np.arange(3)))
        self.assertIsNone(correlation(np.arange(3), np.ones(3)))

    def test_affine_shift_does_not_repair_ranking(self):
        x = np.arange(5.)
        y = -x
        self.assertAlmostEqual(correlation(x, y), -1.)
        self.assertAlmostEqual(correlation(.2*x+100, y), -1.)

    def test_ties_are_supported(self):
        x = np.array([1., 1., 3., 4.])
        self.assertAlmostEqual(correlation(x, x), 1.)


if __name__ == '__main__':
    unittest.main()
