import copy
import unittest
import numpy as np
from scripts.coverage_sampling import coverage_indices


class CoverageSamplingTests(unittest.TestCase):
    def test_full_cycle_and_more(self):
        x, pending = coverage_indices([.5, 1, 2], np.random.default_rng(3), 8, [])
        for start in (0, 3):
            self.assertEqual(set(x[start:start+3]), {0, 1, 2})
        self.assertEqual(len(pending), 1)

    def test_resume_matches_uninterrupted(self):
        rng = np.random.default_rng(8)
        first, pending = coverage_indices(np.ones(11), rng, 5, [])
        state = copy.deepcopy(rng.bit_generator.state)
        second, rest = coverage_indices(np.ones(11), rng, 25, pending)
        restored = np.random.default_rng()
        restored.bit_generator.state = state
        again, again_rest = coverage_indices(np.ones(11), restored, 25, pending)
        np.testing.assert_array_equal(second, again)
        np.testing.assert_array_equal(rest, again_rest)
        total, _ = coverage_indices(np.ones(11), np.random.default_rng(8), 30, [])
        np.testing.assert_array_equal(np.concatenate([first, second]), total)

    def test_invalid_pending(self):
        with self.assertRaises(ValueError):
            coverage_indices([1, 1], np.random.default_rng(), 2, [0, 0])
