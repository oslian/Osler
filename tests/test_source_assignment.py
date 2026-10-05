import unittest
import numpy as np

from scripts.diagnose_source_assignment import assignment_metrics, independent_pair_mae


class AssignmentTests(unittest.TestCase):
    def test_exact_against_cartesian_reference(self):
        y = np.array([2., 2., -3., 8.])
        p = np.array([4., 2., -9.])
        self.assertAlmostEqual(independent_pair_mae(y, p), abs(y[:, None]-p).mean())

    def test_large_common_offset(self):
        y = np.arange(10.)+1e9
        self.assertAlmostEqual(independent_pair_mae(y, y), 3.3)

    def test_perfect_assignment_beats_permutation(self):
        y = np.arange(20.)
        result = assignment_metrics(y, y, np.arange(20) % 2)
        self.assertEqual(result['actual_mae'], 0)
        self.assertGreater(result['assignment_benefit'], 0)

    def test_constant_prediction_has_no_assignment_benefit(self):
        y = np.arange(20.)
        result = assignment_metrics(y, np.ones(20), np.arange(20) % 2)
        self.assertAlmostEqual(result['assignment_benefit'], 0)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            independent_pair_mae([], [1.])


if __name__ == '__main__':
    unittest.main()
