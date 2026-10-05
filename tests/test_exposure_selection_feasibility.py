import unittest
import numpy as np
from scripts.audit_exposure_selection_feasibility import trajectory


class TrajectoryTest(unittest.TestCase):
    def test_order_and_tradeoff(self):
        arm = dict(initial_scores={'a': {'x': 1., 'y': 2.}, 'b': {'x': 3., 'y': 4.}},
                   log=[dict(group_target_scores={'a': {'x': .5, 'y': 2.1}, 'b': {'x': 2., 'y': 3.}})])
        groups, targets, values = trajectory(arm)
        self.assertEqual(groups, ['a', 'b'])
        self.assertEqual(targets, ['x', 'y'])
        np.testing.assert_array_equal(values, [[1., 2., 3., 4.], [.5, 2.1, 2., 3.]])
        self.assertLess(values[1].mean(), values[0].mean())
        self.assertFalse((values[1] <= values[0]).all())

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            trajectory(dict(initial_scores={'a': {'x': float('nan')}}, log=[]))
