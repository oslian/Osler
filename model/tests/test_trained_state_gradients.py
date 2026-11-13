import unittest
import numpy as np
from scripts.audit_trained_state_gradients import geometry


class GradientGeometryTest(unittest.TestCase):
    def test_opposition_and_average_sign(self):
        cosine, dot, norm = geometry(np.array([[1., 0.], [-3., 0.]]))
        self.assertEqual(cosine[0, 1], -1.)
        self.assertLess(dot[0], 0.)
        self.assertGreater(dot[1], 0.)
        np.testing.assert_array_equal(norm, [1., 3.])

    def test_zero_gradient_is_not_conflict(self):
        cosine, dot, norm = geometry(np.array([[0., 0.], [1., 1.]]))
        self.assertTrue(np.isfinite(cosine).all())
        self.assertEqual(dot[0], 0.)
        self.assertEqual(norm[0], 0.)
