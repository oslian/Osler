import unittest
import numpy as np
from scripts.audit_trained_state_gradients import geometry
from scripts.audit_score_gradient_concentration import subset_geometry


class ScoreGradientTest(unittest.TestCase):
    def test_exact_dot_reconstruction(self):
        vectors = np.array([[2., 0.], [-1., 1.], [0., 3.]])
        cosine, dot, norms = geometry(vectors)
        r = subset_geometry(norms, cosine, [True, False, False])
        np.testing.assert_allclose(r['dot_with_average'], dot, atol=1e-12)
        self.assertAlmostEqual(r['norm_sum_share'], 2 / (5 + np.sqrt(2)))

    def test_signed_projection_not_probability(self):
        cosine, _, norms = geometry(np.array([[2., 0.], [-1., 0.]]))
        r = subset_geometry(norms, cosine, [False, True])
        self.assertLess(r['signed_mean_direction_share'], 0)

    def test_zero_gradients_unknown(self):
        r = subset_geometry([0., 0.], np.zeros((2, 2)), [True, False])
        self.assertIsNone(r['norm_sum_share'])
        self.assertIsNone(r['signed_mean_direction_share'])
