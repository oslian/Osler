import unittest

import numpy as np
import pandas as pd
import torch

from nonlinear_latent_rollout import (
    NonlinearLatentRolloutNet,
    effective_rank,
    horizon_suffix,
    state_variables,
)


class NonlinearLatentRolloutTests(unittest.TestCase):
    def test_horizon_suffix_is_stable(self):
        self.assertEqual(horizon_suffix(1), "tp1")
        self.assertEqual(horizon_suffix(48.0), "tp48")
        self.assertEqual(horizon_suffix(1.5), "tp1p5")

    def test_state_variables_require_current_and_age(self):
        frame = pd.DataFrame({
            "glucose_t": [100.0],
            "glucose_age_hr": [0.2],
            "potassium_t": [4.0],
            "map_t": [70.0],
        })
        self.assertEqual(state_variables(frame), ("glucose",))

    def test_rollout_shape_and_finite_output(self):
        torch.manual_seed(0)
        model = NonlinearLatentRolloutNet(n_state=4, input_dim=13, hidden=16, latent=8)
        observations = torch.randn(6, 5, 13)
        initial = torch.randn(6, 4)
        horizon = torch.tensor([1.0, 3.0, 6.0, 12.0, 24.0, 48.0])
        prediction, latent = model(observations, initial, horizon)
        self.assertEqual(tuple(prediction.shape), (6, 4))
        self.assertEqual(tuple(latent.shape), (6, 8))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(latent).all())

        diagnostic_prediction, diagnostic_latent, diagnostics = model(
            observations, initial, horizon, return_diagnostics=True
        )
        self.assertEqual(tuple(diagnostic_prediction.shape), (6, 4))
        self.assertEqual(tuple(diagnostic_latent.shape), (6, 8))
        self.assertEqual(tuple(diagnostics["regime_usage"].shape), (3,))
        self.assertTrue(torch.isfinite(diagnostics["regime_usage"]).all())
        self.assertAlmostEqual(
            float(diagnostics["regime_usage"].sum().detach()), 1.0, places=4
        )

    def test_effective_rank_flags_constant_embedding(self):
        self.assertEqual(effective_rank(np.ones((20, 8), dtype=np.float32)), 0.0)
        varied = np.random.default_rng(0).normal(size=(100, 8)).astype(np.float32)
        self.assertGreater(effective_rank(varied), 3.0)


if __name__ == "__main__":
    unittest.main()
