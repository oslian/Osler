import unittest

import numpy as np
import pandas as pd

from nonlinear_latent_rollout import StateScaler
from whole_body_jepa.representation_repair import observation_features


class RepresentationRepairTests(unittest.TestCase):
    def test_observation_features_distinguish_missing_fresh_and_stale(self):
        scaler = StateScaler(("target", "donor"), np.array([10., 100.]), np.array([2., 10.]))
        frame = pd.DataFrame({
            "target_t": [10., 10., 10.], "target_age_hr": [0., 0., 0.],
            "donor_t": [100., np.nan, 100.], "donor_age_hr": [0., np.nan, 5.],
        })
        raw, observed = observation_features(frame, scaler, ("target", "donor"), "target")
        np.testing.assert_array_equal(raw[:, 0], [0., 0., 0.])
        self.assertFalse(np.array_equal(observed[0], observed[1]))
        self.assertFalse(np.array_equal(observed[0], observed[2]))

    def test_target_anchor_is_excluded_from_cross_variable_block(self):
        scaler = StateScaler(("target", "donor"), np.array([10., 100.]), np.array([2., 10.]))
        frame = pd.DataFrame({
            "target_t": [10.], "target_age_hr": [0.], "donor_t": [110.], "donor_age_hr": [1.],
        })
        raw, observed = observation_features(frame, scaler, ("target", "donor"), "target")
        self.assertEqual(raw.shape, (1, 1))
        self.assertEqual(observed.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
