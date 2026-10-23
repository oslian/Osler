import unittest

import numpy as np

from osler_jepa.site_adaptive_point import PointCorrectionConfig, online_point_correction


def stream(n_stays=300, rows=10, seed=0):
    n = n_stays * rows
    stays = np.repeat(np.arange(n_stays), rows)
    anchors = np.arange(n) * 6.0          # one shared clock: stays follow each other
    rng = np.random.default_rng(seed)
    current = rng.normal(100, 10, size=n)
    return stays, anchors, current, rng


class OnlinePointCorrectionTests(unittest.TestCase):
    def call(self, stays, anchors, current, prediction, observed, config, levels=None):
        return online_point_correction(
            stays=stays, levels=levels or {}, anchor_hours=anchors, resolve_hours=anchors + 6.0,
            current=current, prediction=prediction, observed=observed, scale=5.0, config=config,
        )

    def test_no_steps_returns_the_frozen_prediction(self):
        stays, anchors, current, rng = stream()
        prediction = current + rng.normal(size=len(current))
        out = self.call(stays, anchors, current, prediction, current, PointCorrectionConfig())
        np.testing.assert_allclose(out, prediction)

    def test_offset_learns_a_systematic_site_bias(self):
        stays, anchors, current, rng = stream()
        prediction = current.copy()
        observed = current + 8.0 + rng.normal(size=len(current))
        out = self.call(stays, anchors, current, prediction, observed, PointCorrectionConfig(eta_offset=0.02))
        late = slice(2000, None)
        self.assertLess(np.abs(out[late] - observed[late]).mean(), np.abs(prediction[late] - observed[late]).mean() / 3)

    def test_gain_falls_back_towards_persistence_when_dynamics_do_not_transfer(self):
        stays, anchors, current, rng = stream()
        delta = rng.normal(0, 10, size=len(current))
        prediction = current + delta
        observed = current + rng.normal(size=len(current))   # the site's values do not move as predicted
        out = self.call(stays, anchors, current, prediction, observed, PointCorrectionConfig(eta_gain=0.02))
        late = slice(2000, None)
        self.assertLess(np.abs(out[late] - observed[late]).mean(), np.abs(prediction[late] - observed[late]).mean() / 2)

    def test_future_outcomes_are_not_used(self):
        stays = np.zeros(5, dtype=int)
        anchors = np.arange(5) * 6.0
        current = np.zeros(5)
        prediction = np.zeros(5)
        observed = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        out = online_point_correction(
            stays=stays, levels={}, anchor_hours=anchors, resolve_hours=anchors + 1000.0,
            current=current, prediction=prediction, observed=observed, scale=1.0,
            config=PointCorrectionConfig(eta_offset=1.0),
        )
        np.testing.assert_allclose(out, 0.0)


    def test_another_stays_row_does_not_reveal_a_pending_outcome(self):
        # Two overlapping stays on one clock: stay 0's outcome becomes available at hour 20.
        stays = np.array([0, 1, 0, 1])
        anchors = np.array([0.0, 2.0, 6.0, 8.0])
        resolve = np.array([20.0, 22.0, 26.0, 28.0])
        observed = np.array([100.0, 100.0, 100.0, 100.0])
        out = online_point_correction(
            stays=stays, levels={}, anchor_hours=anchors, resolve_hours=resolve,
            current=np.zeros(4), prediction=np.zeros(4), observed=observed, scale=1.0,
            config=PointCorrectionConfig(eta_offset=1.0),
        )
        np.testing.assert_allclose(out, 0.0)

if __name__ == "__main__":
    unittest.main()
