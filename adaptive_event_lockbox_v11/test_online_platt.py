import unittest

import numpy as np
from scipy.special import expit, logit

from osler_jepa.event_calibration import calibration, calibration_pass
from osler_jepa.online_platt import OnlinePlattConfig, online_platt_scaling, project


def shifted_stream(n: int, shift: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    probability = expit(rng.normal(-2.2, 1.0, n))
    outcome = (rng.random(n) < expit(logit(probability) + shift)).astype(float)
    hours = np.arange(n, dtype=float)
    return hours, probability, outcome


class OnlinePlattTests(unittest.TestCase):
    def test_starts_as_the_frozen_model(self):
        hours, probability, outcome = shifted_stream(50, 1.0)
        never = np.full(50, np.inf)
        calibrated = online_platt_scaling(anchor_hours=hours, resolve_hours=never, probability=probability,
                                          outcome=outcome, config=OnlinePlattConfig())
        np.testing.assert_allclose(calibrated, probability, rtol=1e-9)

    def test_outcomes_are_used_only_after_they_resolve(self):
        hours, probability, outcome = shifted_stream(400, 1.0)
        early = online_platt_scaling(anchor_hours=hours, resolve_hours=hours + 1.0, probability=probability,
                                     outcome=outcome, config=OnlinePlattConfig())
        late = online_platt_scaling(anchor_hours=hours, resolve_hours=hours + 200.0, probability=probability,
                                    outcome=outcome, config=OnlinePlattConfig())
        np.testing.assert_allclose(late[:200], probability[:200], rtol=1e-9)    # nothing resolves before hour 200
        self.assertFalse(np.allclose(early[2:201], probability[2:201]))

    def test_learns_a_site_shift(self):
        hours, probability, outcome = shifted_stream(20000, 0.5)
        calibrated = online_platt_scaling(anchor_hours=hours, resolve_hours=hours + 3.0, probability=probability,
                                          outcome=outcome, config=OnlinePlattConfig())
        later = slice(5000, None)
        self.assertGreater(calibration(probability[later], outcome[later])["citl"], 0.4)
        self.assertLess(abs(calibration(calibrated[later], outcome[later])["citl"]), 0.1)

    def test_projection_stays_in_the_ball(self):
        matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
        projected = project(np.array([300.0, -50.0]), matrix, 100.0)
        self.assertAlmostEqual(float(np.linalg.norm(projected)), 100.0, places=6)
        np.testing.assert_array_equal(project(np.array([1.0, 0.0]), matrix, 100.0), [1.0, 0.0])

    def test_invalid_probabilities_get_nan_and_never_update(self):
        hours, probability, outcome = shifted_stream(300, 1.0)
        broken = probability.copy()
        broken[[10, 20]] = [np.nan, 1.5]
        skipped = outcome.copy()
        skipped[[10, 20]] = np.nan
        config = OnlinePlattConfig()
        with_invalid = online_platt_scaling(anchor_hours=hours, resolve_hours=hours + 1.0, probability=broken,
                                            outcome=outcome, config=config)
        without_update = online_platt_scaling(anchor_hours=hours, resolve_hours=hours + 1.0, probability=probability,
                                              outcome=skipped, config=config)
        self.assertTrue(np.isnan(with_invalid[[10, 20]]).all())
        others = np.setdiff1d(np.arange(300), [10, 20])
        np.testing.assert_allclose(with_invalid[others], without_update[others], rtol=1e-12)

    def test_calibration_gate(self):
        self.assertTrue(calibration_pass({"identifiable": True, "citl": 0.1, "slope": 1.2}))
        self.assertFalse(calibration_pass({"identifiable": True, "citl": 0.25, "slope": 1.0}))
        self.assertFalse(calibration_pass({"identifiable": True, "citl": 0.0, "slope": 0.75}))
        self.assertFalse(calibration_pass({"identifiable": False, "citl": None, "slope": None}))


if __name__ == "__main__":
    unittest.main()
