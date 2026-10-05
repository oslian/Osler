import unittest

import numpy as np
import pandas as pd

from whole_body_jepa.amplitude_calibration import adoption, fit_amplitude


class AmplitudeTests(unittest.TestCase):
    def test_signed_delta_and_state_dependent_overshoot(self):
        x = np.repeat([[-1.], [1.]], 500, axis=0)
        proposal = np.tile([-2., 2.], 500)
        actual = proposal * np.where(x[:, 0] < 0, .2, .8)
        model = fit_amplitude(x, actual, proposal)
        np.testing.assert_allclose(adoption(model, np.array([[-1.], [1.]])), [.2, .8], atol=.02)

    def test_weighted_median_not_unweighted_adoption(self):
        x = np.zeros((101, 1))
        proposal = np.r_[np.ones(100), 1000.]
        actual = np.r_[np.ones(100), 0.]
        model = fit_amplitude(x, actual, proposal, conditional=False)
        self.assertEqual(model["constant"], 0)

    def test_objective_equivalence_after_clipping(self):
        d = np.array([-2., 3., 1., -1.])
        y = np.array([1., 9., .2, -.9])
        r = np.clip(y / d, 0, 1)
        gaps = [np.abs(y - a * d) - np.abs(d) * np.abs(r - a) for a in (0, .2, .7, 1)]
        for gap in gaps[1:]:
            np.testing.assert_allclose(gap, gaps[0])

    def test_zero_and_wrong_direction_are_bounded(self):
        x = np.zeros((100, 1))
        head = fit_amplitude(x, np.ones(100), np.zeros(100))
        np.testing.assert_array_equal(adoption(head, x), 0)
        head = fit_amplitude(x, -np.ones(100), np.ones(100))
        np.testing.assert_array_equal(adoption(head, x), 0)
        head = fit_amplitude(x, np.full(100, 5.), np.ones(100))
        np.testing.assert_array_equal(adoption(head, x), 1)

    def test_invalid_arrays_rejected(self):
        for x, y, d in ((np.zeros((2, 1)), [1], [1, 1]),
                        (np.zeros((2, 1)), [np.nan, 1], [1, 1]),
                        (np.empty((0, 1)), [], [])):
            with self.assertRaises(ValueError):
                fit_amplitude(x, y, d)

    def test_zero_adoption_does_not_erase_uncertainty(self):
        from scripts.audit_amplitude_calibration import distribution, calibrate
        pred = {"expert": np.ones(100), "expert_samples": np.tile(np.linspace(-1, 3, 64), (100, 1)),
                "persistence_samples": np.linspace(-1, 1, 64)}
        alpha = {"conditional": np.zeros(100)}
        scales = calibrate(pred, alpha, np.linspace(-2, 2, 100))
        samples = distribution(pred, alpha, "conditional", scales["conditional"])
        self.assertTrue((np.ptp(samples, axis=1) > 3).all())
        np.testing.assert_allclose(np.median(samples, axis=1), 0, atol=1e-12)

    def test_calibration_and_heldout_audit(self):
        from scripts.audit_amplitude_calibration import LEARNED, calibrate, evaluate
        rng = np.random.default_rng(9)
        def batch(n):
            d = rng.normal(0, .4, n)
            delta = .4 * d + rng.normal(0, .1, n)
            pred = {"expert": d, "expert_samples": d[:, None] + np.linspace(-.3, .3, 64),
                    "persistence_samples": np.linspace(-.6, .6, 64)}
            alpha = {m: np.full(n, .4 if m == "conditional" else .6) for m in LEARNED}
            alpha.update(binary=np.ones(n), persistence=np.zeros(n), always_expert=np.ones(n))
            frame = pd.DataFrame({"subject_id": [str(i) for i in range(n)], "careunit": np.arange(n) % 4,
                                  "_time_hours": np.arange(n), "creatinine_t": 1., "future_creatinine": 1 + delta})
            return pred, alpha, delta, frame
        p, a, d, _ = batch(400)
        scales = calibrate(p, a, d)
        p, a, d, f = batch(400)
        result = evaluate(f, p, a, scales, "creatinine", "synthetic")
        self.assertAlmostEqual(result["methods"]["persistence"]["mae"], np.abs(d).mean())
        self.assertLess(result["methods"]["conditional"]["mae"], result["methods"]["binary"]["mae"])
        self.assertIn("global", result["methods"]["conditional"]["comparisons"])


if __name__ == "__main__":
    unittest.main()
