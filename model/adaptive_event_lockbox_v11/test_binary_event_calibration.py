import unittest

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from osler_jepa.event_calibration import calibration, cluster_bootstrap, group_calibration


def simulated(slope: float, intercept: float, n: int = 40000, seed: int = 0):
    rng = np.random.default_rng(seed)
    truth = expit(rng.normal(-2.0, 1.0, n))
    outcome = (rng.random(n) < truth).astype(float)
    probability = expit((logit(truth) - intercept) / slope)      # true logit = intercept + slope * logit(p)
    return probability, outcome


class CalibrationTests(unittest.TestCase):
    def test_well_calibrated_predictions(self):
        result = calibration(*simulated(1.0, 0.0))
        self.assertAlmostEqual(result["slope"], 1.0, delta=0.05)
        self.assertAlmostEqual(result["citl"], 0.0, delta=0.05)
        self.assertAlmostEqual(result["observed_expected"], 1.0, delta=0.05)

    def test_too_extreme_predictions_have_slope_below_one(self):
        self.assertAlmostEqual(calibration(*simulated(0.7, 0.0))["slope"], 0.7, delta=0.05)

    def test_underpredicted_risk_has_positive_citl(self):
        result = calibration(*simulated(1.0, 0.5))
        self.assertAlmostEqual(result["citl"], 0.5, delta=0.06)
        self.assertGreater(result["observed_expected"], 1.0)

    def test_cluster_bootstrap_covers_the_truth_and_needs_two_clusters(self):
        probability, outcome = simulated(1.0, 0.0, n=6000)
        frame = pd.DataFrame({"subject_id": np.arange(6000) // 4, "hospitalid": "one",
                              "probability": probability, "outcome": outcome})
        patient = cluster_bootstrap(frame, "subject_id", seed=1)
        self.assertLessEqual(patient["citl"][0], 0.0)
        self.assertGreaterEqual(patient["citl"][1], 0.0)
        self.assertIsNone(cluster_bootstrap(frame, "hospitalid", seed=1)["slope"])


class EdgeCaseTests(unittest.TestCase):
    def test_unidentifiable_streams_return_instead_of_raising(self):
        for probability, outcome in ((np.array([]), np.array([])),
                                     (np.full(50, 0.2), np.zeros(50)),
                                     (np.full(50, 0.2), np.ones(50)),
                                     (np.array([0.2, np.nan, 0.3]), np.array([0.0, 1.0, 0.0])),
                                     (np.array([0.2, 1.2, 0.3]), np.array([0.0, 1.0, 0.0]))):
            result = calibration(probability, outcome)
            self.assertFalse(result["identifiable"])
            self.assertIsNone(result["citl"])

    def test_extreme_probabilities_have_a_finite_citl(self):
        probability = np.full(1000, 1e-6)
        outcome = np.r_[np.ones(500), np.zeros(500)]
        result = calibration(probability, outcome)
        self.assertTrue(result["identifiable"])
        self.assertAlmostEqual(result["citl"], -logit(1e-6), places=4)      # observed rate 0.5 at logit(p) = logit(1e-6)

    def test_group_calibration_needs_support(self):
        probability, outcome = simulated(1.0, 0.0, n=3000)
        frame = pd.DataFrame({"unit": np.where(np.arange(3000) < 100, "small", "large"),
                              "probability": probability, "outcome": outcome})
        groups = {row["group"]: row for row in group_calibration(frame, "unit")}
        self.assertIsNone(groups["small"]["estimate"])
        self.assertTrue(groups["large"]["estimate"]["identifiable"])
        self.assertEqual(groups["small"]["rows"] + groups["large"]["rows"], 3000)


if __name__ == "__main__":
    unittest.main()
