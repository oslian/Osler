from __future__ import annotations

import unittest

import numpy as np

from whole_body_jepa.probability_calibration import (
    fit_probability_calibrator,
    patient_equal_weights,
)


class ProbabilityCalibrationTest(unittest.TestCase):
    def test_patient_weights_equalize_long_stays(self) -> None:
        subjects = np.asarray(["a", "a", "a", "b"])
        weights = patient_equal_weights(subjects)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3:].sum()))

    def test_all_calibrators_return_probabilities(self) -> None:
        logits = np.asarray([-3, -2, -1, 1, 2, 3] * 4, dtype=float)
        labels = np.asarray([0, 0, 0, 1, 1, 1] * 4)
        subjects = np.asarray([f"p{index // 2}" for index in range(len(labels))])
        for method in ("temperature", "platt", "beta", "isotonic"):
            calibrator = fit_probability_calibrator(method, logits, labels, subjects)
            probability = calibrator.predict(logits)
            self.assertTrue(np.all(np.isfinite(probability)))
            self.assertTrue(np.all((probability >= 0) & (probability <= 1)))


if __name__ == "__main__":
    unittest.main()
