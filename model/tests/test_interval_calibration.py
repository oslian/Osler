import unittest

import numpy as np

from scripts.calibrate_release_intervals import (
    _adaptive_patient_cluster_radius,
    _patient_cluster_radius,
)
from scripts.audit_normalized_intervals import _fit_scale_model, _predict_scale


class IntervalCalibrationTests(unittest.TestCase):
    def test_patient_cluster_radius_limits_dense_patient_influence(self):
        residual = np.concatenate([np.zeros(100), np.asarray([10.0] * 9)])
        subjects = np.asarray(["dense"] * 100 + [f"p{i}" for i in range(9)])

        radius = _patient_cluster_radius(residual, subjects, 0.90)

        self.assertEqual(radius, 10.0)

    def test_adaptive_selection_is_deterministic(self):
        rng = np.random.default_rng(42)
        subjects = np.repeat(np.arange(80).astype(str), 3)
        residual = np.abs(rng.normal(size=len(subjects)))

        first = _adaptive_patient_cluster_radius(residual, subjects, seed=91)
        second = _adaptive_patient_cluster_radius(residual, subjects, seed=91)

        self.assertEqual(first, second)
        self.assertFalse(first[1]["selection_test_labels_used"])

    def test_scale_model_tracks_heteroscedastic_residual(self):
        feature = np.linspace(0.0, 2.0, 240)[:, None]
        residual = 0.2 + feature[:, 0]

        model = _fit_scale_model(feature, residual)
        predicted = _predict_scale(model, np.asarray([[0.1], [1.9]]))

        self.assertGreater(predicted[1], predicted[0])


if __name__ == "__main__":
    unittest.main()
