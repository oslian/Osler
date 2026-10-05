import unittest
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from whole_body_jepa.continual_learning import (
    ADAPTER_Z_LIMIT,
    ContinualArms,
    PatientResidualState,
    Reservoir,
    DecayedPatientResidualState,
    TimeAwarePatientResidualState,
    UPDATE_BATCH_SIZE,
    initialize_zero_adapter,
)


class ContinualLearningTests(unittest.TestCase):
    def test_reservoir_is_bounded_and_deterministic(self):
        first = Reservoir(3, np.random.default_rng(7))
        second = Reservoir(3, np.random.default_rng(7))
        for index in range(20):
            first.add(np.asarray([index]), index)
            second.add(np.asarray([index]), index)
        self.assertEqual(len(first.items), 3)
        self.assertEqual([item[1] for item in first.items], [item[1] for item in second.items])

    def test_patient_state_uses_only_resolved_prior_errors(self):
        state = PatientResidualState(prior_strength=3)
        self.assertEqual(state.correction, 0.0)
        state.update(4.0)
        self.assertEqual(state.correction, 1.0)
        state.update(-1.0)
        self.assertEqual(state.correction, 0.6)

    def test_all_weight_updates_have_equal_batch_budget(self):
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        labels = np.asarray([0.0, 0.1, -0.1, 0.2])
        scaler = StandardScaler().fit(features)
        model = SGDRegressor(random_state=1).fit(scaler.transform(features), labels)
        arms = ContinualArms.from_initial(
            scaler, model, 1.0, features, labels, seed=5
        )
        sizes = []
        original = arms._update_model

        def recording(model, batch):
            sizes.append(len(batch))
            original(model, batch)

        arms._update_model = recording
        arms.update(np.asarray([4.0]), 0.3)
        self.assertEqual(sizes, [UPDATE_BATCH_SIZE] * 3)

    def test_online_prediction_is_bounded_in_residual_scale_units(self):
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        labels = np.asarray([0.0, 0.1, -0.1, 0.2])
        scaler = StandardScaler().fit(features)
        model = SGDRegressor(random_state=1).fit(scaler.transform(features), labels)
        model.coef_[:] = 1e6
        arms = ContinualArms.from_initial(
            scaler, model, 2.0, features, labels, seed=5
        )
        value = arms.predict_correction("naive_online", np.asarray([4.0]))
        self.assertEqual(value, ADAPTER_Z_LIMIT * 2.0)

    def test_zero_adapter_preserves_the_release_prediction(self):
        features = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
        residuals = np.asarray([-2.0, 1.0, 3.0])
        scaler, model, scale = initialize_zero_adapter(
            features, residuals, seed=7
        )
        prediction = model.predict(scaler.transform(features)) * scale
        np.testing.assert_allclose(prediction, 0.0)

    def test_time_aware_state_uses_only_resolved_history(self):
        state = TimeAwarePatientResidualState(
            reference_scale=10.0,
            half_life_hours=24.0,
            prior_strength=3.0,
            slope_penalty=16.0,
        )
        anchor = datetime(2026, 1, 1, 12)
        self.assertEqual(state.estimate(anchor, 6.0), (0.0, 10.0))

        state.update(anchor - timedelta(hours=6), 8.0)
        correction, scale = state.estimate(anchor, 6.0)

        self.assertGreater(correction, 0.0)
        self.assertLess(correction, 8.0)
        self.assertGreater(scale, 0.0)

    def test_time_aware_state_rejects_future_outcome(self):
        state = TimeAwarePatientResidualState(
            reference_scale=10.0,
            half_life_hours=24.0,
            prior_strength=3.0,
            slope_penalty=16.0,
        )
        anchor = datetime(2026, 1, 1, 12)
        state.update(anchor + timedelta(hours=1), 2.0)

        with self.assertRaisesRegex(ValueError, "unresolved future"):
            state.estimate(anchor, 6.0)

    def test_decayed_state_does_not_extrapolate_a_slope(self):
        state = DecayedPatientResidualState(
            reference_scale=10.0,
            half_life_hours=24.0,
            prior_strength=1.0,
        )
        anchor = datetime(2026, 1, 1, 12)
        state.update(anchor - timedelta(hours=12), 8.0)
        correction, scale = state.estimate(anchor)
        later_correction, later_scale = state.estimate(anchor + timedelta(hours=6))

        self.assertGreater(correction, 0.0)
        self.assertLess(later_correction, correction)
        self.assertGreater(scale, 0.0)
        self.assertGreater(later_scale, 0.0)


if __name__ == "__main__":
    unittest.main()
