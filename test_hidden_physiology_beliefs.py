import unittest

import numpy as np
import pandas as pd

from hidden_physiology_beliefs import (
    ALL_HIDDEN_PHYSIOLOGY_COLUMNS,
    AcidBaseBufferBelief,
    GlycemicClearanceBelief,
    hidden_physiology_belief_features,
    placebo_hidden_physiology_features,
)


class HiddenPhysiologyBeliefTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({
            "stay_id": [1, 1, 1, 2],
            "hours_since_onset": [0.0, 2.0, 6.0, 0.0],
            "glucose_t": [240.0, 180.0, 130.0, 220.0],
            "glucose_age_hr": [0.0, 0.0, 0.0, 0.0],
            "bicarbonate_t": [14.0, 17.0, 21.0, 12.0],
            "ph_t": [7.18, 7.25, 7.33, 7.12],
            "anion_gap_t": [24.0, 19.0, 14.0, 28.0],
            "lactate_t": [3.0, 2.5, 1.6, 4.0],
            "paco2_t": [28.0, 31.0, 35.0, 25.0],
            "hist_insulin": [1.0, 1.0, 1.0, 0.0],
            "act_insulin": [0.0, 1.0, 1.0, 0.0],
        })

    def test_features_are_finite_and_future_safe(self):
        frame = self._frame()
        features = hidden_physiology_belief_features(frame)
        self.assertEqual(tuple(features.columns), ALL_HIDDEN_PHYSIOLOGY_COLUMNS)
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertTrue((features["belief_glycemic_clearance_observed_insulin"].iloc[:3] == 1.0).all())
        self.assertGreater(features.loc[0, "belief_acid_base_buffer_stress"], features.loc[2, "belief_acid_base_buffer_stress"])

        with_future = frame.copy()
        with_future["glucose_tp6"] = [1.0, 1.0, 1.0, 1.0]
        future_features = hidden_physiology_belief_features(with_future)
        pd.testing.assert_frame_equal(features, future_features)

    def test_state_updates_in_time_order_and_isolated_by_stay(self):
        features = hidden_physiology_belief_features(self._frame())
        self.assertNotEqual(
            features.loc[0, "belief_glycemic_clearance_mean"],
            features.loc[3, "belief_glycemic_clearance_mean"],
        )
        self.assertGreaterEqual(features.loc[2, "belief_glycemic_clearance_confidence"], 0.05)

    def test_state_classes_predict_and_update_increase_uncertainty_on_predict(self):
        glycemic = GlycemicClearanceBelief(0.5, 0.04)
        predicted = glycemic.predict(6.0, drift=-0.01)
        updated = predicted.update(0.2, 0.8)
        self.assertGreater(predicted.variance, glycemic.variance)
        self.assertLessEqual(updated.mean, predicted.mean)

        buffer = AcidBaseBufferBelief(0.4, 0.04)
        self.assertGreater(buffer.predict(6.0).variance, buffer.variance)

    def test_placebo_is_capacity_matched(self):
        placebo = placebo_hidden_physiology_features(self._frame(), seed=17)
        self.assertEqual(placebo.shape, (4, len(ALL_HIDDEN_PHYSIOLOGY_COLUMNS)))
        self.assertTrue(np.isfinite(placebo.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
