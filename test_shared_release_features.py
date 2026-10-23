import unittest

import numpy as np
import pandas as pd

from osler_jepa.event_ledger import CanonicalEvent, EventKind, EventLedger
from osler_jepa.shared_release_features import (
    FEATURE_SLICES,
    build_target_head_features,
    normalize_measurement_state,
)


class SharedReleaseFeatureTests(unittest.TestCase):
    def test_contract_places_base_forecast_and_current_target_separately(self):
        features = build_target_head_features(
            shared_context=np.zeros(64),
            patient_belief=np.ones(96),
            target_future=np.full(64, 2.0),
            target_organ=np.full(32, 3.0),
            base_forecast_normalized=4.0,
            measurement_pure_state=np.full(53, 5.0),
            target_current_normalized=6.0,
        )
        self.assertEqual(features.shape, (311,))
        self.assertEqual(features[FEATURE_SLICES["base_point"]][0], 4.0)
        self.assertEqual(features[FEATURE_SLICES["target_current"]][0], 6.0)

    def test_physical_zero_is_observed_and_missing_is_training_median(self):
        variables = tuple(f"v{index}" for index in range(53))
        state = normalize_measurement_state(
            {"v0": 0.0, "v1": None},
            variables=variables,
            medians=np.ones(53) * 10.0,
            scales=np.ones(53) * 2.0,
        )
        self.assertTrue(state.observed[0])
        self.assertEqual(state.normalized[0], -5.0)
        self.assertFalse(state.observed[1])
        self.assertEqual(state.normalized[1], 0.0)

    def test_anchor_view_carries_only_recent_pre_anchor_measurement(self):
        ledger = EventLedger(
            [
                CanonicalEvent(
                    patient_id="p1",
                    encounter_id="e1",
                    event_time=pd.Timestamp("2026-01-01 00:00"),
                    variable="urine_output",
                    kind=EventKind.MEASUREMENT,
                    value=0.0,
                )
            ]
        )
        recent = ledger.anchor_view(
            "e1",
            pd.Timestamp("2026-01-01 05:00"),
            ("urine_output",),
            max_age_hours=6.0,
        )
        stale = ledger.anchor_view(
            "e1",
            pd.Timestamp("2026-01-01 07:00"),
            ("urine_output",),
            max_age_hours=6.0,
        )
        self.assertTrue(recent["observed"]["urine_output"])
        self.assertEqual(recent["values"]["urine_output"], 0.0)
        self.assertFalse(stale["observed"]["urine_output"])
        self.assertIsNone(stale["values"]["urine_output"])


if __name__ == "__main__":
    unittest.main()
