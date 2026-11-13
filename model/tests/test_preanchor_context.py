import unittest

import numpy as np
import pandas as pd

from whole_body_jepa.preanchor_context import (
    CONTEXT_CATEGORIES,
    attach_eicu_infusion_history,
    build_preanchor_context,
    summarize_event_offsets,
)


class PreAnchorContextTests(unittest.TestCase):
    def test_future_events_are_excluded(self) -> None:
        events = {category: {} for category in CONTEXT_CATEGORIES}
        events["renal_support"][7] = np.asarray([-5.0, 55.0, 65.0])
        matrix = summarize_event_offsets([7], [60.0], events)
        self.assertAlmostEqual(float(matrix[0, CONTEXT_CATEGORIES.index("renal_support") * 4]), np.log1p(1.0))
        self.assertAlmostEqual(
            float(matrix[0, CONTEXT_CATEGORIES.index("renal_support") * 4 + 1]),
            np.log1p(2.0),
        )

    def test_context_uses_history_columns_only(self) -> None:
        frame = pd.DataFrame(
            {
                "_raw_stay_id": [7],
                "hist_fluids": [1],
                "hist_fluids_evidence_count": [2],
                "hist_fluids_dose_observed": [1],
                "act_fluids": [999],
            }
        )
        events = {category: {} for category in CONTEXT_CATEGORIES}
        matrix, names = build_preanchor_context(frame, [60.0], events)
        self.assertEqual(matrix.shape[1], len(names))
        self.assertNotIn("act_fluids", names)
        index = names.index("context_treatment_fluids_present")
        self.assertEqual(float(matrix[0, index]), 1.0)

    def test_precise_treatment_uses_history_only(self) -> None:
        frame = pd.DataFrame(
            {
                "_raw_stay_id": [7],
                "hist_fluids_rate_volume_ml_per_hour_time_weighted_mean": [99.0],
                "hist_fluids_current_rate_volume_ml_per_hour": [50.0],
                "hist_fluids_delivered_volume_ml_sum": [200.0],
                "act_fluids_current_rate_volume_ml_per_hour": [999.0],
            }
        )
        events = {category: {} for category in CONTEXT_CATEGORIES}
        matrix, names = build_preanchor_context(frame, [60.0], events)
        mean_index = names.index(
            "context_treatment_fluids_mean_rate_volume_ml_per_hour_log1p"
        )
        self.assertAlmostEqual(float(matrix[0, mean_index]), np.log1p(99.0), places=6)
        self.assertFalse(any(name.startswith("act_") for name in names))

    def test_eicu_rate_summary_truncates_at_anchor(self) -> None:
        frame = pd.DataFrame({"_raw_stay_id": [7]})
        infusions = pd.DataFrame(
            {
                "stay_id": [7, 7],
                "action": ["fluids", "fluids"],
                "start": [55.0, 65.0],
                "end": [70.0, 80.0],
                "rate": [120.0, 500.0],
                "dimension": ["volume_ml_per_hour", "volume_ml_per_hour"],
            }
        )
        output = attach_eicu_infusion_history(frame, [60.0], infusions)
        self.assertEqual(
            float(output.loc[0, "hist_fluids_current_rate_volume_ml_per_hour"]),
            120.0,
        )
        self.assertEqual(
            float(output.loc[0, "hist_fluids_delivered_volume_ml_sum"]),
            10.0,
        )

    def test_future_snapshot_cannot_extend_old_rate(self) -> None:
        frame = pd.DataFrame({"_raw_stay_id": [7]})
        snapshots = pd.DataFrame({
            "stay_id": [7, 7], "action": ["fluids", "fluids"],
            "label": ["NS", "NS"], "start": [0.0, 180.0],
            "dimension": ["volume_ml_per_hour"] * 2, "rate": [100.0, 400.0],
        })
        before = attach_eicu_infusion_history(frame, [90.0], snapshots.iloc[:1])
        after = attach_eicu_infusion_history(frame, [90.0], snapshots)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(after.loc[0, "hist_fluids_current_rate_volume_ml_per_hour"], 0.0)

    def test_zero_snapshot_stops_rate_and_parallel_streams_sum(self) -> None:
        frame = pd.DataFrame({"_raw_stay_id": [7]})
        snapshots = pd.DataFrame({
            "stay_id": [7] * 3, "action": ["fluids"] * 3,
            "label": ["NS", "NS", "LR"], "start": [0.0, 30.0, 20.0],
            "dimension": ["volume_ml_per_hour"] * 3, "rate": [100.0, 0.0, 200.0],
        })
        result = attach_eicu_infusion_history(frame, [40.0], snapshots)
        self.assertEqual(result.loc[0, "hist_fluids_current_rate_volume_ml_per_hour"], 200.0)
        self.assertAlmostEqual(result.loc[0, "hist_fluids_delivered_volume_ml_sum"], 50 + 200/3)


if __name__ == "__main__":
    unittest.main()
