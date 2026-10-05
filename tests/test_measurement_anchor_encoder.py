import unittest

import numpy as np
import pandas as pd

from scripts.audit_measurement_anchor_encoder import (
    common_patients,
    eligible_rows,
    target_only_arrays,
)


class MeasurementAnchorEncoderTests(unittest.TestCase):
    def test_target_only_removes_only_non_target_values(self):
        arrays = {
            "input": np.arange(24, dtype=np.float32).reshape(2, 12),
            "current": np.arange(6, dtype=np.float32).reshape(2, 3),
            "mask": np.ones((2, 3), dtype=np.float32),
        }
        result = target_only_arrays(arrays, 1)
        np.testing.assert_array_equal(result["input"][:, 1], arrays["input"][:, 1])
        np.testing.assert_array_equal(result["input"][:, 4:], arrays["input"][:, 4:])
        np.testing.assert_array_equal(result["input"][:, [0, 2]], 0)
        np.testing.assert_array_equal(result["current"][:, [0, 2]], 0)
        np.testing.assert_array_equal(result["mask"], arrays["mask"])
        self.assertFalse(np.shares_memory(result["input"], arrays["input"]))

    def test_fresh_eligibility_requires_real_zero_age_anchor(self):
        frame = pd.DataFrame({
            "creatinine_t": [1.0, 1.0, 1.0],
            "creatinine_age_hr": [0.0, 1.0, 0.0],
            "future_creatinine": [1.1, 1.1, 1.1],
            "future_label_quality_creatinine": ["strict"] * 3,
            "future_label_weight_creatinine": [1.0] * 3,
            "future_delta_t_hr_creatinine": [6.0, 6.0, 8.0],
            "future_event_time_creatinine": pd.to_datetime([
                "2026-01-01 06:00", "2026-01-01 06:00", "2026-01-01 08:00"
            ]),
            "t": pd.to_datetime(["2026-01-01 00:00"] * 3),
        })
        np.testing.assert_array_equal(eligible_rows(frame, fresh=True), [True, False, False])
        np.testing.assert_array_equal(eligible_rows(frame, fresh=False), [True, True, False])

    def test_common_support_is_intersection_only(self):
        left = pd.DataFrame({"subject_id": ["a", "b", "c"]})
        right = pd.DataFrame({"subject_id": ["b", "c", "d"]})
        result = common_patients(
            left, np.array([True, True, False]), right, np.array([True, False, True])
        )
        self.assertEqual(result, {"b"})


if __name__ == "__main__":
    unittest.main()
