import unittest

import pandas as pd

from osler_jepa.cohort_signature import cohort_signature


class CohortSignatureTests(unittest.TestCase):
    def test_row_order_does_not_change_signature(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["b", "a"],
                "stay_id": ["2", "1"],
                "t": pd.to_datetime(["2020-01-02", "2020-01-01"]),
                "_source_horizon": [6.0, 6.0],
                "glucose_t": [100.0, 90.0],
                "future_glucose": [95.0, 85.0],
                "future_delta_t_hr_glucose": [6.1, 5.9],
                "future_label_weight_glucose": [1.0, 1.0],
            }
        )
        self.assertEqual(
            cohort_signature(frame, ("glucose",)),
            cohort_signature(frame.iloc[::-1].reset_index(drop=True), ("glucose",)),
        )

    def test_supervision_change_changes_signature(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["a"],
                "stay_id": ["1"],
                "t": pd.to_datetime(["2020-01-01"]),
                "_source_horizon": [6.0],
                "glucose_t": [90.0],
                "future_glucose": [85.0],
                "future_label_weight_glucose": [1.0],
            }
        )
        changed = frame.copy()
        changed.loc[0, "future_glucose"] = 86.0
        self.assertNotEqual(
            cohort_signature(frame, ("glucose",)),
            cohort_signature(changed, ("glucose",)),
        )


if __name__ == "__main__":
    unittest.main()
