import unittest

import numpy as np
import pandas as pd

from build_expert_anchor_columns import anchor_predictions


def mimic_raw(stays=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for stay in range(stays):
        level = rng.normal(4.0, 0.4)
        for k in range(8):
            value = level + rng.normal(0, 0.2)
            rows.append({"subject_id": 1000 + stay, "stay_id": 5000 + stay, "first_careunit": "MICU",
                         "t": pd.Timestamp("2100-01-01") + pd.Timedelta(hours=6 * k),
                         "potassium_t": value, "potassium_age_hr": 0.5, "sodium_t": 140 + rng.normal(0, 2), "sodium_age_hr": 1.0,
                         "potassium_tp6": 0.7 * value + 0.3 * level + rng.normal(0, 0.1), "potassium_tp6_offset_hr": 0.5})
    return pd.DataFrame(rows)


def events(future):
    times = pd.Timestamp("2100-01-01") + pd.to_timedelta(np.arange(4) * 6, unit="h")
    return pd.DataFrame({"database": "eicu", "subject_id": "p1", "stay_id": "s1", "anchor_time": times, "horizon_hours": 6,
                         "potassium_t": [3.2, 3.8, 4.6, 5.1], "potassium_age_hr": 0.5, "sodium_t": 139.0, "sodium_age_hr": 1.0,
                         "future_potassium": future}, index=[10, 11, 12, 13])


class ExpertAnchorInvarianceTests(unittest.TestCase):
    def anchors(self, future, exclude=frozenset()):
        return anchor_predictions(events(future), mimic_raw(), "potassium", 6, "measurement_pure_ridge", set(exclude))

    def test_anchors_ignore_future_labels(self):
        full = self.anchors([3.5, 4.0, 4.8, 5.0])
        partial = self.anchors([np.nan, 4.0, np.nan, np.nan])
        pd.testing.assert_series_equal(full, partial)
        self.assertFalse(np.allclose(full.to_numpy(), events([np.nan] * 4)["potassium_t"].to_numpy()))  # a fitted recipe, not persistence

    def test_excluded_subjects_leave_the_training_sample(self):
        base = self.anchors([np.nan] * 4)
        excluded = self.anchors([np.nan] * 4, exclude={str(1000 + s) for s in range(30)})
        self.assertFalse(np.allclose(base.to_numpy(), excluded.to_numpy()))


if __name__ == "__main__":
    unittest.main()
