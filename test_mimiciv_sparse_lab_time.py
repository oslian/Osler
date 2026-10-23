import unittest

import numpy as np
import pandas as pd

from mimiciv_sparse_lab_irregular_time_audit import build_irregular_pairs
from mimiciv_sparse_lab_long_horizon_audit import _event_arrays, _time_ns, build_target_anchors


class MIMICSparseLabTimeTests(unittest.TestCase):
    def _events(self):
        base = pd.Timestamp("2020-01-01 00:00:00")
        times = [base + pd.Timedelta(hours=h) for h in (0, 5, 20, 50)]
        return pd.DataFrame({
            "subject_id": [1] * 4,
            "stay_id": [1] * 4,
            "first_careunit": ["MICU"] * 4,
            "intime": [base] * 4,
            "outtime": [base + pd.Timedelta(hours=72)] * 4,
            "charttime": pd.Series(times).astype("datetime64[us]"),
            "target": ["inr"] * 4,
            "value": [1.0, 1.2, 1.5, 1.8],
        })

    def test_time_unit_normalization(self):
        values = pd.Series(pd.to_datetime(["2020-01-01 00:00", "2020-01-01 06:00"])).astype("datetime64[us]")
        delta = np.diff(_time_ns(values))[0] / 3.6e12
        self.assertAlmostEqual(delta, 6.0)

    def test_endpoint_builder_uses_real_time_units(self):
        events = self._events()
        arrays = _event_arrays(events)
        anchors = build_target_anchors(events, "inr", (6, 24, 48), arrays, endpoint=True)
        ages = anchors["inr_future_age_hr_6"].dropna().to_numpy()
        self.assertTrue(len(ages) > 0)
        self.assertTrue(np.all((ages >= 4.5) & (ages <= 7.5)))

    def test_irregular_builder_keeps_multiple_future_events(self):
        events = self._events()
        pairs = build_irregular_pairs(events, "inr", _event_arrays(events))
        self.assertGreaterEqual(len(pairs), 3)
        self.assertGreater(pairs["forecast_horizon_hr"].max(), 24.0)
        self.assertTrue((pairs["forecast_horizon_hr"] >= 0.5).all())


if __name__ == "__main__":
    unittest.main()
