import unittest

import numpy as np
import pandas as pd

from site_adaptive_interval_dev import network_replay, replay
from site_adaptive_network_dev import CANDIDATES, choose, half


def frame_with(hospitals: dict[str, int], rows_per_stay: int = 3) -> pd.DataFrame:
    records = []
    for hospital, stays in hospitals.items():
        for stay in range(stays):
            for hour in range(rows_per_stay):
                records.append({"hospitalid": hospital, "stay_id": f"{hospital}-{stay}", "hours_since_onset": 6.0 * hour,
                                "future_k_available_hr": 6.5})
    return pd.DataFrame(records)


class NetworkReplayTests(unittest.TestCase):
    def test_network_merges_hospital_clocks_and_keeps_each_stay_chronological(self):
        frame = frame_with({"h1": 20, "h2": 5})
        rows = np.arange(len(frame))
        hospitals = frame["hospitalid"].to_numpy()
        sites = [replay(frame, rows[hospitals == h], "k", seed=i) for i, h in enumerate(("h1", "h2"))]
        network = network_replay(frame, sites)
        self.assertEqual(sorted(network.rows), list(rows))
        self.assertTrue(np.all(np.diff(network.clock) >= 0))                 # one clock
        onset = frame.iloc[network.rows].groupby("stay_id", sort=False)["hours_since_onset"].apply(lambda t: t.is_monotonic_increasing)
        self.assertTrue(onset.all())
        np.testing.assert_allclose(network.resolve - network.clock, 6.5)
        self.assertEqual(int(network.warm.sum()), int(sum(site.warm.sum() for site in sites)))  # each hospital keeps its warm-up

    def test_split_is_fixed_and_covers_both_halves(self):
        labels = {half(str(h)) for h in range(50)}
        self.assertEqual(labels, {"A", "B"})
        self.assertEqual(half("73"), half("73"))

    def test_choose_breaks_ties_to_current(self):
        cells = [{name: {"A": {"interval_robust": True}, "B": {"interval_robust": False}} for name in CANDIDATES}]
        self.assertEqual(choose(cells, "A"), "current")


if __name__ == "__main__":
    unittest.main()
