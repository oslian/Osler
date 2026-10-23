import unittest

import numpy as np
import pandas as pd

from site_adaptive_interval_dev import pseudo_sites


class PseudoSiteTests(unittest.TestCase):
    def test_pseudo_sites_never_split_a_stay_and_cover_every_row(self):
        frame = pd.DataFrame({"stay_id": np.repeat(np.arange(10), 7).astype(str)})
        rows = np.arange(len(frame))
        sites = pseudo_sites(frame, rows, 20, seed=0)
        np.testing.assert_array_equal(np.sort(np.concatenate(sites)), rows)
        owners = [set(frame.iloc[site]["stay_id"]) for site in sites]
        for a in range(len(owners)):
            for b in range(a + 1, len(owners)):
                self.assertFalse(owners[a] & owners[b])
        self.assertTrue(all(len(site) >= 20 for site in sites[:-1]))

    def test_pseudo_sites_are_seeded(self):
        frame = pd.DataFrame({"stay_id": np.repeat(np.arange(10), 7).astype(str)})
        first = pseudo_sites(frame, np.arange(len(frame)), 20, seed=3)
        again = pseudo_sites(frame, np.arange(len(frame)), 20, seed=3)
        for a, b in zip(first, again):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
