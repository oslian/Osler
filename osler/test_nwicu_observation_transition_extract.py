import unittest

import pandas as pd

from nwicu_observation_transition_extract import derive_map


class NwicuExtractTests(unittest.TestCase):
    def test_map_is_derived_from_same_time_pairs_only(self):
        chart = pd.DataFrame({
            "subject_id": [1, 1, 1, 1, 1],
            "stay_id": [10, 10, 10, 10, 10],
            "charttime": pd.to_datetime([
                "2100-01-01 00:00", "2100-01-01 00:00",  # arterial pair
                "2100-01-01 01:00",                      # unpaired systolic
                "2100-01-01 02:00", "2100-01-01 02:00",  # non-invasive pair
            ]),
            "itemid": [320050, 320051, 320179, 320179, 320180],
            "valuenum": [120.0, 60.0, 110.0, 150.0, 90.0],
        })
        derived = derive_map(chart).sort_values("charttime")
        self.assertEqual(derived["valuenum"].round(6).tolist(), [80.0, 110.0])
        self.assertTrue((derived["var"] == "map").all())

    def test_inverted_pair_is_rejected(self):
        chart = pd.DataFrame({
            "subject_id": [1, 1],
            "stay_id": [10, 10],
            "charttime": pd.to_datetime(["2100-01-01 00:00"] * 2),
            "itemid": [320179, 320180],
            "valuenum": [60.0, 90.0],
        })
        self.assertTrue(derive_map(chart).empty)


if __name__ == "__main__":
    unittest.main()
