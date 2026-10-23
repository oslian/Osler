import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import mimiciv_observation_transition_extract as contract
from eicu_sepsis_transition_extract import nearest_value
from emory_challenge2019_transition_extract import COLUMNS, build


class LabelTimingContractTests(unittest.TestCase):
    def test_nearest_value_reports_the_label_offset(self):
        times = np.array([10.0, 13.5, 20.0])
        values = np.array([1.0, 2.0, 3.0])
        self.assertEqual(nearest_value(times, values, 12.0, 2.0), (2.0, 1.5))     # later measurement is nearer
        value, offset = nearest_value(times, values, 16.5, 2.0)
        self.assertTrue(np.isnan(value) and np.isnan(offset))                       # nothing within tolerance

    def test_assemble_state_future_offset(self):
        anchors = pd.DataFrame({"stay_id": [1], "t_plus": [pd.Timestamp("2100-01-01 12:00")]})
        measurements = pd.DataFrame({"stay_id": [1, 1], "var": ["heart_rate", "heart_rate"], "valuenum": [80.0, 90.0],
                                     "charttime": [pd.Timestamp("2100-01-01 10:30"), pd.Timestamp("2100-01-01 13:00")]})
        state = contract.assemble_state(anchors, measurements, "_tp6", "t_plus", "nearest", 2.0)
        self.assertEqual(state.loc[0, "heart_rate_tp6"], 90.0)
        self.assertEqual(state.loc[0, "heart_rate_tp6_offset_hr"], 1.0)

    def test_anchors_without_any_label_are_kept(self):
        header = list(COLUMNS) + ["SBP", "DBP", "EtCO2", "BaseExcess", "SaO2", "Calcium", "Age", "Gender", "Unit1", "Unit2",
                                  "HospAdmTime", "ICULOS", "SepsisLabel"]
        with tempfile.TemporaryDirectory() as folder:
            table = pd.DataFrame(np.nan, index=range(40), columns=header)
            table["ICULOS"] = np.arange(1, 41)
            table["Unit1"], table["Unit2"] = 1.0, 0.0
            table.loc[:10, "HR"] = 80.0                     # measured only during the first 11 hours
            table.to_csv(Path(folder) / "p1.psv", sep="|", index=False, na_rep="NaN")
            frame, _ = build(Path(folder), 6.0, 6.0)
        self.assertEqual(len(frame), 5)                      # anchors at 6, 12, 18, 24, 30 h all kept
        self.assertEqual(int(frame["heart_rate_tp6"].notna().sum()), 1)
        self.assertIn("heart_rate_tp6_offset_hr", frame)


if __name__ == "__main__":
    unittest.main()
