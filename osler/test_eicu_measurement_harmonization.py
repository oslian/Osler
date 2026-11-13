import tempfile
import unittest
from pathlib import Path

import pandas as pd

from eicu_body_system_configs import get_body_system_config
from eicu_body_system_transition_extract import input_state_vars
from eicu_sepsis_transition_extract import (
    LAB_TO_STATE,
    NURSE_CHART_TO_STATE,
    PLAUSIBLE,
    STATE_VARS,
    read_measurements,
)
from mimiciv_observation_transition_extract import LAB_ITEMIDS, STATE_VARS as MIMIC_STATE_VARS, VITAL_ITEMIDS
from nwicu_observation_transition_extract import NWICU_LAB_ITEMIDS


class MeasurementHarmonizationTests(unittest.TestCase):
    def test_bedside_glucose_is_never_the_laboratory_glucose_target(self):
        self.assertEqual(LAB_TO_STATE["bedside glucose"], "glucose_poc")
        self.assertEqual(LAB_TO_STATE["glucose"], "glucose")
        self.assertEqual(NURSE_CHART_TO_STATE[("Bedside Glucose", "Bedside Glucose")], "glucose_poc")
        self.assertIn("glucose_poc", PLAUSIBLE)
        self.assertIn("glucose_poc", STATE_VARS)

    def test_vitals_come_from_nurse_charting_means_only(self):
        variables = set(NURSE_CHART_TO_STATE.values())
        self.assertTrue({"heart_rate", "respiratory_rate", "o2sat", "map", "temperature"} <= variables)
        map_sources = [name for (label, name), var in NURSE_CHART_TO_STATE.items() if var == "map"]
        self.assertTrue(all("Systolic" not in name and "Diastolic" not in name for name in map_sources))

    def test_mimic_point_of_care_glucose_is_a_separate_variable(self):
        self.assertEqual(VITAL_ITEMIDS["glucose_poc"], (225664,))
        self.assertIn("glucose_poc", MIMIC_STATE_VARS)

    def test_glucose_target_is_chemistry_only_and_blood_gas_is_separate(self):
        self.assertEqual(LAB_ITEMIDS["glucose"], (50931, 52569))
        self.assertEqual(LAB_ITEMIDS["glucose_bg"], (50809, 52027))
        self.assertEqual(NWICU_LAB_ITEMIDS["glucose"], (100001,))
        self.assertEqual(NWICU_LAB_ITEMIDS["glucose_bg"], (100045,))
        self.assertIn("glucose_bg", MIMIC_STATE_VARS)
        self.assertIn("glucose_bg", PLAUSIBLE)
        self.assertNotIn("glucose", [LAB_TO_STATE.get(name) for name in ("bedside glucose",)])

    def test_body_system_modules_add_point_of_care_glucose_beside_glucose(self):
        endocrine = input_state_vars(get_body_system_config("endocrine_stress"))
        self.assertIn("glucose", endocrine)
        self.assertEqual(endocrine.count("glucose_poc"), 1)

    def test_missing_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                read_measurements({1}, cache=Path(tmp) / "missing.parquet")

    def test_cache_reader_filters_stays_and_adds_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.parquet"
            pd.DataFrame({
                "stay_id": [1, 2], "offset": [60.0, 120.0], "var": ["glucose", "map"], "valuenum": [100.0, 70.0],
            }).to_parquet(cache, index=False)
            frame = read_measurements({2}, cache=cache)
            self.assertEqual(frame["stay_id"].tolist(), [2])
            self.assertEqual(frame["time_hr"].tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
