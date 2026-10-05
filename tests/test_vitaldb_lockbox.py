from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from whole_body_jepa.raw_waveform import build_raw_waveform_tensor
from whole_body_jepa.vitaldb_lockbox import eligible_case_ids, numeric_history_features


class VitalDBLockboxTest(unittest.TestCase):
    def test_case_selection_requires_portable_tracks(self) -> None:
        cases = pd.DataFrame(
            {"caseid": [1, 2], "caseend": [3600, 3600], "subjectid": [10, 20]}
        )
        required = [
            "Solar8000/HR",
            "Solar8000/PLETH_SPO2",
            "SNUADC/ECG_II",
            "SNUADC/PLETH",
        ]
        tracks = pd.DataFrame(
            [(1, name) for name in required] + [(2, name) for name in required[:-1]],
            columns=["caseid", "tname"],
        )
        self.assertEqual(set(eligible_case_ids(cases, tracks)), {1})

    def test_numeric_history_uses_only_pre_anchor_values(self) -> None:
        values = np.full((1000, 8), np.nan)
        values[:901, 0] = np.arange(901)
        values[901:, 0] = 99999.0
        features = numeric_history_features(values, 900.0)
        self.assertEqual(features["num_heart_rate_300s_last"], 900.0)
        self.assertLess(features["num_heart_rate_300s_median"], 99999.0)

    def test_array_tensor_matches_canonical_shape(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(size=(2000, 4))
        tensor = build_raw_waveform_tensor(
            values,
            ["ECG II", "PLETH", "RESP", "ART"],
            2.0,
            1800,
            points_per_scale=64,
            future_seconds=30,
        )
        self.assertEqual(tensor["past"].shape, (3, 4, 64))
        self.assertEqual(tensor["future"].shape, (4, 64))


if __name__ == "__main__":
    unittest.main()
