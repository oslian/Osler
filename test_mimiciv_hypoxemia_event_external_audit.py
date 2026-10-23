import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mimiciv_hypoxemia_event_external_audit import event_frame


class MimicIVHypoxemiaEventExternalAuditTests(unittest.TestCase):
    def test_event_frame_uses_matching_future_horizon(self):
        source = pd.DataFrame({
            "subject_id": [1],
            "stay_id": [2],
            "t": ["2026-01-01 00:00:00"],
            "first_careunit": ["MICU"],
            "o2sat_t": [96.0],
            "o2sat_tp3": [91.0],
            "o2sat_tp1": [95.0],
            "hist_oxygen": [1.0],
            "act_oxygen": [1.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.parquet"
            source.to_parquet(path, index=False)
            frame = event_frame(path, 3)

        self.assertEqual(frame.loc[0, "future_o2sat"], 91.0)
        self.assertEqual(frame.loc[0, "horizon_hours"], 3)
        self.assertEqual(frame.loc[0, "hospitalid"], "MIMIC-IV")


if __name__ == "__main__":
    unittest.main()
