import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mimiciv_failed_cell_hurdle_audit import cell_frame


class MimicIVFailedCellHurdleAuditTests(unittest.TestCase):
    def test_cell_frame_uses_exact_target_and_horizon(self):
        source = pd.DataFrame({
            "subject_id": [1],
            "stay_id": [2],
            "t": ["2026-01-01"],
            "first_careunit": ["MICU"],
            "creatinine_t": [1.0],
            "creatinine_tp24": [1.3],
            "creatinine_tp48": [1.8],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.parquet"
            source.to_parquet(path, index=False)
            frame = cell_frame(path, "creatinine", 24)
        self.assertEqual(frame.loc[0, "future_creatinine"], 1.3)
        self.assertEqual(frame.loc[0, "horizon_hours"], 24)


if __name__ == "__main__":
    unittest.main()
