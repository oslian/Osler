import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prepare_jepa_event_examples import prepare


class PrepareTests(unittest.TestCase):
    def test_converts_eicu_ionized_calcium_and_hides_excluded_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.parquet", Path(folder) / "b.parquet"
            pd.DataFrame({"database": ["mimiciv"], "ionized_calcium_t": [1.1], "future_ionized_calcium": [1.2],
                          "afp_t": [3.0], "future_afp": [4.0], "future_delta_t_hr_afp": [1.0]}).to_parquet(a)
            pd.DataFrame({"database": ["eicu"], "ionized_calcium_t": [4.008], "future_ionized_calcium": [8.016]}).to_parquet(b)
            frame, report = prepare([a, b])
        self.assertAlmostEqual(frame.loc[1, "ionized_calcium_t"], 1.0)
        self.assertAlmostEqual(frame.loc[1, "future_ionized_calcium"], 2.0)
        self.assertAlmostEqual(frame.loc[0, "ionized_calcium_t"], 1.1)
        self.assertIn("afp_t", frame)
        self.assertNotIn("future_afp", frame)
        self.assertNotIn("future_delta_t_hr_afp", frame)


if __name__ == "__main__":
    unittest.main()
