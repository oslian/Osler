import unittest

import pandas as pd

from eicu_within_bin_gap_diagnostic import absent_inputs, input_columns


class GapDiagnosticTests(unittest.TestCase):
    def test_absent_inputs_skip_target_and_flag_missing_or_empty_columns(self):
        mimic = pd.DataFrame({
            "t": [1, 2], "glucose_t": [1.0, 2.0], "glucose_age_hr": [0.5, 0.5],
            "lactate_t": [1.0, 2.0], "pao2_t": [80.0, 90.0], "hist_insulin": [0.0, 1.0],
            "future_glucose": [1.0, 2.0],
        })
        eicu = pd.DataFrame({"glucose_t": [1.0], "lactate_t": [2.0], "pao2_t": [float("nan")]})
        self.assertNotIn("t", input_columns(mimic))
        self.assertNotIn("future_glucose", input_columns(mimic))
        self.assertEqual(sorted(absent_inputs(mimic, eicu, "glucose")), ["hist_insulin", "pao2_t"])


if __name__ == "__main__":
    unittest.main()
