import unittest

import numpy as np
import pandas as pd

from activity_conditional_interval_audit import CRITERIA, active_rows


class ActivityRuleTests(unittest.TestCase):
    def test_aki_rule_uses_measurements_and_ignores_missing(self):
        frame = pd.DataFrame({
            "creatinine_t": [1.6, 1.0, np.nan, 1.0],
            "urine_output_t": [np.nan, 20.0, np.nan, 50.0],
            "potassium_t": [4.0, 4.0, np.nan, 4.0],
            "bun_t": [10.0, 10.0, np.nan, 10.0],
        })
        self.assertEqual(active_rows(frame, "aki").tolist(), [True, True, False, False])

    def test_rules_never_use_treatment_or_ventilator_columns(self):
        for module, criteria in CRITERIA.items():
            for var, _, _ in criteria:
                self.assertFalse(var.startswith(("hist_", "act_", "vent_mode")), (module, var))

    def test_body_system_thresholds_are_inclusive(self):
        frame = pd.DataFrame({"platelets_t": [100.0, 101.0]})
        self.assertEqual(active_rows(frame, "coagulopathy_heme").tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
