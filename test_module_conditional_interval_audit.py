import unittest

import numpy as np
import pandas as pd

from module_conditional_interval_audit import group_widths, raw_stay


class ModuleConditionalTests(unittest.TestCase):
    def test_raw_stay_strips_database_prefix(self):
        frame = pd.DataFrame({"stay_id": ["MIMIC:30000153", "EICU:141203", "NWICU:7.0"]})
        np.testing.assert_array_equal(raw_stay(frame), [30000153, 141203, 7])

    def test_group_widths_use_only_the_given_pool(self):
        n = 400
        frame = pd.DataFrame({
            "subject_id": [f"MIMIC:{i}" for i in range(n)],
            "creatinine_t": np.tile(np.linspace(0.5, 4.0, 10), n // 10),
        })
        rows = np.arange(n)
        small = np.abs(np.random.default_rng(1).normal(size=n))
        widths_small = group_widths(frame, rows, small, rows[:5], "creatinine")
        widths_large = group_widths(frame, rows, small * 3, rows[:5], "creatinine")
        np.testing.assert_allclose(widths_large, widths_small * 3)


if __name__ == "__main__":
    unittest.main()
