import unittest

import numpy as np
import pandas as pd

from mimic_crystalloid_tte_extract import OUTCOME_LABS, time_zero, with_provenance


def starts(rows):
    columns = ["stay_id", "orderid", "start", "weight", "kind", "bolus", "rate_ml_h"]
    return pd.DataFrame(rows, columns=columns).assign(start=lambda f: pd.to_datetime(f["start"]))


COHORT = pd.DataFrame({"stay_id": [1, 2, 3, 4, 5], "intime": pd.to_datetime(["2150-01-01 00:00"] * 5)})
BASE = [
    (1, 9, "2150-01-01 00:30", 70.0, "saline", False, 10.0),    # keep-vein-open infusion before time zero: adjusted, not time zero
    (1, 10, "2150-01-01 01:00", 70.0, "balanced", True, np.nan), # first fluid therapy (a bolus of any charted volume): time zero
    (1, 11, "2150-01-01 02:00", 70.0, "saline", False, 125.0),
    (2, 20, "2150-01-01 03:00", 80.0, "saline", False, 100.0),
    (2, 21, "2150-01-01 03:00", 80.0, "balanced", False, 5.0),   # other fluid starts at time zero, even at a low rate: excluded
    (3, 30, "2150-01-02 01:00", 90.0, "saline", True, np.nan),   # first fluid therapy outside the 24 h window
    (4, 40, "2149-12-31 23:00", 60.0, "balanced", True, np.nan), # first fluid therapy before ICU admission: not a new user
    (5, 50, "2150-01-01 00:10", 60.0, "saline", False, np.nan),  # infusion of unknown rate before time zero: excluded
    (5, 51, "2150-01-01 01:00", 60.0, "balanced", False, 80.0),
]


class TimeZeroTests(unittest.TestCase):
    def test_assignment_eligibility_and_prior_low_rate_infusions(self):
        frame, counts = time_zero(COHORT, starts(BASE))
        self.assertEqual(frame["stay_id"].tolist(), [1])
        self.assertEqual(frame["kind"].tolist(), ["balanced"])
        self.assertEqual(frame["hours_icu_to_t0"].tolist(), [1.0])
        self.assertEqual(frame[["prior_low_rate_balanced_infusion", "prior_low_rate_saline_infusion"]].values.tolist(), [[0, 1]])
        self.assertEqual(counts, {"stays_with_fluid_therapy": 5, "mixed_start_excluded": 1, "unknown_rate_excluded": 1,
                                  "first_start_outside_0_24h_excluded": 2})

    def test_information_after_time_zero_changes_nothing(self):
        later = BASE + [(1, 12, "2150-01-01 01:30", 99.0, "saline", True, np.nan), (1, 13, "2150-01-01 05:00", 99.0, "saline", False, np.nan)]
        pd.testing.assert_frame_equal(time_zero(COHORT, starts(BASE))[0], time_zero(COHORT, starts(later))[0])



class LabSchemaTests(unittest.TestCase):
    def test_items_without_selected_rows_keep_their_columns(self):
        rows = pd.DataFrame({"stay_id": [1], "name": ["chloride"], "value": [101.0], "time": pd.to_datetime(["2150-01-02"])})
        wide = with_provenance(rows, OUTCOME_LABS, pd.Series([1, 2]))
        self.assertEqual(sorted(wide.columns), ["bicarbonate", "bicarbonate_time", "chloride", "chloride_time"])
        self.assertEqual(wide.index.tolist(), [1, 2])
        empty = with_provenance(rows.iloc[:0], OUTCOME_LABS, pd.Series([1, 2]))
        self.assertEqual(sorted(empty.columns), sorted(wide.columns))
        self.assertTrue(empty.isna().all().all())


if __name__ == "__main__":
    unittest.main()
