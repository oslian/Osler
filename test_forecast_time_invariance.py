import unittest

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import normalize_cohort
from target_specific_domain_conformal_audit import _feature_frame


def stream(future):
    times = pd.to_datetime(["2100-01-01 06:00", "2100-01-01 12:00", "2100-01-01 18:00"])
    return pd.DataFrame({"subject_id": [1, 1, 1], "stay_id": [9, 9, 9], "t": times, "first_careunit": "MICU",
                         "potassium_t": [3.0, 4.0, 5.0], "potassium_age_hr": [0.5, 0.5, 0.5],
                         "potassium_tp6": future, "potassium_tp6_offset_hr": [0.5, np.nan, -1.0]})


class ForecastTimeInvarianceTests(unittest.TestCase):
    def test_histories_ignore_which_rows_have_future_labels(self):
        cohort, _ = normalize_cohort(stream([3.5, np.nan, 5.5]), database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        self.assertEqual(len(cohort), 2)                                        # the unlabelled middle anchor is not scored
        last = cohort[cohort["potassium_t"] == 5.0].iloc[0]
        self.assertEqual(last["lag1__potassium_t"], 4.0)                         # ...but it is still the previous anchor
        self.assertAlmostEqual(last["slope1h__potassium_t"], 1.0 / 6.0)
        self.assertEqual(last["hours_since_onset"], 12.0)

    def test_features_identical_whether_or_not_other_labels_exist(self):
        labelled, _ = normalize_cohort(stream([3.5, 4.5, 5.5]), database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        partial, _ = normalize_cohort(stream([3.5, np.nan, 5.5]), database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        a = _feature_frame(labelled)[labelled["potassium_t"] != 4.0].reset_index(drop=True)
        b = _feature_frame(partial).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[b.columns], b)

    def test_label_availability_includes_measurement_offset(self):
        cohort, _ = normalize_cohort(stream([3.5, np.nan, 5.5]), database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        np.testing.assert_allclose(cohort.sort_values("anchor_time")["future_potassium_available_hr"], [6.5, 7.0])


if __name__ == "__main__":
    unittest.main()
