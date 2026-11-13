import unittest

import numpy as np
import pandas as pd

from scripts.audit_prospective_change import role_masks, shuffled_donors, valid_rows
from whole_body_jepa.prospective_change import (
    alarm_cutoff, cqr_bounds, fit_risk_cqr, raw_prediction, recall_at_fpr,
    trend_features,
)


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class ConstantRisk:
    def predict_proba(self, x):
        return np.tile([.4, .6], (len(x), 1))


class ProspectiveChangeTests(unittest.TestCase):
    def test_history_is_asof_stay_specific_and_deduplicated(self):
        f = pd.DataFrame({"stay_id": [1, 1, 1, 1, 2],
            "t": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00",
                                 "2020-01-01 02:00", "2020-01-01 03:00", "2020-01-01 01:00"]),
            "x_t": [10., 10., 14., 999., 888.], "x_age_hr": [0., 1., 0., 0., 0.]})
        expected = np.array([[14., 2., 2., 0., 2., 2., 4.]])
        np.testing.assert_allclose(trend_features(f, [2], "x"), expected)
        f["future_x"] = -9999
        f.loc[3, "x_t"] = -1000
        np.testing.assert_allclose(trend_features(f, [2], "x"), expected)

    def test_patient_roles_are_disjoint_even_across_hospitals(self):
        f = pd.DataFrame({"subject_id": [str(i // 2) for i in range(400)],
                          "hospitalid": np.arange(400) % 20})
        masks = role_masks(f)
        groups = {r: set(f.loc[m, "subject_id"]) for r, m in masks.items()}
        self.assertTrue(groups["hospital_test"])
        for a in groups:
            for b in groups:
                if a != b:
                    self.assertFalse(groups[a] & groups[b])

    def test_placebo_never_uses_same_patient(self):
        patients = np.array(["a", "a", "b", "c", "c"])
        donors = shuffled_donors(patients)
        self.assertTrue(np.all(patients != patients[donors]))
        self.assertEqual(donors[0], donors[1])
        with self.assertRaises(ValueError):
            shuffled_donors(["a", "a"])

    def test_strict_alarm_cutoff_handles_ties(self):
        score = np.r_[np.zeros(95), np.ones(5)]
        cutoff = alarm_cutoff(score, np.zeros(100, dtype=bool))
        self.assertLessEqual(np.mean(score > cutoff), .05)
        with self.assertRaises(ValueError):
            alarm_cutoff([0.] * 20, [False] * 20)

    def test_ranking_diagnostic_and_one_class(self):
        self.assertEqual(recall_at_fpr([0, 0, 1, 1], [0, .1, .8, .9], list("abcd")), 1.)
        self.assertIsNone(recall_at_fpr([0, 0], [.2, .3], ["a", "b"]))

    def test_cqr_unsupported_bin_abstains(self):
        q = np.tile([-1., 0., 1.], (120, 1))
        high = np.arange(120) >= 110
        radii = fit_risk_cqr(np.full(120, 3.), q, high)
        self.assertEqual(radii["0"], 2.)
        self.assertIsNone(radii["1"])
        lo, hi = cqr_bounds(q, high, radii)
        np.testing.assert_allclose(lo[:110], -3.)
        np.testing.assert_allclose(hi[:110], 3.)
        self.assertTrue(np.isnan(lo[110:]).all())

    def test_delta_readout_can_reverse_and_amplify(self):
        head = {"risk": ConstantRisk(), "quantiles": {
            .05: ConstantRegressor(-20), .5: ConstantRegressor(-5), .95: ConstantRegressor(40)}}
        out = raw_prediction(head, np.zeros((2, 3)))
        np.testing.assert_array_equal(out["delta_quantiles"], [[-20, -5, 40]] * 2)

    def test_target_specific_future_tolerance_and_freshness(self):
        f = pd.DataFrame({"t": pd.to_datetime(["2020-01-01"] * 4)})
        for target in ("map", "glucose"):
            f[f"{target}_t"] = 0.
            f[f"{target}_age_hr"] = [0., 0., 2., 0.]
            f[f"future_{target}"] = 1.
            f[f"future_delta_t_hr_{target}"] = [3., 3.75, 3., 3.]
            f[f"future_event_time_{target}"] = ["2020-01-01 03:00", "2020-01-01 03:45", "2020-01-01 03:00", "NaT"]
            f[f"future_label_quality_{target}"] = "strict"
        np.testing.assert_array_equal(valid_rows(f, "map"), [True, False, False, False])
        np.testing.assert_array_equal(valid_rows(f, "glucose"), [True, True, False, False])


if __name__ == "__main__":
    unittest.main()
