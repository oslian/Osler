import unittest

import numpy as np
import pandas as pd

from measurement_process_shift_diagnostic import (
    _greedy_spacing_keep,
    _reweighted_coverage,
    lab_labels_in_window,
    paired_coverage_shift,
    thin_lab_input,
)


def _stay_frame():
    anchors = pd.date_range("2100-01-01 06:00", periods=4, freq="6h")
    return pd.DataFrame({
        "subject_id": ["MIMIC:1"] * 4,
        "stay_id": ["MIMIC:10"] * 4,
        "anchor_time": anchors,
        # Measured 1h before every anchor: events at 05:00, 11:00, 17:00, 23:00.
        "creatinine_t": [1.0, 2.0, 3.0, 4.0],
        "creatinine_age_hr": [1.0, 1.0, 1.0, 1.0],
    })


def _raw_events():
    times = pd.date_range("2100-01-01 05:00", periods=7, freq="3h")
    return pd.DataFrame({
        "stay_id": [10] * 7,
        "charttime": times,
        "var": ["creatinine"] * 7,
        # 05:00=1.0, 08:00=1.5, 11:00=2.0, 14:00=3.5, 17:00=3.0, 20:00=3.8, 23:00=4.0
        "valuenum": [1.0, 1.5, 2.0, 3.5, 3.0, 3.8, 4.0],
    })


class MeasurementProcessDiagnosticTests(unittest.TestCase):
    def test_greedy_spacing_restarts_per_stay(self):
        keep = _greedy_spacing_keep(
            np.array(["a", "a", "a", "b"]), np.array([0.0, 1.0, 5.0, 0.5]), 4.0
        )
        self.assertEqual(keep.tolist(), [True, False, True, True])

    def test_raw_rebuild_without_thinning_reproduces_asof_input(self):
        frame = _stay_frame()
        rebuilt = thin_lab_input(frame, np.arange(4), _raw_events(), "creatinine", 0.0)
        np.testing.assert_allclose(rebuilt["creatinine_t"], frame["creatinine_t"])
        np.testing.assert_allclose(rebuilt["creatinine_age_hr"], frame["creatinine_age_hr"])

    def test_thinning_carries_older_value_and_never_invents_one(self):
        frame = _stay_frame()
        thinned = thin_lab_input(frame, np.arange(4), _raw_events(), "creatinine", 8.0)
        # Raw events at 05:00, 08:00, 11:00, 14:00, 17:00, 20:00, 23:00.
        # 8h spacing keeps 05:00, 14:00 and 23:00; the 12:00 anchor's latest
        # kept event (05:00) is 7h old, beyond the 6h look-back.
        values = thinned["creatinine_t"].to_numpy()
        ages = thinned["creatinine_age_hr"].to_numpy()
        np.testing.assert_allclose(values, [1.0, np.nan, 3.5, 4.0])
        np.testing.assert_allclose(ages, [1.0, np.nan, 4.0, 1.0])

    def test_thinning_respects_lookback(self):
        frame = _stay_frame()
        thinned = thin_lab_input(frame, np.arange(4), _raw_events(), "creatinine", 12.0)
        # Kept: 05:00 and 17:00. The 12:00 anchor's latest kept event is 7h old.
        self.assertTrue(np.isnan(thinned["creatinine_t"].iloc[1]))
        self.assertEqual(thinned["creatinine_t"].iloc[2], 3.0)

    def test_thinning_is_independent_of_datetime_resolution(self):
        frame = _stay_frame()
        micro = frame.assign(anchor_time=frame["anchor_time"].astype("datetime64[us]"))
        events = _raw_events()
        events = events.assign(charttime=events["charttime"].astype("datetime64[us]"))
        expected = thin_lab_input(frame, np.arange(4), _raw_events(), "creatinine", 8.0)
        observed = thin_lab_input(micro, np.arange(4), events, "creatinine", 8.0)
        np.testing.assert_allclose(
            observed[["creatinine_t", "creatinine_age_hr"]].to_numpy(),
            expected[["creatinine_t", "creatinine_age_hr"]].to_numpy(),
        )

    def test_lab_label_window_picks_nearest_to_horizon_inside_window(self):
        frame = _stay_frame().iloc[:1]
        events = pd.DataFrame({
            "stay_id": [10, 10, 10],
            "charttime": pd.to_datetime(["2100-01-01 09:30", "2100-01-01 13:00", "2100-01-01 15:00"]),
            "var": ["creatinine"] * 3,
            "valuenum": [5.0, 6.0, 7.0],
        })
        values, offsets = lab_labels_in_window(frame, np.array([0]), events, "creatinine", 4.5, 7.5)
        self.assertEqual(values.tolist(), [6.0])
        self.assertEqual(offsets.tolist(), [7.0])
        values, offsets = lab_labels_in_window(frame, np.array([0]), events, "creatinine", 2.5, 4.5)
        self.assertEqual(values.tolist(), [5.0])
        self.assertEqual(offsets.tolist(), [3.5])

    def test_reweighting_follows_hospital_bin_mix(self):
        cal_bins = np.array([0] * 40 + [1] * 40)
        cal_covered = np.array([1.0] * 40 + [0.5, 1.0] * 20)
        self.assertAlmostEqual(_reweighted_coverage(cal_bins, cal_covered, np.array([1, 1, 1, 1])), 0.75)
        self.assertAlmostEqual(_reweighted_coverage(cal_bins, cal_covered, np.array([0, 1])), 0.875)

    def test_paired_shift_is_patient_equal(self):
        subjects = np.array(["a", "a", "a", "b"])
        result = paired_coverage_shift(
            subjects, np.array([1, 1, 1, 1]), np.array([0, 0, 0, 1]), seed=1
        )
        self.assertAlmostEqual(result["coverage_shift"], -0.5)
        self.assertEqual(result["patients"], 2)


if __name__ == "__main__":
    unittest.main()
