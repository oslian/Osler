import unittest

import numpy as np
import pandas as pd

from scripts.audit_anchor_aligned_signal import (
    COMPARISONS, METHODS, external_interpretation, feature_views, window_mask,
)
from scripts.audit_release_masked_cells import input_frame
from scripts.screen_cross_variable_history_catalog import target_role_masks


class AnchorAlignedSignalTests(unittest.TestCase):
    def test_placebos_replace_exactly_the_declared_blocks(self):
        own = np.arange(6).reshape(3, 2)
        current = np.arange(6, 12).reshape(3, 2)
        temporal = np.arange(12, 18).reshape(3, 2)
        full = np.column_stack([current, temporal])
        bundle = np.arange(21).reshape(3, 7)
        contract = {"feature_width": 7, "blocks": [
            {"name": n, "start": i, "stop": i + 1}
            for i, n in enumerate(("shared_context", "patient_belief", "target_future", "target_organ"))]}
        donors = np.array([1, 2, 0])
        views = feature_views(own, current, full, bundle, donors, contract)
        self.assertEqual(set(views), set(METHODS))
        for value in views.values():
            np.testing.assert_array_equal(value[:, :2], own)
        np.testing.assert_array_equal(views["temporal_placebo"][:, 2:4], current)
        np.testing.assert_array_equal(views["temporal_placebo"][:, 4:], temporal[donors])
        np.testing.assert_array_equal(views["full_cross_placebo"][:, 2:], full[donors])
        np.testing.assert_array_equal(views["current_cross_placebo"][:, 2:], current[donors])
        np.testing.assert_array_equal(views["latent_only"][:, 2:], bundle[:, :4])
        np.testing.assert_array_equal(views["latent_only_placebo"][:, 2:], bundle[donors, :4])
        np.testing.assert_array_equal(views["release_features"][:, 2:], bundle)
        with self.assertRaises(ValueError):
            feature_views(own, current, full, bundle[:, :-1], donors, contract)

    def test_window_rejects_missing_future_or_inconsistent_delta(self):
        delta = np.array([2, 5, 6, 7, 10, 6, 6, np.inf])
        frame = pd.DataFrame({"t": pd.Timestamp("2020-01-01", tz="UTC"),
            "future_delta_t_hr_creatinine": delta,
            "future_event_time_creatinine": pd.to_datetime([
                "2020-01-01T02:00Z", "2020-01-01T05:00Z", "2020-01-01T06:00Z",
                "2020-01-01T07:00Z", "2020-01-01T10:00Z", None,
                "2020-01-01T09:00Z", "2020-01-01T06:00Z"])})
        np.testing.assert_array_equal(window_mask(frame, (5, 7)), [False, True, True, True, False, False, False, False])
        np.testing.assert_array_equal(window_mask(frame, (2, 10)), [True] * 5 + [False] * 3)

    def test_future_metadata_never_enters_encoder_frame(self):
        frame = pd.DataFrame({"subject_id": ["a"], "stay_id": ["b"],
            "hours_since_onset": [6], "_time_hours": [6], "creatinine_t": [0.0],
            "creatinine_age_hr": [0.0], "future_creatinine": [999],
            "future_delta_t_hr_creatinine": [6], "hospitalid": [123]})
        x = input_frame(frame, ["creatinine"], 6)
        self.assertFalse(any(c.startswith("future_") or c == "hospitalid" for c in x))
        self.assertEqual(x.creatinine_t.iloc[0], 0)

    def test_failed_readout_is_not_proof_of_information_absence(self):
        result = {"methods": {name: {"overall_pass": False,
            "all_supported_careunits_pass": False} for name in COMPARISONS}}
        verdict = external_interpretation(result)
        self.assertFalse(any(verdict["method_diagnostic_pass"].values()))
        self.assertFalse(verdict["promotion_allowed"])
        self.assertIn("not_information_absence", verdict["conclusion"])

    def test_roles_are_patient_disjoint_even_across_hospitals(self):
        frame = pd.DataFrame({"subject_id": [str(i // 2) for i in range(120)],
            "hospitalid": [str(i % 12) for i in range(120)]})
        roles, _ = target_role_masks(frame, np.ones(len(frame), bool), "creatinine@6h")
        groups = [set(frame.loc[m, "subject_id"]) for m in roles.values()]
        for i, left in enumerate(groups):
            for right in groups[i+1:]:
                self.assertFalse(left & right)


if __name__ == "__main__":
    unittest.main()
