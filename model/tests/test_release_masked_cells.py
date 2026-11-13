import json
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.audit_release_masked_cells import (
    describe_inputs, donor_indices, infer, input_frame, mask_inputs, score, select_rows, valid_future_rows,
)
from whole_body_jepa.runtime import _build_inputs, load_release


class ReleaseMaskedCellsTests(unittest.TestCase):
    def test_mask_removes_every_history_copy_and_preserves_original(self):
        arrays = {"current": np.arange(12, dtype=float).reshape(4, 3),
                  "input": np.arange(40, dtype=float).reshape(4, 10),
                  "history": np.asarray([[0, 1], [0, 2], [1, 2], [2, 3]])}
        masked = mask_inputs(arrays, [0, 2])
        np.testing.assert_array_equal(masked["input"][:, [0, 2]], 0)
        np.testing.assert_array_equal(masked["current"][:, [0, 2]], 0)
        np.testing.assert_array_equal(masked["input"][:, 1], arrays["input"][:, 1])
        self.assertEqual(arrays["current"][3, 2], 11)

    def test_model_input_whitelist_excludes_labels_and_keeps_observed_zero(self):
        frame = pd.DataFrame({"subject_id": ["p", "p"], "stay_id": ["s", "s"],
                              "hours_since_onset": [0, 1], "_time_hours": [0, 1],
                              "urine_output_t": [0., 70.], "urine_output_age_hr": [0., 7.],
                              "future_urine_output": [999., 999.], "careunit": ["x", "y"],
                              "hospitalid": [100, 200]})
        view = input_frame(frame, ["urine_output"], 6)
        self.assertEqual(view["urine_output_t"].iloc[0], 0)
        self.assertTrue(pd.isna(view["urine_output_t"].iloc[1]))
        self.assertNotIn("future_urine_output", view)
        self.assertNotIn("careunit", view)
        self.assertNotIn("hospitalid", view)

    def test_patient_anchor_sampling_is_outcome_independent(self):
        frame = pd.DataFrame({"subject_id": ["a", "a", "b", "b"], "stay_id": ["a", "a", "b", "b"],
                              "t": pd.date_range("2020-01-01", periods=4, freq="h"), "truth": [1, 2, 3, 4]})
        first = select_rows(frame, [True] * 4, 4, "test")
        frame["truth"] *= 999
        np.testing.assert_array_equal(first, select_rows(frame, [True] * 4, 4, "test"))
        self.assertEqual(frame.iloc[first]["subject_id"].nunique(), len(first))

    def test_wrong_patient_control_never_uses_same_subject(self):
        subjects = np.asarray(["a", "b", "c", "d"])
        donors = donor_indices(subjects, 137)
        self.assertTrue(np.all(subjects != subjects[donors]))
        np.testing.assert_array_equal(np.sort(donors), np.arange(4))
        with self.assertRaises(ValueError):
            donor_indices(["a", "a"], 137)

    def test_low_support_cannot_pass(self):
        result = score(pd.DataFrame(index=range(10)), np.zeros(10), np.zeros(10), 0, "test")
        self.assertFalse(result["diagnostic_pass"])

    def test_native_strict_labels_are_accepted_but_near_and_past_are_not(self):
        frame = pd.DataFrame({"t": pd.to_datetime(["2020-01-01"] * 4),
                              "glucose_t": [0., 1., 2., 3.], "future_glucose": [1., 2., 3., 4.],
                              "future_label_quality_glucose": ["strict", "exact", "near", "strict"],
                              "future_event_time_glucose": ["2020-01-02"] * 3 + ["2019-01-01"]})
        np.testing.assert_array_equal(valid_future_rows(frame, "glucose"), [True, True, False, False])

    def test_support_report_serializes_numpy_counts(self):
        frame = pd.DataFrame({"glucose_t": [0., np.nan]})
        info = describe_inputs(frame, np.asarray([0]), ["glucose"], "glucose")
        self.assertEqual(json.loads(json.dumps(info))["current_input_variables_available"], 1)

    def test_real_heads_do_not_receive_hidden_anchor_or_sibling(self):
        torch.set_num_threads(2)
        assets = load_release()
        variables = assets.checkpoint["variables"]
        columns = {"subject_id": ["a", "a"], "stay_id": ["a", "a"],
                   "hours_since_onset": [0., 1.], "_time_hours": [0., 1.]}
        for i, v in enumerate(variables):
            columns[f"{v}_t"] = float(assets.scaler.medians[i])
            columns[f"{v}_age_hr"] = 0.
        frame = pd.DataFrame(columns)
        hidden = [variables.index(v) for v in ("glucose", "bun", "sodium", "serum_osmolality")]
        first = mask_inputs(_build_inputs(frame, assets, 3), hidden)
        for v in ("glucose", "bun", "sodium", "serum_osmolality"):
            frame[f"{v}_t"] = [999., 888.]
        second = mask_inputs(_build_inputs(frame, assets, 3), hidden)
        for k in ("input", "current", "history", "mask", "ages", "presence"):
            np.testing.assert_array_equal(first[k], second[k])
        a = infer(assets, first, np.asarray([1]), "glucose", "glucose@3h")
        b = infer(assets, second, np.asarray([1]), "glucose", "glucose@3h")
        self.assertEqual(len(a), 3)
        for source in a:
            np.testing.assert_array_equal(a[source], b[source])


if __name__ == "__main__":
    unittest.main()
