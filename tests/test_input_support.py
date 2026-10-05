import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.input_support import learn_input_support, apply_input_support


class InputSupportTests(unittest.TestCase):
    def test_train_only_support_does_not_learn_from_future_labels(self):
        frame = pd.DataFrame({"a_t": [0., 1.], "a_age_hr": [0., 8.],
                              "b_t": [5., np.nan], "b_age_hr": [8., np.nan],
                              "b_tp6": [4., 5.]})
        contract = learn_input_support(frame, ["a", "b", "c"], max_age_hours=6)
        self.assertEqual(contract["supported"], ["a"])
        self.assertEqual(contract["source_observation_counts"]["a"], 1)

    def test_masks_entire_history_preserving_zero_labels_and_identity(self):
        frame = pd.DataFrame({"subject_id": ["p", "p"], "t": [1, 2],
                              "a_t": [0., 1.], "a_age_hr": [0., 0.],
                              "b_t": [9., 10.], "b_age_hr": [1., 0.], "b_tp6": [15., 16.]})
        contract = {"variables": ["a", "b"], "supported": ["a"], "unsupported": ["b"]}
        out = apply_input_support(frame, contract)
        self.assertTrue(out.b_t.isna().all())
        self.assertTrue(out.b_age_hr.isna().all())
        pd.testing.assert_frame_equal(out[["subject_id", "t", "a_t", "a_age_hr", "b_tp6"]],
                                      frame[["subject_id", "t", "a_t", "a_age_hr", "b_tp6"]])
        self.assertEqual(frame.b_t.iloc[0], 9.)

    def test_invalid_partition_fails_closed(self):
        with self.assertRaises(ValueError):
            apply_input_support(pd.DataFrame(), {"variables": ["a"], "supported": ["a"], "unsupported": ["a"]})

    def test_masking_source_unsupported_inputs_is_equivalent(self):
        from scripts.audit_release_masked_cells import input_frame
        source = pd.DataFrame({"subject_id": ["p"], "stay_id": ["s"], "t": pd.to_datetime(["2026-01-01"]),
                               "hours_since_onset": [0.], "_time_hours": [0.],
                               "a_t": [0.], "a_age_hr": [0.], "b_t": [10.], "b_age_hr": [8.]})
        contract = learn_input_support(source, ["a", "b"], max_age_hours=6)
        before = input_frame(source, ["a", "b"], 6)
        after = input_frame(apply_input_support(source, contract), ["a", "b"], 6)
        # The release neutralizes all encoder age/mask channels; its measured
        # values must stay identical after applying the age filter.
        pd.testing.assert_frame_equal(before[["a_t", "b_t"]], after[["a_t", "b_t"]])


if __name__ == "__main__":
    unittest.main()
