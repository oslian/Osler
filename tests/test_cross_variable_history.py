import unittest

import numpy as np
import pandas as pd

from whole_body_jepa.cross_variable_history import (
    cross_variable_history_features,
    cross_variable_history_views,
    matched_wrong_patient_rows,
)
from scripts.screen_cross_variable_history_catalog import target_role_masks


class CrossVariableHistoryTests(unittest.TestCase):
    def test_target_roles_are_patient_disjoint_and_support_balanced(self):
        rows = []
        for hospital in range(10):
            for patient in range(10):
                rows.append(
                    {
                        "hospitalid": hospital,
                        "subject_id": f"{hospital}:{patient}",
                    }
                )
        frame = pd.DataFrame(rows)
        eligible = np.ones(len(frame), dtype=bool)
        roles, details = target_role_masks(frame, eligible, "map@3h")
        patient_sets = {
            role: set(frame.loc[mask, "subject_id"]) for role, mask in roles.items()
        }
        self.assertFalse(patient_sets["fit"] & patient_sets["patient_test"])
        self.assertFalse(patient_sets["fit"] & patient_sets["hospital_test"])
        self.assertFalse(patient_sets["patient_test"] & patient_sets["hospital_test"])
        self.assertGreater(details["heldout_eligible_patients"], 0)

    def test_wrong_patient_donor_is_time_matched(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["a", "a", "b", "b"],
                "_time_hours": [1.0, 9.0, 2.0, 10.0],
            }
        )
        donors = matched_wrong_patient_rows(frame, seed=7)
        self.assertTrue(
            np.all(
                frame.subject_id.to_numpy()[donors]
                != frame.subject_id.to_numpy()
            )
        )
        np.testing.assert_array_equal(donors, [2, 3, 0, 1])

    def test_placebos_preserve_the_information_they_are_not_testing(self):
        own = np.asarray([[1.0], [2.0]], dtype=np.float32)
        current = np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32)
        full = np.asarray(
            [[10.0, 11.0, 100.0, 101.0], [20.0, 21.0, 200.0, 201.0]],
            dtype=np.float32,
        )
        views = cross_variable_history_views(own, current, full, np.asarray([1, 0]))

        np.testing.assert_allclose(views["current_cross"][0], [1.0, 10.0, 11.0])
        np.testing.assert_allclose(
            views["current_cross_placebo"][0], [1.0, 20.0, 21.0]
        )
        np.testing.assert_allclose(
            views["cross_history_placebo"][0], [1.0, 10.0, 11.0, 200.0, 201.0]
        )

    def test_rejects_invalid_view_shapes_and_donors(self):
        own = np.ones((2, 1), dtype=np.float32)
        current = np.ones((2, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "narrower"):
            cross_variable_history_views(own, current, np.ones((2, 1)), np.asarray([1, 0]))
        with self.assertRaisesRegex(ValueError, "outside"):
            cross_variable_history_views(own, current, current, np.asarray([2, 0]))

    def test_excludes_target_and_uses_only_distinct_causal_measurements(self):
        frame = pd.DataFrame(
            {
                "stay_id": ["a", "a", "a", "b", "b", "b"],
                "_time_hours": [0.0, 3.0, 6.0, 0.0, 3.0, 6.0],
            }
        )
        variables = ("glucose", "map", "sodium")
        normalized = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [0.5, 1.0, 3.0],
                [1.0, 2.0, 3.0],
                [9.0, 8.0, 7.0],
                [9.5, 8.5, 7.5],
                [10.0, 9.0, 8.0],
            ],
            dtype=np.float32,
        )
        observed = np.ones_like(normalized)
        # Sodium at a/6h is still the a/3h measurement; it must not create a delta.
        ages = np.zeros_like(normalized)
        ages[2, 2] = 3.0 / 24.0
        history = np.asarray(
            [
                [0, 0], [0, 1], [1, 2],
                [3, 3], [3, 4], [4, 5],
            ],
            dtype=np.int64,
        )
        current, temporal = cross_variable_history_features(
            frame,
            np.asarray([2]),
            variables=variables,
            target="glucose",
            normalized=normalized,
            observed=observed,
            ages=ages,
            history_indices=history,
            history_lags_hours=(3.0, 0.0),
            selected_lags_hours=(3.0,),
        )
        self.assertEqual(current.shape, (1, 6))
        self.assertEqual(temporal.shape, (1, 10))
        # Non-target order is MAP, sodium. MAP changed; sodium is the same measurement.
        np.testing.assert_allclose(temporal[0, 6:8], [1.0, 0.0])
        np.testing.assert_allclose(temporal[0, 8:10], [1.0, 0.0])

    def test_rejects_cross_stay_history(self):
        frame = pd.DataFrame(
            {"stay_id": ["a", "b"], "_time_hours": [0.0, 3.0]}
        )
        values = np.ones((2, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "crosses patient stays"):
            cross_variable_history_features(
                frame,
                np.asarray([1]),
                variables=("glucose", "map"),
                target="glucose",
                normalized=values,
                observed=values,
                ages=np.zeros_like(values),
                history_indices=np.asarray([[0], [0]]),
                history_lags_hours=(1.0,),
                selected_lags_hours=(1.0,),
            )


if __name__ == "__main__":
    unittest.main()
