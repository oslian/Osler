import unittest

import numpy as np
import pandas as pd

from osler_jepa.organ_schema import (
    CANONICAL_ORGAN_VARIABLES,
    CANONICAL_SYSTEM_VARIABLES,
    assert_disjoint_contracts,
    canonical_organ_adjacency,
    canonical_organ_contracts,
    canonical_target_organ_membership,
)
from whole_body_joint_jepa import _multiscale_history_indices


class CanonicalOrganSchemaTests(unittest.TestCase):
    def test_contracts_are_disjoint_and_only_use_anatomical_organs(self):
        available = {
            "map",
            "creatinine",
            "potassium",
            "glucose",
            "troponin_i",
            "unknown_marker",
        }
        contracts = canonical_organ_contracts(available)
        assert_disjoint_contracts(contracts)

        flattened = [
            variable
            for variables in contracts.values()
            for variable in variables
        ]
        self.assertEqual(set(flattened), available - {"unknown_marker"})
        self.assertEqual(set(contracts), set(CANONICAL_ORGAN_VARIABLES))
        self.assertIn("stomach", contracts)
        self.assertIn("intestines", contracts)
        self.assertEqual(contracts["stomach"], ())
        self.assertEqual(contracts["intestines"], ())

    def test_system_and_organ_taxonomies_are_distinct(self):
        self.assertIn("acid_base_electrolyte", CANONICAL_SYSTEM_VARIABLES)
        self.assertNotIn("acid_base_electrolyte", CANONICAL_ORGAN_VARIABLES)
        self.assertIn("heart", CANONICAL_ORGAN_VARIABLES)
        self.assertIn("lungs", CANONICAL_ORGAN_VARIABLES)
        self.assertIn("kidneys", CANONICAL_ORGAN_VARIABLES)
        self.assertNotIn("heart", CANONICAL_SYSTEM_VARIABLES)

    def test_targets_map_from_functional_systems_to_anatomical_organs(self):
        organs = (
            "heart",
            "blood_vessels",
            "lungs",
            "kidneys",
            "liver",
            "pancreas",
            "thyroid",
            "adrenal_glands",
        )
        membership = canonical_target_organ_membership(
            ("bicarbonate", "glucose", "heart_rate"),
            organs,
        )
        self.assertEqual(membership.shape, (3, len(organs)))
        self.assertEqual(membership[0, organs.index("kidneys")], 1.0)
        self.assertEqual(membership[0, organs.index("lungs")], 1.0)
        for organ in ("pancreas", "liver", "thyroid", "adrenal_glands"):
            self.assertEqual(membership[1, organs.index(organ)], 1.0)
        for organ in ("heart", "blood_vessels", "kidneys"):
            self.assertEqual(membership[2, organs.index(organ)], 1.0)

    def test_duplicate_ownership_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "belongs to both"):
            assert_disjoint_contracts(
                {
                    "renal": ("creatinine",),
                    "cardiac": ("creatinine",),
                }
            )

    def test_canonical_organ_adjacency_is_sparse_symmetric_and_self_free(self):
        names = (
            "heart",
            "kidneys",
            "lungs",
            "blood_vessels",
            "skeletal_muscle",
        )
        adjacency = np.asarray(canonical_organ_adjacency(names))
        np.testing.assert_array_equal(adjacency, adjacency.T)
        np.testing.assert_array_equal(np.diag(adjacency), np.zeros(len(names)))
        self.assertEqual(adjacency[0, 1], 1.0)
        self.assertEqual(adjacency[0, 2], 1.0)
        self.assertEqual(adjacency[0, 3], 1.0)
        self.assertEqual(adjacency[-1, 1], 1.0)

    def test_multiscale_history_is_causal_and_spans_long_context(self):
        frame = pd.DataFrame(
            {
                "_source_horizon": [6.0] * 6,
                "stay_id": ["a"] * 6,
                "_time_hours": [0.0, 1.0, 2.0, 4.0, 8.0, 24.0],
            }
        )
        history = _multiscale_history_indices(
            frame,
            lags_hours=(24.0, 8.0, 2.0, 0.0),
        )
        selected_times = frame["_time_hours"].to_numpy()[history[-1]]
        np.testing.assert_array_equal(
            selected_times,
            np.asarray([0.0, 8.0, 8.0, 24.0]),
        )
        for row, selected in enumerate(history):
            self.assertTrue(
                np.all(
                    frame["_time_hours"].to_numpy()[selected]
                    <= frame.loc[row, "_time_hours"]
                )
            )


if __name__ == "__main__":
    unittest.main()
