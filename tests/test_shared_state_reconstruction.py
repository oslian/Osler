import unittest

import numpy as np
import pandas as pd

from scripts.validate_low_rank_shared_state import _state_matrix
from scripts.validate_shared_state_reconstruction import (
    _combine_development_sources,
    _permuted_physiology,
    _source_balanced_rows,
)
from whole_body_jepa.nowcast import related_siblings
from whole_body_joint_jepa import _masked_batch_view


class SharedStateReconstructionTests(unittest.TestCase):
    def test_source_balancing_equalizes_database_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "database": ["a", "a", "a", "b"],
                "subject_id": ["1", "2", "3", "4"],
            }
        )
        rows = _source_balanced_rows(frame, np.arange(4), seed=7)
        counts = frame.iloc[rows]["database"].value_counts().to_dict()
        self.assertEqual(counts, {"a": 3, "b": 3})

    def test_combining_sources_offsets_history_indices(self) -> None:
        first = pd.DataFrame(
            {"subject_id": ["1", "1"], "_history_0": [0, 1]}
        )
        second = pd.DataFrame(
            {"subject_id": ["1", "1"], "_history_0": [0, 1]}
        )
        combined = _combine_development_sources([first, second])
        self.assertEqual(combined["_history_0"].tolist(), [0, 1, 2, 3])
        self.assertEqual(combined["subject_id"].nunique(), 2)

    def test_low_rank_state_contains_values_masks_ages_and_organ_presence(self) -> None:
        arrays = [None] * 10
        arrays[2] = np.asarray([[1.0, 2.0]], dtype=np.float32)
        arrays[3] = np.asarray([[1.0, 0.0]], dtype=np.float32)
        arrays[4] = np.asarray([[0.0, 7.0]], dtype=np.float32)
        arrays[9] = np.asarray([[1.0]], dtype=np.float32)
        state = _state_matrix(tuple(arrays))
        np.testing.assert_allclose(
            state,
            np.asarray([[1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0]]),
        )

    def test_urine_representations_are_masked_as_siblings(self) -> None:
        siblings = related_siblings("urine_output")
        self.assertIn("urine_output_per_kg_hour", siblings)
        self.assertIn("urine_output_3h_per_kg_hour", siblings)

    def test_structured_mask_hides_target_and_deterministic_sibling(self) -> None:
        variables = ("ph", "bicarbonate", "heart_rate")
        current = np.asarray([[0.2, -0.4, 0.8]], dtype=np.float32)
        mask = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
        ages = np.asarray([[0.0, 0.0, 7.0]], dtype=np.float32)
        presence = np.asarray([[1.0]], dtype=np.float32)
        input_matrix = np.zeros((1, 3 * len(variables) + 2), dtype=np.float32)
        input_matrix[0, : len(variables)] = current[0]
        input_matrix[0, len(variables) : 2 * len(variables)] = mask[0]
        input_matrix[0, 2 * len(variables) : 3 * len(variables)] = ages[0]
        input_matrix[0, 3 * len(variables) + 1] = 1.0

        view = _masked_batch_view(
            np.asarray([0]),
            current,
            mask,
            ages,
            presence,
            input_matrix,
            np.asarray([[0]]),
            variables,
            ("acid_base_electrolyte",),
            {"acid_base_electrolyte": variables},
            np.random.default_rng(7),
            masked_forecast_probability=0.0,
            module_mask_probability=0.0,
            structured_mask_probability=1.0,
            structured_mask_max_age_hours=1.0,
        )

        self.assertTrue(bool(view[-1][0, 0]))
        self.assertTrue(bool(view[-1][0, 1]))
        self.assertEqual(float(view[1][0, 0]), 0.0)
        self.assertEqual(float(view[1][0, 1]), 0.0)
        self.assertEqual(float(view[4][0, 0, 0]), 0.0)
        self.assertEqual(float(view[4][0, 0, 1]), 0.0)

    def test_structured_mask_does_not_choose_stale_truth(self) -> None:
        variables = ("ph", "bicarbonate")
        current = np.asarray([[0.2, -0.4]], dtype=np.float32)
        mask = np.ones((1, 2), dtype=np.float32)
        ages = np.full((1, 2), 5.0, dtype=np.float32)
        presence = np.ones((1, 1), dtype=np.float32)
        input_matrix = np.zeros((1, 3 * len(variables) + 2), dtype=np.float32)
        view = _masked_batch_view(
            np.asarray([0]),
            current,
            mask,
            ages,
            presence,
            input_matrix,
            np.asarray([[0]]),
            variables,
            ("acid_base_electrolyte",),
            {"acid_base_electrolyte": variables},
            np.random.default_rng(7),
            masked_forecast_probability=0.0,
            module_mask_probability=0.0,
            structured_mask_probability=1.0,
            structured_mask_max_age_hours=1.0,
        )
        self.assertFalse(bool(view[-1].any()))

    def test_placebo_preserves_marginals_and_missingness(self) -> None:
        frame = pd.DataFrame(
            {
                "sodium_t": [1.0, np.nan, 3.0, 5.0],
                "potassium_t": [2.0, 4.0, np.nan, 8.0],
            }
        )
        placebo = _permuted_physiology(
            frame, np.arange(len(frame)), ("sodium", "potassium"), seed=17
        )
        for column in frame.columns:
            original = frame[column].to_numpy()
            shuffled = placebo[column].to_numpy()
            np.testing.assert_array_equal(np.isnan(shuffled), np.isnan(original))
            np.testing.assert_allclose(
                np.sort(shuffled[np.isfinite(shuffled)]),
                np.sort(original[np.isfinite(original)]),
            )


if __name__ == "__main__":
    unittest.main()
