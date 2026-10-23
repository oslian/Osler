import unittest

import numpy as np
import pandas as pd
import torch

from whole_body_joint_jepa import (
    JointWholeBodyJEPA,
    JointWholeBodyJEPAWorldModel,
    _joint_arrays,
    _multiscale_history_indices,
    _precomputed_or_recent_history_indices,
    _target_balanced_scale_loss,
)


def _frame():
    frame = pd.DataFrame(
        {
            "subject_id": ["s1"] * 6,
            "stay_id": ["stay1"] * 6,
            "t": pd.date_range("2025-01-01", periods=6, freq="h"),
            "_time_hours": [0.0, 1.0, 2.0, 4.0, 8.0, 24.0],
            "_source_horizon": [6.0] * 6,
            "_module_present_0": [1.0] * 6,
            "hours_since_onset": [0.0, 1.0, 2.0, 4.0, 8.0, 24.0],
            "x_t": [1.0, 1.1, 1.2, 1.4, 1.8, 2.4],
            "x_age_hr": [0.0] * 6,
            "future_x": [1.1, 1.2, 1.4, 1.8, 2.4, 3.0],
        }
    )
    history = _multiscale_history_indices(
        frame,
        lags_hours=(24.0, 8.0, 2.0, 0.0),
    )
    for index in range(history.shape[1]):
        frame[f"_history_{index}"] = history[:, index]
    return frame, history


class JointHistoryPlumbingTests(unittest.TestCase):
    def test_joint_arrays_uses_precomputed_multiscale_history(self):
        frame, expected = _frame()
        arrays = _joint_arrays(
            frame,
            ("x",),
            ("body",),
            np.arange(len(frame)),
            history=2,
        )
        np.testing.assert_array_equal(arrays[7], expected)
        self.assertEqual(arrays[7].shape[1], 4)

    def test_precomputed_history_rejects_future_index(self):
        frame, _ = _frame()
        frame.loc[0, "_history_0"] = 1
        with self.assertRaisesRegex(ValueError, "future observations"):
            _precomputed_or_recent_history_indices(frame)

    def test_precomputed_history_rejects_cross_stay_index(self):
        frame, _ = _frame()
        frame.loc[5, "stay_id"] = "stay2"
        frame.loc[5, "_history_0"] = 0
        with self.assertRaisesRegex(ValueError, "crosses patient stays"):
            _precomputed_or_recent_history_indices(frame)

    def test_target_specific_scale_is_positive_and_does_not_move_point_head(self):
        model = JointWholeBodyJEPA(
            n_state=2,
            input_dim=8,
            module_count=1,
            hidden=12,
            latent=8,
            token_dim=8,
        )
        output = model(
            torch.randn(4, 3, 8),
            torch.randn(4, 2),
            torch.ones(4, 2),
            torch.zeros(4, 2),
            torch.ones(4, 1),
            torch.full((4,), 6.0),
        )
        self.assertEqual(tuple(output["value_scale"].shape), (4, 2))
        self.assertTrue(bool((output["value_scale"] > 0.0).all()))

        model.zero_grad(set_to_none=True)
        loss = _target_balanced_scale_loss(
            output["value"],
            torch.zeros_like(output["value"]),
            output["value_scale"],
            torch.ones_like(output["value"]),
        )
        loss.backward()
        self.assertIsNone(model.value_heads[0][-1].weight.grad)
        self.assertIsNotNone(model.value_scale_heads[0][-1].weight.grad)

    def test_ema_target_encoder_remains_deterministic_in_training_mode(self):
        model = JointWholeBodyJEPAWorldModel(
            n_state=2,
            input_dim=8,
            module_count=1,
            hidden=12,
            latent=8,
            token_dim=8,
            target_encoder_mode="full",
        )
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.variable_mixer.training)
        self.assertFalse(model.target_variable_mixer.training)
        self.assertFalse(model.target_module_mixer.training)


if __name__ == "__main__":
    unittest.main()
