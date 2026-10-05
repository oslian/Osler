import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.train_release_masked_abc import (
    batch_inputs, candidate_inputs, configure_training, masked_arrays, paired_training_rows,
)
from scripts.audit_masked_readout_ensemble import fit_readout, predict_readouts
from whole_body_jepa.runtime import load_release


class MaskedABC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.assets = load_release()

    def frame(self):
        data = {"subject_id": ["p", "p"], "stay_id": ["s", "s"],
                "hours_since_onset": [0., 1.], "_time_hours": [0., 1.]}
        for i, variable in enumerate(self.assets.checkpoint["variables"]):
            data[f"{variable}_t"] = [float(self.assets.scaler.medians[i])] * 2
            data[f"{variable}_age_hr"] = [0., 0.]
        return pd.DataFrame(data)

    def test_explicit_marker_distinguishes_missing_from_observed_median(self):
        frame = self.frame()
        idx = self.assets.checkpoint["variables"].index("glucose")
        frame.loc[0, "glucose_t"] = np.nan
        frame["urine_output_t"] = 0.
        candidate = candidate_inputs(frame, self.assets, 3, explicit=True)
        np.testing.assert_array_equal(candidate["current"][:, idx], 0.)
        np.testing.assert_array_equal(candidate["mask"][:, idx], [0., 1.])
        urine = self.assets.checkpoint["variables"].index("urine_output")
        np.testing.assert_array_equal(candidate["mask"][:, urine], 1.)
        np.testing.assert_array_equal(candidate["ages"], 0.)

    def test_hidden_group_is_removed_from_whole_history_and_both_paths_agree(self):
        frame = self.frame()
        variables = self.assets.checkpoint["variables"]
        hidden = [variables.index(v) for v in ("hemoglobin", "hematocrit")]
        for explicit in (False, True):
            arrays = candidate_inputs(frame, self.assets, 3, explicit=explicit)
            original = arrays["input"].copy()
            small = batch_inputs(arrays, np.asarray([1]), hidden, explicit=explicit)
            full = batch_inputs(masked_arrays(arrays, hidden, explicit), np.asarray([1]))
            for a, b in zip(small, full):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(small[0][..., hidden], torch.zeros_like(small[0][..., hidden]))
            np.testing.assert_array_equal(arrays["input"], original)
            if explicit:
                torch.testing.assert_close(small[2][:, hidden], torch.zeros_like(small[2][:, hidden]))

    def test_B_changes_only_reconstruction_and_forecast_is_exactly_unchanged(self):
        model = copy.deepcopy(self.assets.model).eval()
        arrays = candidate_inputs(self.frame(), self.assets, 3)
        inputs = batch_inputs(arrays, np.asarray([1]))
        horizon = torch.tensor([3.])
        with torch.no_grad():
            before = model(*inputs, horizon)
        parameters = configure_training(model, "B")
        optimizer = torch.optim.AdamW(parameters, lr=.01)
        state = model.encode_context_components(*inputs)[0]
        loss = (model.reconstruction_heads[0](state) - 2.).square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            after = model(*inputs, horizon)
        for key in ("value", "context_latent", "belief_state", "predicted_target_latent", "organ_latents"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(before["reconstruction"], after["reconstruction"]))

    def test_worldmodel_reconstructs_current_not_future_latent(self):
        model = self.assets.model.eval()
        arrays = candidate_inputs(self.frame(), self.assets, 3)
        inputs = batch_inputs(arrays, np.asarray([1]))
        with torch.no_grad():
            first = model(*inputs, torch.tensor([3.]))
            second = model(*inputs, torch.tensor([12.]))
            expected = torch.cat([h(first["context_latent"]) for h in model.reconstruction_heads], dim=-1)
        torch.testing.assert_close(first["reconstruction"], expected, rtol=0, atol=0)
        torch.testing.assert_close(first["reconstruction"], second["reconstruction"], rtol=0, atol=0)

    def test_reconstruction_semantics_fix_does_not_change_any_forecast(self):
        model = self.assets.model.eval()
        arrays = candidate_inputs(self.frame(), self.assets, 3)
        inputs = batch_inputs(arrays, np.asarray([1]))
        original = model._decode_latent

        def former_semantics(*args, **kwargs):
            kwargs.pop("reconstruction_latent", None)
            return original(*args, **kwargs)

        with torch.no_grad():
            updated = model(*inputs, torch.tensor([12.]))
            with patch.object(model, "_decode_latent", side_effect=former_semantics):
                previous = model(*inputs, torch.tensor([12.]))
        for key in updated:
            if key != "reconstruction":
                torch.testing.assert_close(previous[key], updated[key], rtol=0, atol=0)

    def test_C_does_not_train_future_or_measurement_process_heads(self):
        model = copy.deepcopy(self.assets.model)
        configure_training(model, "C")
        enabled = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any(n.startswith("belief_filter.") for n in enabled))
        self.assertTrue(any(n.startswith("state_encoder.") for n in enabled))
        self.assertFalse(any(n.startswith(("target_", "value_heads.", "measurement_time_heads.", "observation_heads.")) for n in enabled))

    def test_B_and_C_draw_identical_minibatches_even_without_B_retention_loss(self):
        first, second = np.random.default_rng(73), np.random.default_rng(73)
        pool, retention = np.arange(300), np.arange(500)
        for _ in range(5):
            b, _ = paired_training_rows(first, pool, retention, 128)
            c, _ = paired_training_rows(second, pool, retention, 128)
            np.testing.assert_array_equal(b, c)

    def test_ensemble_training_preserves_initial_head_and_is_seed_reproducible(self):
        initial = self.assets.model.reconstruction_heads[0]
        weights = {k: v.clone() for k, v in initial.state_dict().items()}
        state = torch.zeros(20, self.assets.checkpoint["latent"])
        label = torch.ones(20)
        a = fit_readout(initial, state, label, 7, steps=2, batch_size=8)
        b = fit_readout(initial, state, label, 7, steps=2, batch_size=8)
        for k, v in initial.state_dict().items():
            torch.testing.assert_close(v, weights[k], rtol=0, atol=0)
        predictions = predict_readouts([a, b], state, median=5., scale=2.)
        np.testing.assert_array_equal(predictions[0], predictions[1])
        self.assertEqual(predictions.shape, (2, 20))


if __name__ == "__main__":
    unittest.main()
