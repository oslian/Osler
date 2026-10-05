import copy
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.audit_masked_generalization import replay_loss
from scripts.masked_generalization_methods import (
    cached_context, dro_loss, fit_curve, group_weights, raw_view, selection_score, source_roles,
)
from scripts.train_release_masked_abc import candidate_inputs
from whole_body_jepa.multisource_intervals import stable_hash
from whole_body_jepa.runtime import load_release


class GeneralizationMethodsTests(unittest.TestCase):
    def test_hospital_roles_are_patient_disjoint_and_order_independent(self):
        frame = pd.DataFrame({"subject_id": [f"p{i}" for i in range(100)],
                              "hospitalid": [str(i // 5) for i in range(100)]})
        roles = source_roles(frame)
        np.testing.assert_array_equal(sum(v.astype(int) for v in roles.values()), 1)
        train_hospitals = set(frame.loc[roles["fit"], "hospitalid"])
        self.assertFalse(train_hospitals & set(frame.loc[roles["hospital_test"], "hospitalid"]))
        other = frame.sample(frac=1, random_state=19)
        for name, values in source_roles(other).items():
            self.assertEqual(set(other.loc[values, "subject_id"]), set(frame.loc[roles[name], "subject_id"]))

    def test_hospital_crossing_patient_fails_closed(self):
        held = next(str(i) for i in range(100) if stable_hash("masked_generalization_hospital", str(i)) % 5 == 0)
        fit = next(str(i) for i in range(100) if stable_hash("masked_generalization_hospital", str(i)) % 5 != 0)
        frame = pd.DataFrame({"subject_id": ["p", "p"], "hospitalid": [held, fit]})
        with self.assertRaises(ValueError):
            source_roles(frame)

    def test_group_weights_preserve_rare_patients_and_increase_small_supported_unit(self):
        units = np.asarray(["large"] * 100 + ["small"] * 20 + ["rare"] * 2)
        weights, codes, names = group_weights(units)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertTrue((weights > 0).all())
        self.assertGreater(weights[100], weights[0])
        self.assertNotIn("rare", names)
        self.assertEqual(codes[-1], -1)

    def test_dro_upweights_larger_training_error_and_preserves_gradient(self):
        errors = torch.tensor([1., 1., 3., 3.], requires_grad=True)
        weights = torch.zeros(2)
        loss = dro_loss(errors, torch.tensor([0, 0, 1, 1]), weights)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.exp().sum()), 1., places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(errors.grad).all())

    def test_selection_uses_worst_supported_unit(self):
        y = np.zeros(100)
        pred = np.r_[np.ones(60), np.full(40, 3.)]
        units = np.asarray(["a"] * 60 + ["b"] * 40)
        result = selection_score(y, pred, units)
        self.assertAlmostEqual(result["mae"], 1.8)
        self.assertAlmostEqual(result["selection_score"], 2.4)

    def test_fit_curve_is_seed_reproducible_and_does_not_mutate_initial(self):
        torch.manual_seed(7)
        initial = torch.nn.Linear(3, 1)
        old = copy.deepcopy(initial.state_dict())
        state, y = torch.ones(50, 3), torch.ones(50)
        units = np.asarray(["a"] * 25 + ["b"] * 25)
        for method in ("erm", "balanced", "dro"):
            a = fit_curve(initial, state, y, units, 7, method, steps=(1, 2))
            b = fit_curve(initial, state, y, units, 7, method, steps=(1, 2))
            for key, value in a[2].state_dict().items():
                torch.testing.assert_close(value, b[2].state_dict()[key], rtol=0, atol=0)
        for key, value in initial.state_dict().items():
            torch.testing.assert_close(value, old[key], rtol=0, atol=0)


class GeneralizationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.assets = load_release()

    def inputs(self):
        frame = {"subject_id": ["p", "p"], "stay_id": ["s", "s"],
                 "hours_since_onset": [0., 1.], "_time_hours": [0., 1.]}
        for i, v in enumerate(self.assets.checkpoint["variables"]):
            frame[f"{v}_t"] = self.assets.scaler.medians[i]
            frame[f"{v}_age_hr"] = 0.
        return candidate_inputs(pd.DataFrame(frame), self.assets, 3)

    def test_raw_probe_cannot_see_hidden_target_or_history(self):
        arrays = self.inputs()
        rows, hidden = np.asarray([1]), [0, 2]
        a = raw_view(arrays, rows, hidden)
        arrays["input"][:, hidden] = 999.
        arrays["current"][:, hidden] = 999.
        np.testing.assert_array_equal(a, raw_view(arrays, rows, hidden))

    def test_cached_body_projection_matches_same_shared_state(self):
        arrays = self.inputs()
        state, before_projection = cached_context(self.assets.model, arrays, np.asarray([1]), [0, 2])
        with torch.no_grad():
            expected = self.assets.model.state_encoder(before_projection)
        torch.testing.assert_close(state, expected, rtol=0, atol=0)

    def test_replay_does_not_require_future_labels_and_detects_changed_state(self):
        arrays = self.inputs()
        model = copy.deepcopy(self.assets.model).eval()
        rows = np.asarray([1])
        first = replay_loss(model, self.assets.model, arrays, rows, 3)
        self.assertLess(float(first.detach()), 1e-12)
        with torch.no_grad():
            model.state_encoder[0].bias[0] += .5
        second = replay_loss(model, self.assets.model, arrays, rows, 3)
        self.assertGreater(float(second.detach()), 0.)


if __name__ == "__main__":
    unittest.main()
