import unittest

import numpy as np
import torch

from scripts.masked_guided_mixture_readouts import (
    GuidedMixtureReadout, components, fit_family, gate_diagnostics, predict,
)


class GuidedMixtureTests(unittest.TestCase):
    def test_parameter_counts_and_ordered_fit_only_bins(self):
        y = torch.linspace(-3, 3, 101)
        three = GuidedMixtureReadout(64, y, 3)
        five = GuidedMixtureReadout(64, y, 5)
        self.assertEqual(sum(p.numel() for p in three.parameters()), 3561)
        self.assertEqual(sum(p.numel() for p in five.parameters()), 3535)
        self.assertTrue(torch.all(three.boundaries[1:] > three.boundaries[:-1]))
        self.assertEqual(three.labels(torch.tensor([-9., 0., 9.])).tolist(), [0, 1, 2])

    def test_every_branch_gets_finite_gradient(self):
        torch.manual_seed(7)
        x, y = torch.randn(90, 64), torch.cat([torch.randn(30) - 3, torch.randn(30), torch.randn(30) + 3])
        head = GuidedMixtureReadout(64, y, 5)
        optimizer = torch.optim.Adam(head.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(x, y)
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                            for p in head.parameters()))

    def test_shuffling_only_changes_routing(self):
        torch.manual_seed(19)
        head = GuidedMixtureReadout(64, torch.linspace(-2, 2, 30), 3)
        torch.nn.init.normal_(head.net[-1].weight, std=.1)
        x = torch.randn(20, 64)
        donors = np.roll(np.arange(20), 2)
        legal = components([head], [x])
        shuffled = components([head], [x], "shuffled", [donors])
        torch.testing.assert_close(shuffled[0], legal[0][donors])
        torch.testing.assert_close(shuffled[1], legal[1])
        torch.testing.assert_close(shuffled[2], legal[2])
        with self.assertRaises(ValueError):
            predict([head], [x], "oracle")

    def test_tiny_training_is_reproducible_and_disjoint(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        x = torch.randn(100, 64)
        y = x[:, 0] + torch.randn(100) * .1
        fit = {"latent": x[:50], "y": y[:50], "subjects": np.arange(50), "units": ["a"] * 50}
        tune = {"latent": x[50:], "y": y[50:], "subjects": np.arange(50, 100), "units": ["b"] * 50}
        one, info_one = fit_family("guided_3", fit, tune, "x", steps=(2, 4), seeds=(7, 19))
        two, info_two = fit_family("guided_3", fit, tune, "x", steps=(2, 4), seeds=(7, 19))
        self.assertEqual(info_one, info_two)
        np.testing.assert_array_equal(predict(one["heads"], [tune["latent"]] * 2),
                                      predict(two["heads"], [tune["latent"]] * 2))
        diagnostics = gate_diagnostics(one["heads"], [tune["latent"]] * 2, tune["y"])
        self.assertTrue(0 <= diagnostics["mean_gate_bin_accuracy"] <= 1)
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_family("guided_3", fit, fit, "x", steps=(2,), seeds=(7,))


if __name__ == "__main__":
    unittest.main()
