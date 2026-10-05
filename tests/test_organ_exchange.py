import unittest

import torch
import pandas as pd

from scripts.audit_organ_exchange import label_arrays, TARGETS

from whole_body_jepa.organ_exchange import ARMS, ORGANS, OrganExchangeResidual, graph_for, mask_to_owner
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class OrganExchangeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def layer(self, feedback=False):
        return PersistentPhysiologyStateLayer(body_dim=4, organ_dim=4, organ_count=3,
                                             adjacency=torch.eye(3), body_feedback=feedback)

    def test_body_bypass_closed_over_multiple_steps(self):
        layer = self.layer()
        a = layer.initialize(torch.zeros(1, 4), torch.randn(1, 3, 4), torch.ones(1, 3))
        changed = a.organs.clone()
        changed[:, 0] += 3
        b = layer.initialize(torch.ones(1, 4) * 9, changed, torch.ones(1, 3))
        x, y = (layer.rollout(s, torch.tensor([1., 3., 6.])) for s in (a, b))
        torch.testing.assert_close(x["organs"][:, :, 1:], y["organs"][:, :, 1:], atol=0, rtol=0)

    def test_existing_feedback_can_bypass_identity_graph(self):
        layer = self.layer(True)
        organs = torch.randn(1, 3, 4)
        a = layer.initialize(torch.zeros(1, 4), organs, torch.ones(1, 3))
        b = layer.initialize(torch.ones(1, 4), organs, torch.ones(1, 3))
        self.assertGreater(float((layer.transition(a, torch.ones(1)).organs
                                  - layer.transition(b, torch.ones(1)).organs).abs().max().detach()), 0)

    def test_equal_parameter_counts_and_zero_start(self):
        counts = []
        for arm in ARMS:
            model = OrganExchangeResidual([3, 3], arm=arm, latent=4)
            counts.append(sum(p.numel() for p in model.parameters()))
            base, latent = torch.randn(2, 2), torch.randn(2, 2, 4)
            out = model(torch.randn(2, len(ORGANS), 4), torch.ones(2, len(ORGANS)),
                        torch.randn(2, 2), torch.ones(2, 2), torch.tensor([[6., 3.], [2., 1.]]), base, latent)
            torch.testing.assert_close(out["value"], base, atol=0, rtol=0)
            torch.testing.assert_close(out["future_latent"], latent, atol=0, rtol=0)
        self.assertEqual(len(set(counts)), 1)

    def test_future_query_time_does_not_modify_other_target(self):
        model = OrganExchangeResidual([3, 3], arm="sparse", latent=4)
        for head in model.value_heads:
            torch.nn.init.normal_(head[-1].weight)
        args = (torch.randn(2, len(ORGANS), 4), torch.ones(2, len(ORGANS)),
                torch.randn(2, 2), torch.ones(2, 2))
        tail = (torch.randn(2, 2), torch.randn(2, 2, 4))
        x = model(*args, torch.tensor([[2.3, 3.5], [2.3, 3.5]]), *tail)
        y = model(*args, torch.tensor([[2.3, 8.7], [2.3, 8.7]]), *tail)
        torch.testing.assert_close(x["value"][:, 0], y["value"][:, 0], atol=0, rtol=0)

    def test_non_owner_history_is_removed_without_mutating_input(self):
        args = (torch.ones(2, 5, 13), torch.ones(2, 3), torch.ones(2, 3),
                torch.ones(2, 3), torch.ones(2, 2))
        result = mask_to_owner(*args, allowed=[1])
        self.assertTrue((result[0][..., [0, 2, 3, 5, 6, 8]] == 0).all())
        self.assertTrue((result[0][..., 1] == 1).all())
        self.assertTrue((result[0][..., 9] == 1).all())  # onset is preserved
        self.assertTrue((result[0][..., 10:12] == 0).all())
        self.assertTrue((args[0] == 1).all())

    def test_graph_changes_only_buffers_and_is_normalized(self):
        for arm in ARMS:
            graph = graph_for(arm)
            torch.testing.assert_close(graph.sum(-1), torch.ones(len(ORGANS)))
        self.assertFalse(torch.equal(graph_for("sparse"), graph_for("permuted")))
        self.assertEqual(int((graph_for("sparse") > 0).sum()), int((graph_for("permuted") > 0).sum()))

    def test_strict_labels_require_times_and_do_not_accept_missing_or_stale(self):
        columns = {}
        for target in TARGETS:
            columns.update({f"{target}_t": [1., 1., 1.], f"{target}_age_hr": [0., 9., 0.],
                f"future_{target}": [2., 2., 2.], f"future_delta_t_hr_{target}": [5.9, 6., float("nan")],
                f"future_label_quality_{target}": ["strict"] * 3})
        frame = pd.DataFrame(columns)
        valid, time = label_arrays(frame, 6., strict=True)
        self.assertTrue(valid[0].all())
        self.assertFalse(valid[1:].any())
        self.assertAlmostEqual(float(time[0, 0]), 5.9, places=5)
        with self.assertRaises(ValueError):
            label_arrays(frame.drop(columns=["future_delta_t_hr_bun"]), 6., strict=True)

    def test_value_loss_reaches_transition_after_zero_head_starts_learning(self):
        model = OrganExchangeResidual([3], arm="sparse", latent=4)
        args = (torch.randn(2, len(ORGANS), 4), torch.ones(2, len(ORGANS)),
                torch.randn(2, 1), torch.ones(2, 1), torch.ones(2, 1) * 3,
                torch.zeros(2, 1), torch.zeros(2, 1, 4))
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(2):
            opt.zero_grad()
            (model(*args)["value"] - 1).square().mean().backward()
            opt.step()
        grad = model.state_layer.organ_dynamics[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
