import unittest

import numpy as np
import torch

from osler_jepa.adjacent_hierarchy import (
    AdjacentThreeLevelResidualAdapter,
    assert_adjacent_hierarchy,
    split_adjacent_training_epochs,
)


class AdjacentThreeLevelHierarchyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = AdjacentThreeLevelResidualAdapter(
            target_indices=(0, 1),
            target_module_membership=np.eye(2, dtype=np.float32),
            latent_dim=6,
            organ_dim=4,
            module_count=2,
            organ_names=("heart", "kidneys"),
            adapter_dim=8,
        )
        self.inputs = {
            "teacher_anchor": torch.tensor([[1.0, -0.5], [0.2, 0.4]]),
            "context_latent": torch.randn(2, 6),
            "target_latent": torch.randn(2, 2, 6),
            "organ_latents": torch.randn(2, 2, 4),
            "organ_presence": torch.ones(2, 2),
            "current": torch.tensor([[0.8, -0.4], [0.1, 0.5]]),
            "current_mask": torch.ones(2, 2),
            "ages": torch.zeros(2, 2),
            "horizon": torch.tensor([6.0, 12.0]),
        }

    def test_contract_has_only_adjacent_layer_edges(self):
        assert_adjacent_hierarchy(self.model)
        names = set(dict(self.model.named_parameters()))
        self.assertFalse(any("system_to_body" in name for name in names))
        self.assertFalse(any("body_to_system" in name for name in names))

    def test_zero_initialized_residual_preserves_expert_anchor(self):
        output = self.model(**self.inputs)
        torch.testing.assert_close(output["value"], self.inputs["teacher_anchor"])
        self.assertEqual(tuple(output["system_tokens"].shape), (2, 2, 8))
        self.assertEqual(tuple(output["organ_tokens"].shape), (2, 2, 8))
        self.assertEqual(tuple(output["body_token"].shape), (2, 8))

    def test_distributional_scale_is_target_specific_and_positive(self):
        model = AdjacentThreeLevelResidualAdapter(
            target_indices=(0, 1),
            target_module_membership=np.eye(2, dtype=np.float32),
            latent_dim=6,
            organ_dim=4,
            module_count=2,
            organ_names=("heart", "kidneys"),
            adapter_dim=8,
            distributional_scale=True,
        )
        output = model(**self.inputs)
        self.assertEqual(tuple(output["scale"].shape), (2, 2))
        self.assertTrue(torch.isfinite(output["scale"]).all())
        self.assertTrue(torch.all(output["scale"] > 0.0))
        self.assertTrue(hasattr(model, "scale_heads"))

    def test_scale_loss_cannot_update_physiology_latent(self):
        model = AdjacentThreeLevelResidualAdapter(
            target_indices=(0, 1),
            target_module_membership=np.eye(2, dtype=np.float32),
            latent_dim=6,
            organ_dim=4,
            module_count=2,
            organ_names=("heart", "kidneys"),
            adapter_dim=8,
            distributional_scale=True,
        )
        model.zero_grad(set_to_none=True)
        model(**self.inputs)["scale"].sum().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if not name.startswith("scale_heads")
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if name.startswith("scale_heads")
            )
        )

    def test_hierarchy_gates_start_neutral_without_moving_anchor(self):
        for parameter in (
            self.model.organ_feedback_gates,
            self.model.system_feedback_gates,
            self.model.body_system_feedback_gates,
            self.model.organ_residual_gates,
            self.model.body_residual_gates,
        ):
            torch.testing.assert_close(
                torch.sigmoid(parameter),
                torch.full_like(parameter, 0.5),
            )
        output = self.model(**self.inputs)
        torch.testing.assert_close(output["value"], self.inputs["teacher_anchor"])

    def test_body_context_reaches_system_only_through_organ_feedback(self):
        altered = dict(self.inputs)
        altered["context_latent"] = self.inputs["context_latent"] + 10.0

        no_route_a = self.model(**self.inputs, disable_organ_route=True)
        no_route_b = self.model(**altered, disable_organ_route=True)
        torch.testing.assert_close(
            no_route_a["system_tokens"], no_route_b["system_tokens"]
        )

        routed_a = self.model(**self.inputs)
        routed_b = self.model(**altered)
        self.assertFalse(
            torch.allclose(routed_a["system_tokens"], routed_b["system_tokens"])
        )

    def test_expert_values_enter_at_system_layer(self):
        with_expert = self.model(**self.inputs)
        without_expert = self.model(
            **self.inputs, disable_sparse_experts=True
        )
        self.assertFalse(
            torch.allclose(
                with_expert["system_tokens"], without_expert["system_tokens"]
            )
        )

    def test_adjacent_organs_exchange_messages_and_ablation_stops_it(self):
        altered = dict(self.inputs)
        altered["organ_latents"] = self.inputs["organ_latents"].clone()
        altered["organ_latents"][:, 1] += 8.0

        coupled_a = self.model(**self.inputs)
        coupled_b = self.model(**altered)
        self.assertFalse(
            torch.allclose(
                coupled_a["organ_tokens"][:, 0],
                coupled_b["organ_tokens"][:, 0],
            )
        )
        self.assertTrue(
            torch.any(torch.abs(coupled_a["target_organ_coupling"]) > 0.0)
        )
        target_weights = coupled_a["target_organ_coupling_weights"]
        torch.testing.assert_close(
            target_weights[:, 0, 1], torch.ones(2)
        )
        torch.testing.assert_close(
            target_weights[:, 1, 0], torch.ones(2)
        )
        self.assertTrue(torch.all(target_weights[:, 0, 0] == 0.0))
        self.assertTrue(torch.all(target_weights[:, 1, 1] == 0.0))

        isolated_a = self.model(
            **self.inputs, disable_organ_coupling=True
        )
        isolated_b = self.model(
            **altered, disable_organ_coupling=True
        )
        torch.testing.assert_close(
            isolated_a["organ_tokens"][:, 0],
            isolated_b["organ_tokens"][:, 0],
        )
        torch.testing.assert_close(
            isolated_a["target_organ_coupling"],
            torch.zeros_like(isolated_a["target_organ_coupling"]),
        )
        torch.testing.assert_close(
            isolated_a["target_organ_coupling_weights"],
            torch.zeros_like(isolated_a["target_organ_coupling_weights"]),
        )

    def test_absent_organ_cannot_send_or_receive_messages(self):
        inputs = dict(self.inputs)
        inputs["organ_presence"] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        inputs["current_mask"] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        output = self.model(**inputs)
        weights = output["organ_coupling_weights"]
        self.assertTrue(torch.all(weights[:, 0, 1] == 0.0))
        self.assertTrue(torch.all(weights[:, 1] == 0.0))

    def test_system_evidence_can_activate_an_unmeasured_organ_node(self):
        inputs = dict(self.inputs)
        inputs["organ_presence"] = torch.zeros(2, 2)
        output = self.model(**inputs)
        torch.testing.assert_close(
            output["organ_presence"],
            torch.ones_like(output["organ_presence"]),
        )

    def test_missing_expert_is_explicit_and_never_propagates_nan(self):
        inputs = dict(self.inputs)
        inputs["teacher_anchor"] = self.inputs["teacher_anchor"].clone()
        inputs["teacher_anchor"][0, 1] = float("nan")
        output = self.model(**inputs)
        for key in ("value", "system_tokens", "organ_tokens", "body_token"):
            self.assertTrue(torch.isfinite(output[key]).all(), key)
        # Normalized zero is the population-median fallback. It is not a
        # physical zero and the expert-availability bit remains in the token.
        self.assertEqual(float(output["value"][0, 1].detach()), 0.0)

    def test_system_stage_disables_hierarchy_and_freezes_hierarchy_parameters(self):
        self.model.set_training_stage("system")
        staged = self.model(**self.inputs)
        no_route = self.model(**self.inputs, disable_organ_route=True)
        torch.testing.assert_close(staged["value"], no_route["value"])
        self.assertTrue(
            all(
                parameter.requires_grad
                for name, parameter in self.model.named_parameters()
                if name.startswith("system_residual_heads")
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in self.model.named_parameters()
                if name.startswith("organ_residual_heads")
            )
        )

    def test_organ_stage_freezes_system_and_body_predictors(self):
        self.model.set_training_stage("organ")
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in self.model.named_parameters()
                if name.startswith("system_residual_heads")
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for name, parameter in self.model.named_parameters()
                if name.startswith("organ_residual_heads")
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in self.model.named_parameters()
                if name.startswith("body_residual_heads")
            )
        )

    def test_every_stage_interface_is_trained_by_its_stage_point_loss(self):
        interfaces = {
            "organ": ("system_to_organ", "observed_organ_projection", "organ_to_system", "system_feedback_gates",
                      "organ_residual_heads", "organ_residual_gates", "target_organ_value", "target_organ_message"),
            "body": ("organ_to_body", "observed_body_projection", "body_to_organ", "organ_feedback_gates",
                     "body_system_feedback_gates", "body_residual_heads", "body_residual_gates"),
        }
        torch.manual_seed(0)
        for stage, modules in interfaces.items():
            for parameter in self.model.parameters():         # a partly trained model: no exact-zero heads
                if parameter.abs().sum() == 0:
                    parameter.data.normal_(0.0, 0.1)
            self.model.set_training_stage(stage)
            self.model.zero_grad()
            self.model(**self.inputs)["value"].pow(2).sum().backward()
            for module in modules:
                grads = [p.grad for n, p in self.model.named_parameters() if n == module or n.startswith(f"{module}.")]
                self.assertTrue(grads and all(g is not None for g in grads), f"{stage}: {module} has no gradient")
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0, f"{stage}: {module} gets no point-loss signal")

    def test_structural_ablation_holds_in_training_and_prediction(self):
        from osler_jepa.adjacent_hierarchy import HIERARCHY_MODELS, MATCHED_ABLATIONS

        self.assertEqual(set(MATCHED_ABLATIONS), set(HIERARCHY_MODELS))
        for model, ablations in MATCHED_ABLATIONS.items():
            for ablation in ablations:                                   # each ablation removes exactly one component
                self.assertEqual(len(set(HIERARCHY_MODELS[ablation]) ^ set(HIERARCHY_MODELS[model])), 1)
        torch.manual_seed(0)
        model = AdjacentThreeLevelResidualAdapter(
            target_indices=(0, 1), target_module_membership=np.eye(2, dtype=np.float32), latent_dim=6, organ_dim=4,
            module_count=2, organ_names=("heart", "kidneys"), adapter_dim=8, disabled_paths=("body_feedback",))
        for parameter in model.parameters():
            if parameter.abs().sum() == 0:
                parameter.data.normal_(0.0, 0.1)
        self.assertEqual(model.trainable_stages(), ("system", "organ"))
        model.set_training_stage("all")
        output = model(**self.inputs)
        output["value"].pow(2).sum().backward()
        body = [p.grad for n, p in model.named_parameters() if n.startswith(("body_residual_heads", "body_to_organ"))]
        self.assertTrue(all(g is None or float(g.abs().sum()) == 0.0 for g in body))   # the disabled path never trains
        torch.testing.assert_close(output["body_residual"], torch.zeros_like(output["body_residual"]))
        system_only = AdjacentThreeLevelResidualAdapter(
            target_indices=(0, 1), target_module_membership=np.eye(2, dtype=np.float32), latent_dim=6, organ_dim=4,
            module_count=2, organ_names=("heart", "kidneys"), adapter_dim=8, disabled_paths=HIERARCHY_MODELS["system_only"])
        self.assertEqual(system_only.trainable_stages(), ("system",))
        with self.assertRaises(ValueError):
            AdjacentThreeLevelResidualAdapter(
                target_indices=(0,), target_module_membership=np.eye(1, 2, dtype=np.float32), latent_dim=6, organ_dim=4,
                module_count=2, organ_names=("heart", "kidneys"), disabled_paths=("system_to_body",))

    def test_body_stage_freezes_system_and_organ_predictors(self):
        self.model.set_training_stage("body")
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in self.model.named_parameters()
                if name.startswith("system_residual_heads")
                or name.startswith("organ_residual_heads")
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for name, parameter in self.model.named_parameters()
                if name.startswith("body_residual_heads")
            )
        )

    def test_invalid_training_stage_fails_closed(self):
        with self.assertRaises(ValueError):
            self.model.set_training_stage("invalid")

    def test_organ_names_must_match_module_count(self):
        with self.assertRaises(ValueError):
            AdjacentThreeLevelResidualAdapter(
                target_indices=(0, 1),
                target_module_membership=np.eye(2, dtype=np.float32),
                latent_dim=6,
                organ_dim=4,
                module_count=2,
                organ_names=("kidneys",),
            )

    def test_epoch_budget_is_split_without_increasing_it(self):
        self.assertEqual(split_adjacent_training_epochs(3), (1, 1, 1))
        self.assertEqual(split_adjacent_training_epochs(7), (2, 2, 3))
        with self.assertRaises(ValueError):
            split_adjacent_training_epochs(2)


if __name__ == "__main__":
    unittest.main()
