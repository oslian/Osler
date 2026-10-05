import unittest

import torch

import numpy as np

from whole_body_joint_jepa import (
    PersistentWholeBodyJEPAWorldModel,
    _coupling_batch_view,
    _owner_organ_batch_view,
)
from whole_body_jepa.physiology_state import (
    LinearConservationProjector,
    PersistentPhysiologyStateLayer,
    PhysiologicalReadoutProjector,
    body_system_adjacency,
)


class PhysiologyStateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_linear_conservation_is_exact(self) -> None:
        projector = LinearConservationProjector(torch.tensor([[1.0, 1.0, 1.0]]))
        state = torch.tensor([[2.0, 3.0, 9.0], [1.0, 1.0, 1.0]])
        totals = torch.tensor([[10.0], [12.0]])
        projected = projector(state, totals)
        actual = projected.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(actual, totals, atol=1e-6, rtol=1e-6)

    def test_derived_readouts_are_exact(self) -> None:
        variables = (
            "anion_gap",
            "bicarbonate",
            "bun",
            "chloride",
            "glucose",
            "serum_osmolality",
            "sodium",
        )
        projector = PhysiologicalReadoutProjector(
            variables,
            medians=[0.0] * len(variables),
            scales=[1.0] * len(variables),
        )
        values = torch.tensor([[99.0, 18.0, 28.0, 102.0, 180.0, 999.0, 138.0]])
        projected = projector(values)
        self.assertAlmostEqual(float(projected[0, 0]), 18.0, places=5)
        self.assertAlmostEqual(float(projected[0, 5]), 296.0, places=5)

    def test_multistep_rollout_is_path_consistent(self) -> None:
        adjacency = body_system_adjacency(("cardiovascular_perfusion", "renal_urinary"))
        layer = PersistentPhysiologyStateLayer(
            body_dim=5,
            organ_dim=3,
            organ_count=2,
            adjacency=adjacency,
            max_step_hours=1.0,
        )
        initial = layer.initialize(
            torch.randn(2, 5),
            torch.randn(2, 2, 3),
            torch.ones(2, 2),
        )
        direct = layer.rollout(initial, torch.tensor([[2.0], [2.0]]))
        first = layer.rollout(initial, torch.tensor([[1.0], [1.0]]))
        intermediate = initial.__class__(
            body=first["body"][:, -1],
            organs=first["organs"][:, -1],
            uncertainty=first["uncertainty"][:, -1],
            elapsed_hours=torch.ones(2),
        )
        second = layer.rollout(intermediate, torch.tensor([[1.0], [1.0]]))
        torch.testing.assert_close(
            direct["body"][:, -1], second["body"][:, -1], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            direct["organs"][:, -1], second["organs"][:, -1], atol=1e-6, rtol=1e-6
        )

    def test_connected_organ_can_inform_masked_organ(self) -> None:
        adjacency = body_system_adjacency(
            ("cardiovascular_perfusion", "renal_urinary", "hepatic_synthetic")
        )
        layer = PersistentPhysiologyStateLayer(
            body_dim=4,
            organ_dim=3,
            organ_count=3,
            adjacency=adjacency,
        )
        body = torch.zeros(1, 4)
        organs = torch.zeros(1, 3, 3)
        presence = torch.tensor([[1.0, 0.0, 0.0]])
        initial = layer.initialize(body, organs, presence)
        baseline = layer.rollout(initial, torch.tensor([[1.0]]))["organs"]
        informed_organs = organs.clone()
        informed_organs[:, 0] = torch.tensor([1.0, -0.5, 0.25])
        informed = layer.initialize(body, informed_organs, presence)
        response = layer.rollout(informed, torch.tensor([[1.0]]))["organs"]
        renal_change = (response[:, :, 1] - baseline[:, :, 1]).abs().sum()
        self.assertGreater(float(renal_change.detach()), 0.0)

    def test_observation_update_reduces_present_organ_uncertainty(self) -> None:
        layer = PersistentPhysiologyStateLayer(
            body_dim=4,
            organ_dim=3,
            organ_count=2,
            adjacency=torch.eye(2),
        )
        initial = layer.initialize(
            torch.zeros(1, 4), torch.zeros(1, 2, 3), torch.zeros(1, 2)
        )
        updated = layer.update(
            initial,
            torch.ones(1, 4),
            torch.ones(1, 2, 3),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([1.0]),
        )
        self.assertLess(float(updated.uncertainty[0, 0].detach()), 1.0)
        self.assertEqual(float(updated.uncertainty[0, 1].detach()), 1.0)

    def test_all_target_heads_read_one_persistent_trajectory(self) -> None:
        model = PersistentWholeBodyJEPAWorldModel(
            4,
            7,
            2,
            hidden=12,
            latent=8,
            token_dim=8,
            variable_module_ids=[0, 0, 1, 1],
            variable_module_membership=[[1, 1, 0, 0], [0, 0, 1, 1]],
            module_names=("cardiovascular_perfusion", "renal_urinary"),
            horizon_values=(1.0, 3.0),
            target_encoder_mode="fast",
        )
        output = model(
            torch.randn(2, 3, 7),
            torch.randn(2, 4),
            torch.ones(2, 4),
            torch.zeros(2, 4),
            torch.ones(2, 2),
            torch.tensor([3.0, 3.0]),
            window_horizons=torch.tensor([1.0, 3.0]),
        )
        self.assertEqual(output["value"].shape, (2, 4))
        self.assertEqual(output["predicted_trajectory_latent"].shape, (2, 2, 8))
        self.assertEqual(output["predicted_organ_trajectory"].shape, (2, 2, 2, 8))
        self.assertEqual(
            output["predicted_target_trajectory_latent"].shape,
            (2, 2, 4, 8),
        )
        self.assertEqual(output["predicted_physiology_body"].shape, (2, 2, 8))
        torch.testing.assert_close(
            output["persistent_value_delta"],
            torch.zeros_like(output["persistent_value_delta"]),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output["value"], output["base_value"], atol=0.0, rtol=0.0
        )
        self.assertNotIn("persistent_state_gate", output)

    def test_persistent_future_starts_from_trained_base_trajectory(self) -> None:
        model = PersistentWholeBodyJEPAWorldModel(
            4,
            7,
            2,
            hidden=12,
            latent=8,
            token_dim=8,
            variable_module_ids=[0, 0, 1, 1],
            variable_module_membership=[[1, 1, 0, 0], [0, 0, 1, 1]],
            module_names=("cardiovascular_perfusion", "renal_urinary"),
            horizon_values=(1.0, 3.0),
            target_encoder_mode="fast",
        )
        context = torch.randn(3, 8)
        organs = torch.randn(3, 2, 8)
        presence = torch.ones(3, 2)
        horizons = torch.tensor([[1.0, 3.0]]).expand(3, -1)
        actual = model._predict_state_trajectory(
            context, organs, presence, horizons
        )["body"]
        expected = super(PersistentWholeBodyJEPAWorldModel, model)._predict_state_trajectory(
            context, organs, presence, horizons
        )["body"]
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

        target_latents = torch.randn(3, 2, 4, 8)
        state = model._predict_state_trajectory(
            context, organs, presence, horizons
        )
        conditioned = model._condition_target_latents_on_state(
            target_latents, state
        )
        torch.testing.assert_close(
            conditioned, target_latents, atol=0.0, rtol=0.0
        )

    def test_persistent_cross_organ_graph_is_fail_closed(self) -> None:
        model = PersistentWholeBodyJEPAWorldModel(
            4,
            7,
            2,
            hidden=12,
            latent=8,
            token_dim=8,
            variable_module_ids=[0, 0, 1, 1],
            variable_module_membership=[[1, 1, 0, 0], [0, 0, 1, 1]],
            module_names=("cardiovascular_perfusion", "renal_urinary"),
            horizon_values=(1.0, 3.0),
            target_encoder_mode="fast",
        )
        torch.testing.assert_close(
            model.persistent_physiology.adjacency,
            torch.eye(2),
            atol=0.0,
            rtol=0.0,
        )

    def test_target_focused_coupling_does_not_mask_unsupervised_target(self) -> None:
        view = _coupling_batch_view(
            batch=np.asarray([0]),
            current=np.ones((1, 2), dtype=np.float32),
            mask=np.ones((1, 2), dtype=np.float32),
            ages=np.zeros((1, 2), dtype=np.float32),
            presence=np.ones((1, 2), dtype=np.float32),
            input_matrix=np.ones((1, 10), dtype=np.float32),
            history=np.asarray([[0]]),
            future_mask=np.ones((1, 2), dtype=np.float32),
            variable_module_membership=np.eye(2, dtype=np.float32),
            rng=np.random.default_rng(17),
            eligible_target_mask=np.asarray([False, True]),
        )
        self.assertEqual(float(view[5][0, 0]), 0.0)
        self.assertEqual(float(view[5][0, 1]), 1.0)
        self.assertEqual(float(view[1][0, 0]), 1.0)
        self.assertEqual(float(view[1][0, 1]), 0.0)

    def test_owner_organ_view_removes_non_owner_physio_but_keeps_owner(self) -> None:
        view = (
            np.ones((2, 4), dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 3, 15), dtype=np.float32),
            np.zeros((2, 4), dtype=bool),
        )
        owner = _owner_organ_batch_view(
            view,
            target_index=0,
            variable_module_membership=np.asarray(
                [[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32
            ),
        )
        np.testing.assert_array_equal(owner[1][:, :2], 1.0)
        np.testing.assert_array_equal(owner[1][:, 2:], 0.0)
        np.testing.assert_array_equal(owner[3][:, 0], 1.0)
        np.testing.assert_array_equal(owner[3][:, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
