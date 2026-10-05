import unittest
import torch
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class ObservationReplayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.layer = PersistentPhysiologyStateLayer(body_dim=4, organ_dim=3,
            organ_count=2, adjacency=torch.eye(2), body_feedback=False)
        self.initial = self.layer.initialize(torch.zeros(1, 4), torch.zeros(1, 2, 3), torch.zeros(1, 2))

    def test_prediction_then_update_counts_time_once(self):
        predicted = self.layer.advance(self.initial, torch.tensor([2.5]))
        updated = self.layer.update(predicted, torch.ones(1, 4), torch.ones(1, 2, 3),
            torch.tensor([[1., 0.]]), torch.tensor([2.5]))
        torch.testing.assert_close(updated.elapsed_hours, torch.tensor([2.5]))
        torch.testing.assert_close(updated.organs[:, 1], predicted.organs[:, 1])

    def test_shared_replay_matches_manual_sequence_and_backpropagates(self):
        bodies = torch.ones(1, 2, 4, requires_grad=True)
        organs = torch.ones(1, 2, 2, 3, requires_grad=True)
        presence = torch.tensor([[[1., 0.], [0., 1.]]])
        times = torch.tensor([[1., 2.5]])
        result = self.layer.replay_observations(self.initial, bodies, organs, presence, times)
        manual = self.initial
        for index, delta in enumerate((1., 1.5)):
            manual = self.layer.advance(manual, torch.tensor([delta]))
            manual = self.layer.update(manual, bodies[:, index], organs[:, index],
                presence[:, index], torch.tensor([delta]))
        for name in ('body', 'organs', 'uncertainty', 'elapsed_hours'):
            torch.testing.assert_close(getattr(result, name), getattr(manual, name))
        (result.body.sum()+result.organs.sum()).backward()
        self.assertGreater(float(organs.grad[0, 0, 0].abs().sum()), 0)
        self.assertGreater(float(organs.grad[0, 1, 1].abs().sum()), 0)
        self.assertEqual(float(organs.grad[0, 0, 1].abs().sum()), 0)

    def test_absent_observation_is_prediction_only(self):
        result = self.layer.replay_observations(self.initial, torch.full((1, 1, 4), float('nan')),
            torch.full((1, 1, 2, 3), float('nan')), torch.zeros(1, 1, 2), torch.tensor([[2.]]))
        expected = self.layer.advance(self.initial, torch.tensor([2.]))
        torch.testing.assert_close(result.body, expected.body)
        torch.testing.assert_close(result.organs, expected.organs)
        torch.testing.assert_close(result.uncertainty, expected.uncertainty)

    def test_rejects_duplicate_or_backwards_events(self):
        for times in ([[1., 1.]], [[2., 1.]], [[-1., 2.]]):
            with self.assertRaises(ValueError):
                self.layer.replay_observations(self.initial, torch.ones(1, 2, 4),
                    torch.ones(1, 2, 2, 3), torch.ones(1, 2, 2), torch.tensor(times))
