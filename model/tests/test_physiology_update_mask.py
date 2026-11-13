import unittest
import torch
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class PhysiologyUpdateMaskTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.layer = PersistentPhysiologyStateLayer(body_dim=4, organ_dim=3,
            organ_count=2, adjacency=torch.eye(2))
        self.state = self.layer.initialize(torch.ones(2, 4), torch.ones(2, 2, 3), torch.ones(2, 2))

    def test_missing_observations_do_not_rewrite_state(self):
        for value in (99., float('nan')):
            updated = self.layer.update(self.state, torch.full((2, 4), value),
                torch.full((2, 2, 3), value), torch.zeros(2, 2), torch.zeros(2))
            for name in ('body', 'organs', 'uncertainty', 'elapsed_hours'):
                torch.testing.assert_close(getattr(updated, name), getattr(self.state, name), rtol=0, atol=0)

    def test_only_present_organ_corrects_and_missing_patient_stays_unchanged(self):
        body = torch.full((2, 4), float('nan')); body[0] = 2
        organs = torch.full((2, 2, 3), float('nan')); organs[0, 0] = 2
        organs.requires_grad_(); body.requires_grad_()
        updated = self.layer.update(self.state, body, organs,
            torch.tensor([[1., 0.], [0., 0.]]), torch.zeros(2))
        self.assertTrue(torch.isfinite(updated.organs).all())
        torch.testing.assert_close(updated.organs[:, 1], self.state.organs[:, 1])
        torch.testing.assert_close(updated.body[1], self.state.body[1])
        self.assertGreater(float((updated.organs[0, 0]-self.state.organs[0, 0]).abs().sum().detach()), 0)
        (updated.body.sum()+updated.organs.sum()).backward()
        self.assertTrue(torch.isfinite(organs.grad).all())
        self.assertEqual(float(organs.grad[:, 1].abs().sum()), 0)

    def test_present_nan_and_invalid_time_rejected(self):
        with self.assertRaises(ValueError):
            self.layer.update(self.state, torch.ones(2, 4), torch.full((2, 2, 3), float('nan')),
                torch.ones(2, 2), torch.zeros(2))
        with self.assertRaises(ValueError):
            self.layer.update(self.state, torch.ones(2, 4), torch.ones(2, 2, 3),
                torch.ones(2, 2), -torch.ones(2))
