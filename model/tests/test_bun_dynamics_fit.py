import unittest
import numpy as np
import torch
from scripts.audit_bun_dynamics_fit import smooth_l1_from_error


class SmoothLossTests(unittest.TestCase):
    def test_matches_training_library(self):
        e=torch.tensor([0.,.2,.999,1.,2.,9.],dtype=torch.float64)
        expected=torch.nn.functional.smooth_l1_loss(e,torch.zeros_like(e),reduction='none')
        np.testing.assert_allclose(smooth_l1_from_error(e.numpy()),expected.numpy())

    def test_invalid(self):
        for x in ([np.nan],[-1.]):
            with self.assertRaises(ValueError):smooth_l1_from_error(x)
