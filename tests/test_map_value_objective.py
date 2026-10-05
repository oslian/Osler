import unittest
import numpy as np
import torch
from scripts.audit_map_value_objective import smooth_l1


class ValueObjectiveTest(unittest.TestCase):
    def test_matches_training_loss_at_boundary_and_both_signs(self):
        error = np.array([-100., -1., -.5, 0., .5, 1., 100.])
        reference = torch.nn.functional.smooth_l1_loss(
            torch.tensor(error), torch.zeros(len(error), dtype=torch.float64), reduction='none')
        np.testing.assert_array_equal(smooth_l1(error), reference.numpy())


if __name__ == '__main__':
    unittest.main()
