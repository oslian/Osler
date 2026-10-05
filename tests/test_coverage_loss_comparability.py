import unittest
import numpy as np
import torch
from scripts.audit_coverage_loss_comparability import smooth_l1_numpy


class LossTests(unittest.TestCase):
    def test_matches_training_loss(self):
        x=np.array([-3.,-1.,-.2,0.,.3,1.,4.])
        expected=torch.nn.functional.smooth_l1_loss(torch.tensor(x),torch.zeros(7,dtype=torch.float64),reduction='none').numpy()
        np.testing.assert_allclose(smooth_l1_numpy(x),expected)


if __name__=='__main__':unittest.main()
