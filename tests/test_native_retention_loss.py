import unittest
import torch
from scripts.train_native_glucose_retention import retention_loss


class RetentionLossTest(unittest.TestCase):
    def test_empty_and_excluded(self):
        p = torch.tensor([[2., 10.]], requires_grad=True)
        y = torch.zeros_like(p)
        loss = retention_loss(p,y,torch.tensor([[True,False]])); loss.backward()
        self.assertEqual(float(p.grad[0,1]),0.)
        self.assertEqual(float(retention_loss(p,y,torch.zeros_like(p,dtype=torch.bool))),0.)

    def test_target_balanced(self):
        p = torch.tensor([[2.,4.],[2.,0.]])
        loss = retention_loss(p,torch.zeros_like(p),torch.tensor([[True,True],[True,False]]))
        self.assertAlmostEqual(float(loss),2.5)
