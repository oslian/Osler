import unittest
import torch
from scripts.audit_expanded_reconstruction_numeric_path import neutralize_numbers


class NumericPathTest(unittest.TestCase):
    def test_preserves_every_non_numeric_channel(self):
        x=torch.arange(30,dtype=torch.float32).reshape(3,10);original=x.clone()
        y=neutralize_numbers(x,2)
        self.assertTrue(torch.equal(x,original))
        self.assertTrue(torch.equal(y[:,2:],x[:,2:]))
        self.assertTrue((y[:,:2]==0).all())

    def test_schema_rejects(self):
        with self.assertRaises(ValueError):neutralize_numbers(torch.ones(3,4),2)
