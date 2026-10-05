import unittest
import torch
from scripts.polish_reconstruction_bias import shifted_state


class BiasPolishTests(unittest.TestCase):
    def test_only_requested_bias_changes(self):
        original={'readout.bias':torch.tensor([1.,2.]),'readout.weight':torch.ones(2,3)}
        updated=shifted_state(original,['a','b'],{'b':-.25})
        self.assertTrue(torch.equal(original['readout.bias'],torch.tensor([1.,2.])))
        self.assertTrue(torch.equal(updated['readout.bias'],torch.tensor([1.,1.75])))
        x=torch.randn(7,3)
        delta=torch.nn.functional.linear(x,updated['readout.weight'],updated['readout.bias'])-torch.nn.functional.linear(x,original['readout.weight'],original['readout.bias'])
        self.assertTrue(torch.allclose(delta,torch.tensor([0.,-.25]).expand_as(delta),atol=1e-6))
