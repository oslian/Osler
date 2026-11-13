"""Characterize the existing direct decoder's hard amplitude bound."""
from copy import deepcopy
from pathlib import Path
import unittest
import torch
from whole_body_jepa.runtime import load_release


class ConstantHead(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x.new_full((len(x),1),self.value)


class DecoderBoundTest(unittest.TestCase):
    def test_identity_activation_removes_bound_without_new_parameters(self):
        torch.set_num_threads(1)
        assets=load_release(Path('models'))
        model=deepcopy(assets.model)
        keys=set(model.state_dict())
        model.value_delta_activation=torch.nn.Identity()
        self.assertEqual(keys,set(model.state_dict()))
        x=torch.tensor([-4.,4.],requires_grad=True)
        delta=.75*model.value_delta_activation(x)
        torch.testing.assert_close(delta,torch.tensor([-3.,3.]))
        delta.sum().backward()
        torch.testing.assert_close(x.grad,torch.full((2,),.75))

    def test_arbitrary_head_strength_cannot_exceed_bound(self):
        torch.set_num_threads(1)
        assets=load_release(Path('models'))
        model=deepcopy(assets.model)
        index=assets.checkpoint['variables'].index('glucose')
        model.uncertainty_gate_enabled=False
        current=torch.zeros(2,model.n_state)
        for sign in (-1,1):
            model.value_heads[index]=ConstantHead(sign*10000.)
            with torch.no_grad():
                out=model._decode_latent(torch.zeros(2,model.latent),current,
                    torch.full((2,),6.),current_mask=torch.ones_like(current))
            torch.testing.assert_close(out['value'][:,index],torch.full((2,),sign*.75))


if __name__=='__main__':
    unittest.main()
