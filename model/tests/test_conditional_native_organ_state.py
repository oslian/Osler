import unittest
import torch
from whole_body_joint_jepa import JointWholeBodyJEPAWorldModel
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.conditional_native_organ_state import ConditionalNativeOrganStateResidual as NativeOrganStateResidual


class ConditionalNativeStateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        jepa = JointWholeBodyJEPAWorldModel(3,14,3,hidden=12,latent=8,token_dim=8,
            variable_module_ids=[0,1,2], horizon_values=(3.,), target_encoder_mode='fast')
        expand_history_projection(jepa,6); jepa.belief_input_mode='observed_steps'
        self.model = NativeOrganStateResidual(jepa,('map','creatinine','glucose'),('map','creatinine')).eval()
        tokens = torch.zeros(2,20); tokens[:,3:6] = 1
        self.args = [tokens, torch.tensor([0.,1.]),1.,torch.ones(1,2),torch.ones(1,2),
            torch.tensor([[3.,6.]]),torch.ones(1,2),torch.ones(1,2,8)]

    def test_zero_start_exactly_preserves_both_baselines(self):
        out = self.model(*self.args)
        torch.testing.assert_close(out['value'],self.args[-2],rtol=0,atol=0)
        torch.testing.assert_close(out['future_latent'],self.args[-1],rtol=0,atol=0)

    def test_no_future_event(self):
        self.args[1] = torch.tensor([0.,2.])
        with self.assertRaises(ValueError): self.model(*self.args)

    def test_absolute_double_clock_does_not_create_negative_anchor_gap(self):
        self.args[1] = torch.tensor([12345.01,12346.01],dtype=torch.float64)
        self.args[2] = 12346.01
        result = self.model(*self.args)
        torch.testing.assert_close(result['value'],self.args[-2],rtol=0,atol=0)

    def test_state_path_receives_supervision_after_zero_start_head_step(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(); out = self.model(*self.args)
            loss = (out['value']-2).square().mean()+(out['future_latent']-2).square().mean()
            loss.backward()
            optimizer.step()
        grad = self.model.layer.organ_dynamics[0].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()),0)

