import unittest
import torch
from whole_body_joint_jepa import JointWholeBodyJEPAWorldModel
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction,mask_native_target


class BodyReconstructionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        j=JointWholeBodyJEPAWorldModel(3,14,3,hidden=12,latent=8,token_dim=8,
            variable_module_ids=[0,1,2],horizon_values=(3.,),target_encoder_mode='fast')
        expand_history_projection(j,6);j.belief_input_mode='observed_steps'
        self.model=NativeBodyReconstruction(NativeOrganStateResidual(j,('map','creatinine','glucose'),('map',))).eval()
        self.tokens=torch.zeros(2,20);self.tokens[:,3:6]=1
        self.times=torch.tensor([0.,1.])

    def test_target_entire_history_cannot_change_reconstruction(self):
        before=self.model(self.tokens,self.times,1.,'map')
        altered=self.tokens.clone();altered[:,0]=torch.tensor([100.,-100.]);altered[:,6]=99
        torch.testing.assert_close(before,self.model(altered,self.times,1.,'map'),atol=0,rtol=0)

    def test_body_update_and_dynamics_receive_gradient(self):
        self.model(self.tokens,self.times,2.,'map').square().sum().backward()
        layer=self.model.trajectory.layer
        for module in (layer.body_update_gate,layer.body_dynamics):
            total=sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)
            self.assertGreater(total,0)

    def test_empty_independent_history_rejected(self):
        self.tokens[:,4:6]=0
        with self.assertRaises(ValueError):self.model(self.tokens,self.times,1.,'map')

    def test_brothers_and_target_only_event_removed(self):
        tokens=torch.zeros(2,20);tokens[0,3:5]=1;tokens[1,5]=1
        clean,t=mask_native_target(tokens,self.times,('hemoglobin','hematocrit','glucose'),'hemoglobin')
        self.assertEqual(len(clean),1)
        torch.testing.assert_close(t,self.times[1:])

    def test_invalid_times_cannot_be_hidden_by_target_mask(self):
        tokens=self.tokens.clone()
        tokens[1,4:6]=0
        for time in (2.,float('nan'),float('inf')):
            with self.subTest(time=time), self.assertRaises(ValueError):
                self.model(tokens,torch.tensor([0.,time]),1.,'map')

    def test_out_of_order_target_only_event_rejected(self):
        tokens=self.tokens.clone()
        tokens[1,4:6]=0
        with self.assertRaises(ValueError):
            self.model(tokens,torch.tensor([1.,0.]),1.,'map')
