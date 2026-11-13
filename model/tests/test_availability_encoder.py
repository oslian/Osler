"""Ensure the trial updates context and keeps future labels stop-gradient."""
from copy import deepcopy
from pathlib import Path
import unittest
import torch

from whole_body_jepa.runtime import load_release
from scripts.train_availability_encoder import configure,teacher_target,CONTEXT_PREFIXES,drop_context_variables


class AvailabilityEncoderTest(unittest.TestCase):
    def test_empty_history_prefix_is_ignored_in_observed_steps_mode(self):
        model=deepcopy(self.assets.model)
        model.belief_input_mode='observed_steps'
        n=model.n_state
        idx=self.assets.checkpoint['variables'].index('glucose')
        current=torch.zeros(1,n); mask=torch.zeros_like(current);mask[:,idx]=1
        ages=torch.ones_like(current)*7;ages[:,idx]=0
        presence=(mask@model.variable_module_membership.T>0).float()
        history=torch.cat([current,mask,ages,torch.zeros(1,1),presence,torch.ones(1,1)*.1],1)[:,None,:]
        empty=torch.zeros(1,8,model.input_dim);empty[:,:,2*n:3*n]=7
        with torch.no_grad():
            a=model.encode_context_components(history,current,mask,ages,presence)
            b=model.encode_context_components(torch.cat([empty,history],1),current,mask,ages,presence)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertGreater(float(a[1].abs().sum()),0.)

    def test_context_dropout_preserves_time_and_target_and_does_not_mutate(self):
        n=3
        membership=torch.tensor([[1.,1.,0.],[0.,0.,1.]])
        current=torch.tensor([[0.,2.,3.],[4.,5.,6.]])
        mask=torch.ones_like(current); ages=torch.zeros_like(current)
        presence=torch.ones(2,2)
        history=torch.cat([current,mask,ages,torch.ones(2,1),presence,torch.ones(2,1)],dim=1)[:,None,:].repeat(1,3,1)
        inputs=(history,current,mask,ages,presence,torch.ones(2)*6)
        before=tuple(x.clone() for x in inputs)
        drop=torch.tensor([[False,True,True],[False,False,True]])
        h,c,m,a,p,t=drop_context_variables(inputs,drop,membership)
        for old,copy in zip(inputs,before):torch.testing.assert_close(old,copy)
        torch.testing.assert_close(c[:,0],current[:,0])
        self.assertEqual(float(m[0,0]),1.)  # Physical zero stays observed.
        self.assertTrue(bool((c[drop]==0).all()))
        self.assertTrue(bool((a[drop]==7).all()))
        torch.testing.assert_close(h[:,:,n:2*n],m[:,None,:].expand(-1,3,-1))
        torch.testing.assert_close(h[:,:,:n],c[:,None,:].expand(-1,3,-1))
        torch.testing.assert_close(p,torch.tensor([[1.,0.],[1.,0.]]))
        torch.testing.assert_close(h[:,:,3*n],history[:,:,3*n])
        torch.testing.assert_close(h[:,:,-1],history[:,:,-1])
        torch.testing.assert_close(t,inputs[-1])

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.assets=load_release(Path('models'))

    def test_context_trainability_and_teacher_stop_gradient(self):
        for enabled in (False,True):
            model=deepcopy(self.assets.model)
            configure(model,enabled)
            context=[p for n,p in model.named_parameters() if n.startswith(CONTEXT_PREFIXES)]
            self.assertTrue(context)
            self.assertEqual(any(p.requires_grad for p in context),enabled)
            index=self.assets.checkpoint['variables'].index('glucose')
            current=torch.zeros(2,model.n_state,requires_grad=True)
            target=teacher_target(model,current,torch.ones(2),index)
            self.assertEqual(target.shape,(2,model.latent))
            self.assertFalse(target.requires_grad)
            self.assertFalse(model.target_variable_embedding.requires_grad)

    def test_single_target_encoding_matches_existing_implementation(self):
        model=self.assets.model
        index=self.assets.checkpoint['variables'].index('glucose')
        current=torch.zeros(2,model.n_state)
        truth=torch.tensor([.1,-.2])
        actual=teacher_target(model,current,truth,index)
        values=torch.zeros_like(current); values[:,index]=truth
        mask=torch.zeros_like(current); mask[:,index]=1
        with torch.no_grad():
            expected,_=model._encode_target_specific_values(values,mask)
        torch.testing.assert_close(actual,expected[:,index],rtol=1e-5,atol=1e-6)

    def test_teacher_uses_only_selected_delta_at_nonzero_anchor(self):
        model=self.assets.model
        index=self.assets.checkpoint['variables'].index('glucose')
        current=torch.full((3,model.n_state),3.5,requires_grad=True)
        with torch.no_grad():current[:,index]=torch.tensor([-2.,0.,4.])
        delta=torch.tensor([.5,-.7,0.])
        truth=current[:,index].detach()+delta
        actual=teacher_target(model,current,truth,index)
        values=torch.zeros_like(current);values[:,index]=delta
        mask=torch.zeros_like(current);mask[:,index]=1
        with torch.no_grad():expected,_=model._encode_target_specific_values(values,mask)
        torch.testing.assert_close(actual,expected[:,index],rtol=1e-5,atol=1e-6)
        shifted=current.detach()+10
        same_delta=teacher_target(model,shifted,truth+10,index)
        torch.testing.assert_close(actual,same_delta,rtol=1e-5,atol=1e-6)
        self.assertFalse(actual.requires_grad)


if __name__=='__main__': unittest.main()
