from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.diagnose_parameter_path import (tangent,endpoints,finite_difference,path_metrics,
                                            summarize,classify,clean_example)
from scripts.train_organ_assimilation import AssimilatedReconstruction
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class ParameterPathTest(unittest.TestCase):
    def candidate(self):
        torch.manual_seed(83)
        model=nn.Module();model.trajectory=nn.Module()
        model.trajectory.layer=PersistentPhysiologyStateLayer(body_dim=8,organ_dim=4,
            organ_count=3,adjacency=torch.eye(3)).double()
        model.readout=nn.Linear(8,2).double()
        candidate=AssimilatedReconstruction(model,'immediate').double().eval().requires_grad_(False)
        with torch.no_grad():candidate.bridge.projection.weight.normal_(0,.3)
        e=dict(body=torch.randn(3,8,dtype=torch.double),organs=torch.randn(3,3,4,dtype=torch.double),
            presence=torch.ones(3,3,dtype=torch.double),times=torch.tensor([0.,1.,2.],dtype=torch.double),
            anchor=3.,target=0)
        return candidate,e

    def test_real_recurrent_gradient_finite_difference_and_no_mutation(self):
        model,e=self.candidate();saved=deepcopy(model.state_dict())
        b,s=tangent(model,e);self.assertNotEqual(s,0)
        self.assertEqual(b,endpoints(model,e)[0])
        self.assertLess(finite_difference(model,e,s),1e-6)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_inference_mode_generator_isolation(self):
        model,e=self.candidate();expected=tangent(model,e)
        with torch.inference_mode():
            inferred={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in e.items()}
            self.assertTrue(inferred['body'].is_inference())
            self.assertEqual(tangent(model,inferred),expected)
        self.assertTrue(torch.is_grad_enabled())

    def test_zero_weight_zero_direction(self):
        model,e=self.candidate()
        with torch.no_grad():model.bridge.projection.weight.zero_()
        b,s=tangent(model,e);self.assertEqual(s,0.);self.assertEqual(endpoints(model,e),[b,b])

    def test_time_precision_survives_model_dtype_conversion(self):
        _,e=self.candidate()
        e['times']=torch.tensor([0.,1.0000000001,2.123456789],dtype=torch.float64)
        with torch.inference_mode():
            original={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in e.items()}
        with torch.inference_mode(False):
            copy=clean_example(original,torch.float32)
        self.assertEqual(copy['body'].dtype,torch.float32)
        self.assertEqual(copy['times'].dtype,torch.float64)
        torch.testing.assert_close(copy['times'],original['times'],atol=0,rtol=0)
        self.assertFalse(copy['times'].is_inference())

    def test_nonlinear_path_and_zero_residual_decomposition(self):
        # p(a)=a-2a^2 moves initially toward y=1, but ends on the wrong side.
        m=path_metrics([0,0,0],[ -1,2,-2],[1,0,1],[1,-1,-1])
        np.testing.assert_allclose(m.parameter_derivative,[-1,1,1])
        np.testing.assert_allclose(m.nonlinear_remainder,[2,1,1])
        np.testing.assert_allclose(m.mae_delta,m.parameter_derivative+m.nonlinear_remainder+m.crossing_curvature)

    def test_patient_component_weighting_and_classification(self):
        f=pd.DataFrame([dict(patient=p,component=c,fold=fold,context=0,variable='map',
            truth=1.,base64=0.,pred64=-1.,slope64=s) for p,c,fold,s in
            [('a','x',0,1.),('a','x',1,1.),('b','y',0,-4.),('c','y',0,-4.)]])
        results,error=summarize(f,['map','bun']);self.assertLess(error,1e-12)
        self.assertEqual(results['bun'],dict(status='unsupported'))
        self.assertAlmostEqual(results['map']['metrics']['patient']['parameter_derivative'],7/3)
        self.assertAlmostEqual(results['map']['metrics']['component']['parameter_derivative'],1.5)
        self.assertEqual(classify(results,'patient')['counts']['harmful_initial_direction'],1)
        f['slope64']=1.
        results,_=summarize(f,['map'])
        self.assertEqual(classify(results,'patient')['counts']['harmful_chord_beneficial_initial'],1)

    def test_invalid_slopes_and_nonfinite(self):
        for s in ([np.nan],[],[1,2]):
            with self.assertRaises(ValueError):path_metrics([0],[1],[2],s)
        with self.assertRaises(ValueError):path_metrics([np.inf],[1],[2],[0])


if __name__=='__main__':unittest.main()
