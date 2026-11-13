from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.diagnose_adam_origins import AdamOrigins,loss_contributions,project,summarize,classify,trace_fit
from scripts.train_frozen_bridge import fit_bridge
from tests import test_parameter_path


class AdamOriginsTest(unittest.TestCase):
    def test_prior_target_momentum_survives_another_targets_step(self):
        w=torch.nn.Parameter(torch.zeros(2))
        opt=torch.optim.AdamW([w],lr=.01,weight_decay=.1)
        account=AdamOrigins(2,2)
        for target,g in ((0,[1.,-2.]),(1,[0.,0.]),(1,[.2,.4]),(0,[0.,0.])):
            before=w.detach().double().numpy().copy();w.grad=torch.tensor(g)
            parts_before=account.parts.copy();opt.step()
            account.record(before,w.detach().double().numpy(),g,opt.state[w],opt.param_groups[0],target)
            np.testing.assert_allclose(account.parts.sum(0),w.detach().numpy(),atol=1e-12)
            if account.steps==2:
                self.assertGreater(np.linalg.norm(account.parts[0]-parts_before[0]),0.)
                np.testing.assert_array_equal(account.parts[1],0.)
        self.assertGreater(np.linalg.norm(account.parts[-2]),0.)

    def test_invalid_step_and_gradient(self):
        account=AdamOrigins(2,2)
        with self.assertRaises(ValueError):account.record([0,0],[0,0],[np.nan,0],{}, {},0)
        with self.assertRaises(ValueError):account.record([0,0],[0,0],[0,0],{}, {},2)
        with self.assertRaises(ValueError):loss_contributions([1,np.nan],0.)

    def test_zero_residual_total_ray_not_independent_absolute_costs(self):
        x=loss_contributions([2.,-1.],0.)
        np.testing.assert_array_equal(x,[2.,-1.]);self.assertEqual(x.sum(),1.)
        self.assertNotEqual(x.sum(),abs(2.)+abs(-1.))
        np.testing.assert_array_equal(loss_contributions([2.,-2.],0.),[0,0])

    def test_exact_fit_and_full_origin_projection(self):
        helper=test_parameter_path.ParameterPathTest()
        candidate,e=helper.candidate();original=deepcopy(candidate.model).float()
        e={k:v.float() if k in ('body','organs','presence') else v for k,v in e.items()}
        examples=[dict(e,truth=2.,target=0),dict(e,truth=-1.,target=1)]
        bound=[e['organs'],e['organs']]
        baseline,log=fit_bridge(original,examples,bound,'immediate')
        traced,parts,info=trace_fit(original,examples,bound,2)
        self.assertEqual(log,info['training'])
        for k,v in baseline.state_dict().items():torch.testing.assert_close(v,traced.state_dict()[k],atol=0,rtol=0)
        double=traced.double().requires_grad_(False)
        saved=deepcopy(double.state_dict())
        b,s=test_parameter_path.tangent(double,e)
        with torch.inference_mode():
            z,values=project(double,e,parts)
        self.assertEqual(z,b);self.assertAlmostEqual(values.sum(),s,places=12)
        for k,v in double.state_dict().items():torch.testing.assert_close(v,saved[k],atol=0,rtol=0)

    def test_weighting_unsupported_and_conflict_classification(self):
        sources=['map','bun','weight_decay','rounding']
        f=pd.DataFrame([dict(patient=p,component=c,fold=fold,context=0,variable='map',
            origin_map=-1.,origin_bun=other,origin_weight_decay=0.,origin_rounding=0.,
            parameter_derivative=other-1) for p,c,fold,other in
            [('a','x',0,2.),('a','x',1,2.),('b','y',0,5.),('c','y',0,5.)]])
        r,error=summarize(f,sources);self.assertLess(error,1e-12)
        self.assertEqual(r['bun'],dict(status='unsupported'))
        self.assertEqual(r['map']['metrics']['patient']['other'],4.)
        self.assertEqual(r['map']['metrics']['component']['other'],3.5)
        self.assertEqual(classify(r,'patient')['counts']['own_helpful_other_harmful'],1)
        with self.assertRaises(ValueError):summarize(pd.concat([f,f.iloc[:1]]),sources)


if __name__=='__main__':unittest.main()
