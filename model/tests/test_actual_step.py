from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

from scripts import train_actual_step as audit
from tests import test_parameter_path


class ActualStepTest(unittest.TestCase):
    def test_projection_against_scipy_and_symmetry(self):
        rng = np.random.default_rng(42)
        for _ in range(40):
            d, a = rng.normal(size=5), rng.normal(size=(2,5))
            answer = audit.project_step(torch.tensor(d), list(torch.tensor(a))).numpy()
            opt = minimize(lambda x:.5*np.sum((x-d)**2),np.zeros(5),jac=lambda x:x-d,
                constraints={'type':'ineq','fun':lambda x:-a@x,'jac':lambda x:-a},
                method='SLSQP',options={'ftol':1e-12,'maxiter':200})
            self.assertTrue(opt.success)
            np.testing.assert_allclose(answer,opt.x,atol=1e-7,rtol=0)
            np.testing.assert_allclose(answer,audit.project_step(torch.tensor(d),list(torch.tensor(a[::-1].copy()))),atol=1e-12)
            self.assertLessEqual(float(np.max(a@answer)),1e-12)

    def test_degenerate_zero_opposing_tiny_and_invalid(self):
        d = torch.tensor([1.,2.],dtype=torch.float64); g = torch.tensor([1.,0.],dtype=torch.float64)
        for normals in ([g,g],[g,-g],[g,torch.zeros_like(g)],[g*1e-30,-g*1e-30]):
            torch.testing.assert_close(audit.project_step(d,normals),torch.tensor([0.,2.],dtype=torch.float64))
        torch.testing.assert_close(audit.project_step(-d,[g,g]),-d,atol=0,rtol=0)
        torch.testing.assert_close(audit.project_step(d,[g*0,g*0]),d,atol=0,rtol=0)
        with self.assertRaises(ValueError):audit.project_step(d,[g])
        with self.assertRaises(ValueError):audit.project_step(d,[g,torch.ones(3)])
        with self.assertRaises(ValueError):audit.project_step(d,[g,g*float('nan')])

    def test_stored_float32_and_unchanged_safe_step(self):
        before=torch.tensor([1.,1.]); after=torch.tensor([1.1,1.2]);g=torch.tensor([1.,0.])
        value, changed, veto=audit.corrected_weight(before,after,[g,-g])
        self.assertTrue(changed);self.assertFalse(veto);self.assertEqual(float(value[0]),1.)
        self.assertFalse(audit.harmful(g,value.double()-before.double()))
        value,changed,veto=audit.corrected_weight(before,after,[-g,-g])
        self.assertFalse(changed);torch.testing.assert_close(value,after,atol=0,rtol=0)
        with patch.object(audit,'project_step',return_value=torch.ones(2,dtype=torch.float64)):
            value,changed,veto=audit.corrected_weight(before,after,[g,g])
            self.assertTrue(veto);torch.testing.assert_close(value,before,atol=0,rtol=0)

    def test_actual_model_matches_parent_when_disabled(self):
        c,e=test_parameter_path.ParameterPathTest().candidate();original=deepcopy(c.model).float()
        e={k:v.float() if k in ('body','organs','presence') else v for k,v in e.items()}
        examples=[dict(e,truth=t,target=i%2) for i,t in enumerate((2.,-1.,3.,-2.))]
        bound=[e['organs']]*4
        parent,logs=audit.base.fit_pair(original,examples,bound,'pcgrad')
        candidate,new,trace=audit.fit(original,examples,bound,protect=False)
        for k,v in parent.state_dict().items():torch.testing.assert_close(v,candidate.state_dict()[k],atol=0,rtol=0)
        for a,b in zip(logs,new):
            for k in b:self.assertEqual(a[k],b[k])
        candidate,_,trace=audit.fit(original,examples,bound)
        part=trace[trace.cross_target]
        self.assertFalse((part.actual_dot>part.actual_tolerance).any())
        audit.base.assert_frozen(candidate,original)
        self.assertEqual(len(trace),8)
        with self.assertRaises(ValueError):audit.fit(original,[],[])

    def test_same_target_no_projection_and_trace(self):
        c,e=test_parameter_path.ParameterPathTest().candidate();original=deepcopy(c.model).float()
        e={k:v.float() if k in ('body','organs','presence') else v for k,v in e.items()}
        es=[dict(e,truth=t,target=0) for t in (2.,-1.)];bound=[e['organs']]*2
        a,_,trace=audit.fit(original,es,bound);b,_,_=audit.fit(original,es,bound,False)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
        self.assertFalse(trace.projected.any())
        r=audit.trace_summary(trace.assign(fold=0));self.assertEqual(r['all']['examples'],4)
        self.assertEqual(r['cross_target']['examples'],0)

    def test_fixed_screen_and_patient_weighting(self):
        f=pd.DataFrame([dict(patient=str(i),component=str(i%3),fold=0,context=0,variable='map',
            truth=0.,actual_step=0.,**{c:1. for c in audit.CONTROLS}) for i in range(30)])
        with patch.object(audit.base,'paired_summary',return_value=dict(ci95=[-1.,-.1],p_less=0.)):
            self.assertTrue(all(r['screen_pass'] for r in audit.assess(f,1)['comparisons'].values()))
            self.assertFalse(any(r['screen_pass'] for r in audit.assess(f.iloc[:29],1)['comparisons'].values()))
        with patch.object(audit.base,'paired_summary',return_value=dict(ci95=[-1.,-.1],p_less=.05/(53*10))):
            self.assertFalse(any(r['screen_pass'] for r in audit.assess(f,1)['comparisons'].values()))
        self.assertEqual(audit.describe(f,['map','bun'])['bun'],dict(status='unsupported'))
        with self.assertRaises(ValueError):audit.describe(pd.concat([f,f.iloc[:1]]),['map'])


if __name__=='__main__':unittest.main()
