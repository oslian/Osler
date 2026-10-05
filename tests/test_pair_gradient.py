from copy import deepcopy
import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch
from scripts.train_pair_gradient import combine,fit_pair,describe,assess,CONTROLS
from tests import test_parameter_path


class PairGradientTest(unittest.TestCase):
    def test_conflict_and_symmetry(self):
        g=torch.tensor([1.,0.],dtype=torch.float64);h=torch.tensor([-1.,1.],dtype=torch.float64)
        m=combine(g,h,True)
        torch.testing.assert_close(m,torch.tensor([.25,.75],dtype=torch.float64))
        self.assertGreaterEqual(float(g@m),0);self.assertGreaterEqual(float(h@m),0)
        torch.testing.assert_close(m,combine(h,g,True))
        torch.testing.assert_close(combine(g,-g,True),torch.zeros_like(g))

    def test_nonconflict_zero_and_invalid(self):
        g=torch.tensor([1.,2.]);h=torch.tensor([2.,1.])
        torch.testing.assert_close(combine(g,h,True),combine(g,h,False),atol=0,rtol=0)
        torch.testing.assert_close(combine(g,torch.zeros_like(g),True),g/2)
        torch.testing.assert_close(combine(g,g,True),g)
        with self.assertRaises(ValueError):combine(g,torch.tensor([float('nan'),0.]),True)
        with self.assertRaises(ValueError):combine(g,torch.ones(3),True)

    def test_tiny_conflict_does_not_underflow(self):
        g=torch.tensor([1e-30,0.]);h=torch.tensor([-1e-30,1e-30])
        actual=combine(g,h,True)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.double()/1e-30,torch.tensor([.25,.75],dtype=torch.float64))

    def test_same_target_fit_exact_and_frozen(self):
        c,e=test_parameter_path.ParameterPathTest().candidate();original=deepcopy(c.model).float()
        e={k:v.float() if k in ('body','organs','presence') else v for k,v in e.items()}
        examples=[dict(e,truth=2.,target=0),dict(e,truth=-1.,target=0)]
        bound=[e['organs']]*2;a,al=fit_pair(original,examples,bound,'mean');b,bl=fit_pair(original,examples,bound,'pcgrad')
        self.assertEqual(al,bl)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
        for k,v in original.state_dict().items():torch.testing.assert_close(v,b.model.state_dict()[k],atol=0,rtol=0)
        self.assertEqual(sum(x['steps'] for x in al),2)
        self.assertEqual(sum(x['examples'] for x in al),4)
        self.assertGreater(float(b.bridge.projection.weight.detach().norm()),0)
        with self.assertRaises(ValueError):fit_pair(original,examples[:1],bound[:1],'pcgrad')

    def test_patient_component_weighting(self):
        f=pd.DataFrame([dict(patient=p,component=c,fold=fold,context=0,variable='map',truth=0.,pcgrad=v,mean=2.)
            for p,c,fold,v in [('a','x',0,1.),('a','x',1,1.),('b','y',0,4.),('c','y',0,4.)]])
        r=describe(f,['map','bun'])
        self.assertEqual(r['map']['normalized_mae']['patient']['pcgrad'],3.)
        self.assertEqual(r['map']['normalized_mae']['component']['pcgrad'],2.5)
        self.assertEqual(r['bun'],dict(status='unsupported'))
        with self.assertRaises(ValueError):describe(pd.concat([f,f.iloc[:1]]),['map'])
        with self.assertRaises(ValueError):describe(f,['bun'])

    def test_fixed_screen_support_and_family(self):
        f=pd.DataFrame([dict(patient=str(i),component=str(i%3),fold=0,context=0,
            variable='map',truth=0.,pcgrad=0.,**{c:1. for c in CONTROLS}) for i in range(30)])
        good=dict(ci95=[-1.,-.1],p_less=0.)
        with patch('scripts.train_pair_gradient.paired_summary',return_value=good):
            self.assertTrue(all(x['screen_pass'] for x in assess(f,1)['comparisons'].values()))
            self.assertFalse(any(x['screen_pass'] for x in assess(f.iloc[:29],1)['comparisons'].values()))
            self.assertFalse(any(x['screen_pass'] for x in assess(f.assign(component='one'),1)['comparisons'].values()))
        for bad in (dict(ci95=[-1.,0.],p_less=0.),dict(ci95=[-1.,-.1],p_less=.05/(53*len(CONTROLS)))):
            with patch('scripts.train_pair_gradient.paired_summary',return_value=bad):
                self.assertFalse(any(x['screen_pass'] for x in assess(f,1)['comparisons'].values()))


if __name__=='__main__':unittest.main()
