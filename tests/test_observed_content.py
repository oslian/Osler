import unittest
import numpy as np
import pandas as pd
import torch
from scripts.probe_observed_content import contexts,last_observed,assess,ARMS


class ObservedContentTest(unittest.TestCase):
    def test_context_sampling_one_per_patient_order_independent(self):
        keys=[('a',0),('a',1),('b',0),('c',2),('c',1)]
        chosen=contexts(keys,3)
        self.assertEqual(chosen,contexts(keys[::-1],3))
        self.assertEqual(len({p for p,c in chosen}),3)
        self.assertEqual(contexts(keys,1),chosen[:1])
        with self.assertRaises(ValueError): contexts(keys+keys[:1],3)

    def test_zero_is_observed_and_equal_repeat_uses_latest_event(self):
        x=torch.tensor([[0.,7.,1.,0.],[0.,8.,1.,1.],[9.,8.,0.,1.]])
        present,pos=last_observed(x,torch.tensor([0.,1.,2.]),3.,2)
        np.testing.assert_array_equal(present,[True,True])
        np.testing.assert_array_equal(pos,[1,2])

    def test_missing_has_no_position(self):
        x=torch.tensor([[0.,7.,1.,0.]])
        present,pos=last_observed(x,torch.tensor([0.]),1.,2)
        np.testing.assert_array_equal(present,[True,False])
        np.testing.assert_array_equal(pos,[0,-1])

    def test_time_and_mask_guard(self):
        x=torch.tensor([[1.,1.],[2.,1.]])
        for t in (torch.tensor([2.,1.]),torch.tensor([0.,3.]),torch.tensor([0.,float('nan')])):
            with self.assertRaises(ValueError): last_observed(x,t,2.,1)
        with self.assertRaises(ValueError): last_observed(torch.tensor([[1.,.5]]),torch.tensor([0.]),1.,1)

    def test_patient_aggregation_not_duplicate_fold_independence(self):
        f=pd.DataFrame(dict(patient=['a','a','b'],fold=[0,1,0],component=['x','x','y'],variable=['sodium']*3,truth=[0.]*3))
        for a in ARMS: f[a]=[1.,3.,6.]
        r=assess(f,0,repetitions=20)
        self.assertEqual(r['normalized_mae']['patient']['ridge_event'],4.)
        self.assertEqual(r['patients'],2)
        self.assertFalse(any(x['screen_pass'] for x in r['comparisons'].values()))
        with self.assertRaises(ValueError): assess(pd.concat([f,f.iloc[:1]]),0,repetitions=20)


if __name__=='__main__': unittest.main()
