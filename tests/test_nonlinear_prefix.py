import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from scripts.probe_nonlinear_prefix import STAGES, ARMS, HEADS, crossfit_stage, stage_predictions, assess


class PrefixTests(unittest.TestCase):
    def test_held_group_truth_does_not_select_its_stage(self):
        groups = np.repeat(['a','b','c'],20)
        pred = np.tile(np.arange(6.),(60,1))
        truth = np.zeros(60)
        p,s,ok,_ = crossfit_stage(truth,pred,groups)
        changed = truth.copy(); changed[:20] = 5.
        q,t,_,_ = crossfit_stage(changed,pred,groups)
        np.testing.assert_array_equal(p[:20],q[:20])
        np.testing.assert_array_equal(s[:20],t[:20])
        self.assertTrue(ok.all())
        self.assertTrue((s==1).all())

    def test_equal_components_not_patient_count_and_ties_choose_earliest(self):
        groups = np.array(['a']*5+['b']*100+['c']*30)
        truth = np.zeros(len(groups))
        p = np.full((len(groups),6),10.)
        p[groups=='b',0],p[groups=='b',1] = 0.,3.
        p[groups=='c',0],p[groups=='c',1] = 4.,0.
        _,stage,_,_ = crossfit_stage(truth,p,groups)
        self.assertTrue((stage[groups=='a']==5).all())
        _,stage,_,_ = crossfit_stage(truth,np.zeros_like(p),groups)
        self.assertTrue((stage==1).all())

    def test_unsupported_is_explicit_not_a_pass(self):
        p,s,ok,rows = crossfit_stage([0,1],np.ones((2,6)),['a','b'])
        self.assertFalse(ok.any())
        self.assertTrue((s==100).all())
        self.assertTrue(all(r['stage'] is None for r in rows))

    def test_public_stages_reproduce_predict_without_mutation(self):
        x = np.arange(80.).reshape(-1,1)
        with threadpool_limits(limits=1):
            m = HistGradientBoostingRegressor(max_iter=100,early_stopping=False,random_state=19).fit(x,np.sin(x[:,0]))
            before = m.predict(x)
        staged = stage_predictions(m,x)
        np.testing.assert_array_equal(staged[:,-1],before)
        self.assertEqual(m.n_iter_,100)

    def test_invalid_and_outer_inputs_rejected(self):
        with self.assertRaises(ValueError): crossfit_stage([np.nan],np.ones((1,6)),['a'])
        with self.assertRaises(ValueError): crossfit_stage([0],np.ones((1,5)),['a'])
        frame = pd.DataFrame(dict(patient=['p'],fold=[0],target=['x'],component=['a'],role=['test'],truth=[0.],supported=[True],**{a:[0.] for a in ARMS}))
        with self.assertRaises(ValueError): assess(frame,19,20)

    def test_assessment_averages_loss_not_predictions_across_folds(self):
        frame = pd.DataFrame(dict(patient=['p','p','q'],fold=[0,1,0],target=['x']*3,component=['a','a','b'],role=['selection']*3,truth=[0.]*3,supported=[True]*3,**{a:[1.,-1.,3.] for a in ARMS},**{a+'_stage':[1,1,5] for a in HEADS}))
        result = assess(frame,19,20)
        self.assertEqual(result['patients'],2)
        self.assertEqual(result['normalized_mae']['patient']['body_probe_prefix'],2.)
        self.assertFalse(any(c['diagnostic_family_screen'] for c in result['comparisons'].values()))


if __name__=='__main__': unittest.main()
