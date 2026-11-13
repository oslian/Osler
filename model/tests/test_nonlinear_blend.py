import unittest

import numpy as np
import pandas as pd

from scripts.probe_nonlinear_blend import fit_weight,crossfit,assess,ARMS,HEADS


def frame():
    n=60
    return pd.DataFrame(dict(patient=[str(i) for i in range(n)],fold=[0]*n,target=['x']*n,
        role=['selection']*n,component=np.repeat(['a','b','c'],20),truth=np.ones(n),
        native=np.zeros(n),population=np.zeros(n),body_probe=np.full(n,2.),
        process_summary=np.full(n,3.),trained_shuffle=np.full(n,-1.)))


class BlendTests(unittest.TestCase):
    def test_exact_solution_matches_dense_objective(self):
        rng=np.random.default_rng(19)
        a,b,y=rng.normal(size=(3,120))
        groups=np.array(['a']*100+['b']*20)
        w=fit_weight(y,a,b,groups)
        def loss(v): return pd.Series(abs((1-v)*a+v*b-y)).groupby(groups).mean().mean()
        self.assertLessEqual(loss(w),min(loss(v) for v in np.linspace(0,1,101))+1e-12)
        self.assertGreaterEqual(w,0); self.assertLessEqual(w,1)

    def test_held_truth_cannot_change_own_weight(self):
        original=frame(); modified=original.copy(); modified.loc[:19,'truth']=100
        p,_=crossfit(original); q,_=crossfit(modified)
        for arm in HEADS:
            np.testing.assert_array_equal(p.loc[:19,arm+'_blend'],q.loc[:19,arm+'_blend'])
        self.assertTrue((p.body_probe_weight==.5).all())

    def test_constant_and_tiny_proposals(self):
        self.assertEqual(fit_weight([2],[1],[1],['a']),0)
        self.assertEqual(fit_weight([1],[0],[1e-200],['a']),1)
        self.assertEqual(fit_weight([-1],[0],[1e-200],['a']),0)
        self.assertEqual(fit_weight([0],[1],[-1],['a']),.5)

    def test_unsupported_explicit_and_outer_rejected(self):
        f=frame().iloc[:20]
        p,rows=crossfit(f)
        self.assertFalse(p.supported.any()); self.assertTrue(p.body_probe_weight.isna().all())
        self.assertEqual(rows[0]['status'],'insufficient_training_support')
        bad=f.copy(); bad['role']='test'
        with self.assertRaises(ValueError): crossfit(bad)
        with self.assertRaises(ValueError): fit_weight([np.nan],[0],[1],['a'])

    def test_cancellation_is_not_best_member_claim(self):
        p,_=crossfit(frame()); r=assess(p,19,20)
        self.assertEqual(r['complementarity']['cancellation_vs_average_member_loss_patient'],1)
        self.assertEqual(r['normalized_mae']['patient']['equal_blend'],0)
        p['body_probe']=p.truth; p['equal_blend']=(p.native+p.body_probe)/2
        r=assess(p,19,20)
        self.assertEqual(r['complementarity']['cancellation_vs_average_member_loss_patient'],0)
        self.assertGreater(r['normalized_mae']['patient']['equal_blend'],r['normalized_mae']['patient']['body_probe'])

    def test_duplicate_fold_rows_rejected(self):
        p,_=crossfit(frame())
        with self.assertRaises(ValueError): assess(pd.concat([p,p]),19,20)


if __name__=='__main__': unittest.main()
