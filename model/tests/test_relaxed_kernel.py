import unittest
import numpy as np
import pandas as pd
from scripts.train_kernel_probe import fit_regressor
from scripts.train_relaxed_kernel import refit,MULTIPLIER
from scripts.score_relaxed_kernel import assess,ALL_ARMS


class RelaxedKernelTests(unittest.TestCase):
    def test_only_C_changes_and_parent_is_untouched(self):
        rng=np.random.default_rng(9); x=rng.normal(size=(80,6)); y=3*x[:,0]+rng.normal(size=80); w=np.ones(80)
        model,offset,_=fit_regressor(x,y,w); before=model.coef_.copy()
        old=dict(model=model,offset=offset,mapping={'identity':'frozen'},scaler=object())
        new,diag=refit(old,x,y,w)
        self.assertEqual(diag['converged'],diag['convergence_warnings']==0 and diag['iterations']<model.max_iter)
        self.assertEqual(new['model'].C,model.C*MULTIPLIER)
        self.assertIs(new['mapping'],old['mapping']); self.assertIs(new['scaler'],old['scaler'])
        self.assertEqual(new['offset'],offset)
        np.testing.assert_array_equal(model.coef_,before)
        p,q=model.get_params(),new['model'].get_params(); q['C']=p['C']; self.assertEqual(p,q)

    def test_statistical_roles_do_not_swap_old_and_candidate(self):
        rows=[]
        for p in range(40):
            for f in range(2):
                row=dict(patient=str(p),component=str(p//10),fold=f,target='x',truth=0.)
                row.update({a:1.+i+p*.001*(i+1) for i,a in enumerate(ALL_ARMS)})
                rows.append(row)
        frame=pd.DataFrame(rows); r=assess(frame,4,50)
        self.assertEqual(set(r['candidates']),{'relaxed_body','relaxed_raw'})
        for a in ALL_ARMS: self.assertAlmostEqual(r['normalized_mae']['patient'][a],frame[a].mean())
        self.assertIn('scaled_body',r['candidates']['relaxed_body']['comparisons'])
        self.assertNotIn('kernel_body',r['candidates']['relaxed_body']['comparisons'])
        self.assertAlmostEqual(r['candidates']['relaxed_body']['comparisons']['scaled_body']['paired']['patient']['mean'],
                               float((frame.relaxed_body-frame.scaled_body).mean()))
        with self.assertRaises(ValueError): assess(pd.concat([frame,frame.iloc[:1]]),4,50)

    def test_nonconvergence_remains_visible(self):
        rng=np.random.default_rng(5); x=rng.normal(size=(100,8)); y=rng.normal(size=100); w=np.ones(100)
        model,offset,_=fit_regressor(x,y,w)
        model.set_params(max_iter=1)
        _,diag=refit(dict(model=model,offset=offset),x,y,w)
        self.assertFalse(diag['converged'])
        self.assertGreater(diag['convergence_warnings'],0)


if __name__=='__main__': unittest.main()
