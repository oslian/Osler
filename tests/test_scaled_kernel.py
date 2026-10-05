import unittest
import numpy as np
import pandas as pd
from scripts.train_scaled_kernel import standardize_fit, fit_heads
from scripts.train_kernel_probe import fit_map, map_query
from scripts.score_scaled_kernel import assess, ALL_ARMS


class ScaledKernelTests(unittest.TestCase):
    def test_unit_change_does_not_change_metric_or_landmarks(self):
        rng=np.random.default_rng(7); x=rng.normal(size=(40,8)); w=np.linspace(.5,2,40)
        scaler,z=standardize_fit(x,w)
        other,zz=standardize_fit(x*np.arange(1,9)*100+40,w)
        np.testing.assert_allclose(z,zz,atol=1e-13)
        m,a=fit_map(z,w,4); n,b=fit_map(zz,w,4)
        np.testing.assert_array_equal(m['kernel'].component_indices_,n['kernel'].component_indices_)
        np.testing.assert_allclose(a,b,atol=1e-7)
        self.assertAlmostEqual(m['kernel'].gamma,n['kernel'].gamma,places=12)

    def test_fit_only_and_constant_features(self):
        x=np.array([[0.,1.,5.],[2.,3.,5.],[4.,5.,5.]])
        scaler,z=standardize_fit(x,np.array([1.,2.,1.]))
        np.testing.assert_allclose(scaler.mean_,[2,3,5])
        np.testing.assert_allclose(scaler.var_,[2,2,0])
        self.assertEqual(scaler.scale_[2],1.)
        before=scaler.mean_.copy(); query=scaler.transform(x+100)
        np.testing.assert_array_equal(before,scaler.mean_)
        self.assertEqual(query[0,2],100.)
        with self.assertRaises(ValueError): standardize_fit(x,np.zeros(3))
        with self.assertRaises(ValueError): standardize_fit(x*np.nan,np.ones(3))

    def test_controls_shared_map_and_scale(self):
        rng=np.random.default_rng(8)
        d=dict(body=rng.normal(size=(50,8)),raw=rng.normal(size=(50,265)),truth=rng.normal(size=50))
        models,pred,checks,n=fit_heads(d,d,[str(i) for i in range(50)],['a']*25+['b']*25,9)
        self.assertEqual(n,0)
        self.assertIs(models['scaled_body']['scaler'],models['scaled_shuffle']['scaler'])
        self.assertIs(models['scaled_body']['mapping'],models['scaled_shuffle']['mapping'])
        self.assertTrue(np.all(models['scaled_process']['scaler'].var_[:106]==0))
        self.assertTrue(all(c['converged'] for c in checks.values()))
        for arm,m in models.items():
            source=d['body'] if arm in ('scaled_body','scaled_shuffle') else d['raw'].copy()
            if arm=='scaled_process': source[:,:106]=0
            p=m['model'].predict(map_query(m['scaler'].transform(source),m['mapping']))+m['offset']
            np.testing.assert_allclose(p,pred[arm])

    def test_patient_fold_equalization_and_identity_guard(self):
        frame=pd.DataFrame([dict(patient=p,fold=f,component=str(p//10),target='x',truth=0.,
                            **{a:float(p%3+1) for a in ALL_ARMS}) for p in range(40) for f in range(2)])
        r=assess(frame,9,50)
        self.assertEqual(r['unique_patients'],40)
        self.assertFalse(any(c['full_screen'] for c in r['candidates'].values()))
        with self.assertRaises(ValueError): assess(pd.concat([frame,frame.iloc[:1]]),9,50)


if __name__=='__main__': unittest.main()
