import unittest
import numpy as np
from scipy.stats import ortho_group
from threadpoolctl import threadpool_limits

from scripts.train_kernel_probe import fit_map,map_query,fit_regressor,rotation_check,fit_heads


class KernelProbeTests(unittest.TestCase):
    def test_rotation_invariant_kernel_and_predictions(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(160,8)); q=rng.normal(size=(40,8)); w=np.ones(160)
        with threadpool_limits(limits=2):
            mapping,z=fit_map(x,w,9); model,offset,diag=fit_regressor(z,x[:,0]**2,w)
            check=rotation_check(q,map_query(q,mapping),mapping,model,offset,8)
            self.assertLess(check['output_error'],1e-7)
            self.assertTrue(diag['converged'])
            rotation=ortho_group.rvs(8,random_state=8)
            changed,_=fit_map(x@rotation,w,9)
            self.assertAlmostEqual(mapping['kernel'].gamma,changed['kernel'].gamma,places=12)
            np.testing.assert_array_equal(mapping['kernel'].component_indices_,changed['kernel'].component_indices_)

    def test_query_does_not_change_fit_map(self):
        x=np.random.default_rng(2).normal(size=(30,4))
        mapping,_=fit_map(x,np.ones(30),1)
        before=mapping['kernel'].components_.copy(); center=mapping['center'].copy()
        map_query(x+100,mapping)
        np.testing.assert_array_equal(before,mapping['kernel'].components_)
        np.testing.assert_array_equal(center,mapping['center'])

    def test_degenerate_inputs_and_invalid_weights(self):
        m,z=fit_map(np.zeros((10,4)),np.ones(10),1)
        self.assertEqual(m['kernel'].gamma,1.)
        self.assertTrue(np.isfinite(z).all())
        with self.assertRaises(ValueError): fit_map(np.zeros((10,4)),np.zeros(10),1)
        with self.assertRaises(ValueError): map_query(np.ones((3,2)),m)

    def test_all_controls_use_same_width_and_real_labels(self):
        rng=np.random.default_rng(2)
        data=dict(body=rng.normal(size=(100,8)),raw=rng.normal(size=(100,110)),truth=rng.normal(size=100))
        models,pred,checks,n=fit_heads(data,data,[str(i) for i in range(100)],['a']*50+['b']*50,2)
        self.assertEqual(n,0); self.assertEqual(len(models),4)
        self.assertIs(models['kernel_body']['mapping'],models['kernel_shuffle']['mapping'])
        self.assertTrue(all(c['feature_width']==100 for c in checks.values()))
        self.assertTrue(all(c['converged'] for c in checks.values()))
        self.assertTrue(all(np.isfinite(p).all() for p in pred.values()))


if __name__=='__main__': unittest.main()
