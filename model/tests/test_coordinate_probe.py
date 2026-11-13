import unittest
import numpy as np
import pandas as pd

from scripts.train_coordinate_probe import coordinates, transform, fit_heads
from scripts.score_coordinate_probe import ALL_ARMS, assess


class CoordinateTests(unittest.TestCase):
    def test_full_basis_preserves_rank_deficient_inputs(self):
        x = np.array([[1.,0.,3.,0.],[2.,0.,6.,0.],[4.,0.,12.,0.]])
        for mapping in coordinates(x, np.ones(3), 4).values():
            z,e = transform(x, mapping)
            self.assertEqual(z.shape, x.shape)
            self.assertLess(e, 1e-12)
            np.testing.assert_allclose(z@mapping['basis'].T+mapping['center'],x,atol=1e-12)

    def test_query_cannot_change_fitted_transform(self):
        x = np.random.default_rng(1).normal(size=(10,4))
        maps = coordinates(x, np.ones(10), 8)
        before = {k:{c:v.copy() for c,v in m.items()} for k,m in maps.items()}
        for k,m in maps.items():
            transform(x+10000,m)
            for c in m: np.testing.assert_array_equal(m[c],before[k][c])

    def test_rotations_repeat_and_preserve_distances(self):
        x = np.random.default_rng(2).normal(size=(20,8))
        maps = coordinates(x, np.arange(1,21), 9)
        again = coordinates(x, np.arange(1,21), 9)
        for k,m in maps.items():
            z,_ = transform(x,m)
            np.testing.assert_allclose(np.linalg.norm(z-z[0],axis=1),np.linalg.norm(x-x[0],axis=1),atol=1e-12)
            np.testing.assert_array_equal(m['basis'],again[k]['basis'])

    def test_invalid_and_lossy_transforms_rejected(self):
        with self.assertRaises(ValueError): coordinates(np.zeros((3,4)),[1,0,1],8)
        with self.assertRaises(ValueError): transform(np.ones((3,4)),dict(center=np.zeros(4),basis=np.eye(4)*.5))
        with self.assertRaises(ValueError): transform(np.full((3,4),np.nan),dict(center=np.zeros(4),basis=np.eye(4)))

    def test_fixed_budget_and_width(self):
        rng = np.random.default_rng(4)
        f = dict(body=rng.normal(size=(80,4)),raw=rng.normal(size=(80,6)),truth=rng.normal(size=80))
        models,pred,checks = fit_heads(f,f,np.ones(80),8)
        self.assertEqual(len(models),4)
        for a,m in models.items():
            self.assertEqual(m['model'].n_iter_,100)
            self.assertEqual(m['model'].n_features_in_,4 if a.startswith('body') else 6)
            self.assertTrue(np.isfinite(pred[a]).all())
            self.assertLess(checks[a]['query_inverse_error'],1e-12)

    def test_patient_equalization_and_duplicate_guard(self):
        f = pd.DataFrame(dict(patient=['a','a','b'],component=['c','c','d'],target=['x']*3,
            fold=[0,1,0],truth=[0.]*3,**{a:[2.,2.,0.] for a in ALL_ARMS}))
        r = assess(f, 4, repetitions=10)
        self.assertEqual(r['normalized_mae']['patient']['body_pca'],1.)
        self.assertFalse(r['full_screen'])
        with self.assertRaises(ValueError): assess(pd.concat([f,f.iloc[:1]]),4,10)


if __name__ == '__main__': unittest.main()
