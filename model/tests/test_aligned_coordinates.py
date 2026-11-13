import unittest
import numpy as np
import pandas as pd
from scripts.train_aligned_coordinates import fit_axes, transform, retention, fit_heads, ARMS
from scripts.score_aligned_coordinates import assess, ALL


class AlignmentTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(5)
        b=rng.normal(size=(100,64));r=rng.normal(size=(100,265))
        r[:,0]=b[:,0]*2;r[:,3]=49;r[:,106:159]=1
        people=[str(i) for i in range(100)];groups=['a']*50+['b']*50
        return b,r,np.ones(100),people,groups

    def test_lossless_body_and_equal_width(self):
        b,r,w,p,g=self.fixture();m=fit_axes(b,r,w,p,g,1)
        for a in ARMS:
            z=transform(b,m,a)
            self.assertEqual(z.shape,(100,329))
            np.testing.assert_array_equal(z[:,:64],b)
            self.assertTrue(np.isfinite(z).all())
        self.assertEqual(m['unchanged_singletons'],0)

    def test_constant_outputs_zero(self):
        b,r,w,p,g=self.fixture();m=fit_axes(b,r,w,p,g,2)
        for a in ('aligned_body','shuffled_axes'):
            z=transform(b+10000,m,a)
            np.testing.assert_array_equal(z[:,64+3],np.zeros(100))

    def test_no_query_raw_or_labels_needed(self):
        b,r,w,p,g=self.fixture();m=fit_axes(b,r,w,p,g,3)
        before=m['aligned_body'].coef_.copy()
        q=b[:8]+10
        np.testing.assert_array_equal(transform(q,m,'aligned_body'),transform(q,m,'aligned_body'))
        np.testing.assert_array_equal(m['aligned_body'].coef_,before)
        again=fit_axes(b,r,w,p,g,3)
        for a in ARMS: np.testing.assert_array_equal(transform(q,m,a),transform(q,again,a))

    def test_observed_retention_does_not_score_missing_zero(self):
        b,r,w,p,g=self.fixture();r[:60,106]=0;r[:60,0]=0
        m=fit_axes(b,r,w,p,g,4);decoded=m['raw_scaler'].transform(r)
        first=retention(r,decoded,m)
        decoded[:60,0]=1e6
        second=retention(r,decoded,m)
        self.assertEqual(first['latest'],second['latest'])

    def test_invalid_fit_rejected(self):
        b,r,w,p,g=self.fixture();w[0]=0
        with self.assertRaises(ValueError):fit_axes(b,r,w,p,g,1)
        with self.assertRaises(ValueError):fit_axes(b,np.full_like(r,np.nan),np.ones(100),p,g,1)

    def test_fixed_training_and_query_truth_independence(self):
        b,r,w,p,g=self.fixture();f=dict(body=b,raw=r,truth=r[:,0])
        q=dict(body=b[:10],raw=r[:10],truth=np.ones(10)*999)
        models,pred,checks,_=fit_heads(f,q,w,p,g,4)
        for a in ARMS:
            self.assertEqual(models['models'][a].n_iter_,100)
            self.assertEqual(checks[a]['width'],329)
            self.assertTrue(np.isfinite(pred[a]).all())

    def test_paired_patient_weight_and_duplicate_guard(self):
        f=pd.DataFrame(dict(patient=['a','a','b'],component=['c','c','d'],target=['x']*3,
            fold=[0,1,0],truth=[0.]*3,**{a:[2.,2.,0.] for a in ALL}))
        result=assess(f,4,10)
        self.assertEqual(result['normalized_mae']['patient']['aligned_body'],1.)
        self.assertFalse(result['full_screen'])
        with self.assertRaises(ValueError):assess(pd.concat([f,f.iloc[:1]]),4,10)


if __name__=='__main__':unittest.main()
