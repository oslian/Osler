import unittest
import numpy as np
import pandas as pd
from scripts.audit_reconstruction_raw_blocks import ARMS, block_inputs, assess, verify_reference


class RawBlocksTest(unittest.TestCase):
    def test_only_requested_blocks_removed(self):
        raw = np.arange(530,dtype=float).reshape(2,265)
        original = raw.copy()
        arms = block_inputs(raw)
        for name, removed in (('latest_process',slice(53,106)),('mean_process',slice(0,53)),
                              ('values_presence',slice(159,265))):
            expected = raw.copy(); expected[:,removed]=0
            np.testing.assert_array_equal(arms[name],expected)
            np.testing.assert_array_equal(arms[name][:,106:159],raw[:,106:159])
        arms['raw'][0,0] = -999
        np.testing.assert_array_equal(raw,original)

    def test_invalid_inputs_rejected(self):
        for x in (np.zeros((0,265)),np.zeros((1,264)),np.full((1,265),np.nan)):
            with self.assertRaises(ValueError): block_inputs(x)

    def frame(self):
        f = pd.DataFrame(dict(patient=['a','a','b'],fold=[0,1,0],component=['x','x','y'],
                             target=['bun']*3,truth=[0.]*3))
        for arm in ARMS: f[arm]=[1.,3.,6.]
        return f

    def test_repeated_folds_patient_equalized(self):
        r = assess(self.frame(),repetitions=20)
        self.assertEqual(r['patients'],2)
        self.assertEqual(r['normalized_mae']['patient']['raw'],4.)
        self.assertTrue(all(not x['screen_pass'] for x in r['comparisons'].values()))

    def test_invalid_pairing_rejected(self):
        for kind in ('duplicate','component','nonfinite'):
            f = self.frame()
            if kind=='duplicate': f.loc[1,'fold']=0
            if kind=='component': f.loc[1,'component']='z'
            if kind=='nonfinite': f.loc[0,'raw']=np.inf
            with self.assertRaises(ValueError): assess(f,repetitions=20)

    def test_reference_truth_and_order_checked(self):
        f = pd.DataFrame(dict(patient=['a','b'],component=['x','y'],target=['bun']*2,fold=[0]*2,
                             truth=[1.,2.],hgb_raw=[1.,2.],hgb_body=[0.,1.],hgb_process=[2.,3.]))
        args = (['a','b'],np.array([1.,2.]),dict(a='x',b='y'),'bun',0)
        verify_reference(f,*args)
        for bad in (f.iloc[::-1], f.assign(truth=[2.,1.]), f.assign(fold=1)):
            with self.assertRaises(ValueError): verify_reference(bad,*args)


if __name__ == '__main__': unittest.main()
