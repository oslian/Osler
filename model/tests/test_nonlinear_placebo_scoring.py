import unittest

import pandas as pd

from scripts.score_nonlinear_probe import ARMS
from scripts.score_nonlinear_placebo import assess, verify_rows


def example():
    f=pd.DataFrame(dict(patient=['a','b'],target=['bun']*2,component=['x','y'],
        fold=[0,0],role=['selection']*2,truth=[0.,0.],trained_shuffle=[2.,4.]))
    for arm in ARMS: f[arm]=[1.,3.]
    return f


class PlaceboScoringTests(unittest.TestCase):
    def test_repeated_source_people_count_once(self):
        a=example(); b=example(); b['fold']=1; b.loc[0,'body_probe']=3
        r=assess(pd.concat([a,b]),19,repetitions=50)
        self.assertEqual(r['query_occurrences'],4)
        self.assertEqual(r['unique_patients'],2)
        self.assertEqual(r['normalized_mae']['patient']['body_probe'],2.5)
        self.assertAlmostEqual(r['body_minus_trained_shuffle']['patient']['mean'],-.5)
        self.assertFalse(r['paired_47_screen'])

    def test_changed_original_predictions_rejected(self):
        f=example(); verify_rows(f,f.iloc[::-1],'selection',0)
        old=f.copy(); f.loc[0,'body_probe']+=1
        with self.assertRaises(ValueError): verify_rows(f,old,'selection',0)
        with self.assertRaises(ValueError): verify_rows(old.iloc[:1],old,'selection',0)

    def test_duplicate_and_nonfinite_rejected(self):
        f=example()
        with self.assertRaises(ValueError): assess(pd.concat([f,f]),19,50)
        f.loc[0,'trained_shuffle']=float('nan')
        with self.assertRaises(ValueError): assess(f,19,50)


if __name__=='__main__': unittest.main()
