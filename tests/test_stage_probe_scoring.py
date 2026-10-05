import unittest
import pandas as pd
from scripts.score_reconstruction_stage_probe import ARMS, assess


class StageScoringTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for p,f,v in [('a',0,1.),('a',1,3.),('b',0,4.)]:
            rows.append(dict(patient=p,component='one',target='x',fold=f,truth=0.,**{a:v for a in ARMS}))
        return pd.DataFrame(rows)

    def test_average_losses_per_patient(self):
        r=assess(self.frame(),1,repetitions=50)
        self.assertEqual(r['unique_patients'],2)
        self.assertEqual(r['normalized_mae']['patient']['hgb_body'],3.)
        self.assertFalse(any(c['screen_pass'] for c in r['comparisons'].values()))

    def test_duplicate_and_nonfinite_fail(self):
        f=self.frame()
        with self.assertRaises(ValueError): assess(pd.concat([f,f.iloc[:1]]),1,50)
        f.loc[0,'hgb_body']=float('nan')
        with self.assertRaises(ValueError): assess(f,1,50)
