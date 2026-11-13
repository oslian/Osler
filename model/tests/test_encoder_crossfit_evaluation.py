import unittest
import numpy as np
import pandas as pd
from scripts.evaluate_encoder_crossfit import summarize


class EvaluationTests(unittest.TestCase):
    def test_abstention_remains_in_coverage_not_error_denominator(self):
        f=pd.DataFrame(dict(patient=['a','b','c'],site=['x','x','y'],component=['x','x','y'],
            truth=[2.,2.,2.],real=[2.,3.,np.nan],process=[4.,4.,np.nan],population=[3.,3.,3.],status=['ok','ok','empty']))
        r=summarize(f,19)
        self.assertEqual(r['total'],3);self.assertEqual(r['eligible'],2)
        self.assertEqual(r['abstentions'],{'empty':1})
        self.assertEqual(r['mae']['real']['patient'],.5)
        self.assertFalse(r['descriptive_screen_pass'])

    def test_duplicate_patient_target_rejected(self):
        with self.assertRaisesRegex(ValueError,'appears twice'):
            summarize(pd.DataFrame({'patient':['a','a']}),19)
