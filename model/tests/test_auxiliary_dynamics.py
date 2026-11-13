import unittest
import numpy as np
import pandas as pd
from scripts.audit_auxiliary_learning import patient_quantile, movement_metrics


class DynamicsTest(unittest.TestCase):
    def test_patient_weighted_center(self):
        values=[100]*100+[0,0]
        subjects=['long']*100+['b','c']
        self.assertEqual(patient_quantile(values,subjects,.5),0)

    def test_zero_movement_has_no_correlation(self):
        frame=pd.DataFrame({'subject_id':['a','a','b'],'truth':[1.,3.,4.]})
        current=np.array([1.,1.,1.])
        result=movement_metrics(frame,current,current)
        self.assertEqual(result['absolute_movement'],0)
        self.assertIsNone(result['delta_correlation'])
        self.assertEqual(result['signed_error'],-2)

    def test_perfect_deltas(self):
        frame=pd.DataFrame({'subject_id':['a','b','c'],'truth':[1.,3.,4.]})
        result=movement_metrics(frame,np.zeros(3),frame.truth.to_numpy())
        self.assertAlmostEqual(result['delta_correlation'],1)
        self.assertEqual(result['signed_error'],0)
