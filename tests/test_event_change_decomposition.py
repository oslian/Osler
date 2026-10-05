import unittest
import numpy as np
import pandas as pd
from scripts.audit_potassium_event_changes import decomposition,directions,large_change_mask


class ChangeDecompositionTest(unittest.TestCase):
    def test_quantized_boundary_is_not_rounding_dependent(self):
        delta=np.array([.5,.6,.6000000000000005,.7])
        for threshold in (.5999999999999996,.6000000000000005):
            np.testing.assert_array_equal(large_change_mask(delta,threshold),[False,False,False,True])

    def test_patient_weighted_components_sum(self):
        f=pd.DataFrame({'subject_id':['a','a','b'],'truth':[1.,1.,1.]})
        d=decomposition(f,np.zeros(3),np.array([1.,1.,3.]),np.zeros(3),np.array([True,False,False]))
        self.assertAlmostEqual(d['pooled'],0.)
        self.assertAlmostEqual(d['large'],-.25)
        self.assertAlmostEqual(d['smaller'],.25)

    def test_direction_does_not_count_zero_truth_as_correct(self):
        f=pd.DataFrame({'subject_id':['a','b'],'truth':[0.,1.]})
        self.assertEqual(directions(f,np.zeros(2),np.zeros(2)),0.)
        self.assertEqual(directions(f,np.zeros(2),np.array([99.,1.])),1.)


if __name__=='__main__':unittest.main()
