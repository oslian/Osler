import unittest
import numpy as np
import pandas as pd
from scripts.audit_crossfit_group_shift import decompose,group_stats


class GroupShiftTests(unittest.TestCase):
    def test_same_means_different_weights_is_composition(self):
        a=pd.DataFrame({'mean':[-1.,1.],'count':[90,10]},index=['a','b'])
        b=pd.DataFrame({'mean':[-1.,1.],'count':[10,90]},index=['a','b'])
        r=decompose(a,b)
        self.assertEqual(r['within_component_term'],0.)
        self.assertAlmostEqual(r['composition_term'],-1.6)

    def test_equal_weights_is_within_difference(self):
        a=pd.DataFrame({'mean':[0.,2.],'count':[10,10]},index=['a','b'])
        b=pd.DataFrame({'mean':[-1.,1.],'count':[10,10]},index=['a','b'])
        r=decompose(a,b)
        self.assertEqual(r['composition_term'],0.)
        self.assertEqual(r['within_component_term'],1.)

    def test_small_harmful_component_not_removed(self):
        f=pd.DataFrame({'component':['a']*100+['b'],'delta':[-.1]*100+[1.]})
        _,r=group_stats(f)
        self.assertLess(r['patient_mean'],0.)
        self.assertGreater(r['equal_component_mean'],0.)
        self.assertEqual(r['patients'],101)
        self.assertEqual(r['positive_component_mass_from_under30'],1.)
