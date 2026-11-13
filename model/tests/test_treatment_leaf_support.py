import unittest
from threadpoolctl import threadpool_limits
from scripts.audit_treatment_leaf_support import synthetic_probe
import numpy as np
from whole_body_jepa.potassium_treatment_context import action_support_status


class LeafSupportTests(unittest.TestCase):
    def test_empty_evaluation_is_not_negative_evidence(self):
        subjects=np.array(['a','a','b','c'])
        exposed=np.array([True,True,False,False])
        partitions=[np.array([1,1,0,0],bool),np.array([0,0,1,0],bool),np.array([0,0,0,1],bool)]
        result=action_support_status(subjects,exposed,partitions)
        self.assertEqual(result['fit']['patients'],1)
        self.assertEqual(result['fit']['rows'],2)
        self.assertFalse(result['pure_exposure_split_supported'])
        self.assertFalse(result['evaluation_has_exposure'])

    def test_perfect_rare_signal_blocked_by_leaf_minimum(self):
        with threadpool_limits(limits=1):
            blocked=synthetic_probe(30)
            available=synthetic_probe(10)
        self.assertEqual(blocked['contrast'],0)
        self.assertGreater(available['contrast'],.99)
        self.assertLess(available['mae'],1e-5)
