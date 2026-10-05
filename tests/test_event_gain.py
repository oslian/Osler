import unittest
import numpy as np
from scripts.audit_potassium_event_changes import fit_global_gain


class EventGainTests(unittest.TestCase):
    def test_amplification_and_sign(self):
        self.assertEqual(fit_global_gain([2,-4],[1,-2],['a','b']),2.)
        self.assertEqual(fit_global_gain([-2,4],[1,-2],['a','b']),0.)

    def test_patient_equal_and_zero(self):
        self.assertEqual(fit_global_gain([2]*10+[1,1],[1]*12,['a']*10+['b','c']),1.)
        self.assertEqual(fit_global_gain([3],[0],['a']),1.)
        with self.assertRaises(ValueError):fit_global_gain([np.nan],[1],['a'])
