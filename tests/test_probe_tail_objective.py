import unittest
import numpy as np
from scripts.audit_probe_tail_objective import weighted_median,source_statistics


class TailTests(unittest.TestCase):
    def test_weighted_median(self):
        self.assertEqual(weighted_median(np.array([10.,0.,2.]),np.array([.1,.8,.1])),0.)

    def test_tail_and_constant_case(self):
        y=np.r_[np.zeros(100),1000.];r=source_statistics(y,np.ones(len(y)))
        self.assertEqual(r['median'],0.)
        self.assertGreater(r['tail_squared_loss_fraction'],.9)
        self.assertEqual(source_statistics(np.ones(10),np.ones(10))['tail_squared_loss_fraction'],0.)
