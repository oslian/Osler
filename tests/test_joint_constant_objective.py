import unittest
import numpy as np
from scripts.audit_joint_constant_objective import huber_center


class ConstantObjectiveTest(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(huber_center([np.array([3.,3.])]),3.)

    def test_symmetric(self):
        self.assertAlmostEqual(huber_center([np.array([-2.,0.,2.])]),0.)

    def test_balanced_groups_not_row_counts(self):
        x=huber_center([np.array([0.]),np.full(100,.5)])
        self.assertAlmostEqual(x,.25)
