import unittest
import numpy as np
from scripts.train_joint_source_reconstruction import equal_group_median


class JointSourceTest(unittest.TestCase):
    def test_unequal_group_size_does_not_set_prior_weight(self):
        a=np.array([0.,2.]);b=np.array([10.]*100)
        self.assertEqual(equal_group_median([a,b]),2.)

    def test_same_groups(self):
        self.assertEqual(equal_group_median([np.array([1.,3.,5.])]*2),3.)
