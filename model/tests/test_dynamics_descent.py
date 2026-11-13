import unittest
import numpy as np
from scripts.audit_dynamics_descent import protected_direction


class ProtectedDirectionTests(unittest.TestCase):
    def test_removes_conflicting_component(self):
        d,s=protected_direction([1.,-1.],[[0.,1.]])
        self.assertTrue(s['success']);np.testing.assert_allclose(d,[1.,0.],atol=1e-6)

    def test_opposition_can_remove_all_progress(self):
        d,s=protected_direction([1.,0.],[[-1.,0.],[1.,0.]])
        self.assertTrue(s['success']);np.testing.assert_allclose(d,[0.,0.],atol=1e-6)

    def test_zero_constraint_is_ignored(self):
        d,s=protected_direction([1.,2.],[[0.,0.]])
        self.assertTrue(s['success']);np.testing.assert_allclose(d,[1.,2.])
