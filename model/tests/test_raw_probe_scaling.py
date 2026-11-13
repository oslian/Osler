import unittest
import warnings
import numpy as np
from scripts.audit_raw_probe_scaling import scale_exact_constants


class ScalingTests(unittest.TestCase):
    def test_exact_constants_no_warning_and_preserve_query_difference(self):
        x=np.c_[np.full(100,49.),np.arange(100.)];q=np.array([[50.,10.]])
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            a,b=scale_exact_constants(x,q,np.linspace(.5,2,100))
        np.testing.assert_array_equal(a[:,0],np.zeros(100));self.assertEqual(b[0,0],1.)

    def test_all_constant(self):
        a,b=scale_exact_constants(np.full((20,3),49.),np.full((2,3),49.),np.ones(20))
        self.assertFalse(a.any());self.assertFalse(b.any())
