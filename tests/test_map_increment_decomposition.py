import unittest
import numpy as np
from scripts.audit_map_native_increment_decomposition import components


class DecompositionTests(unittest.TestCase):
    def test_ordered_contrasts_are_exact(self):
        c = components(np.array([0.,10.]),np.array([3.,8.]),
                       np.array([1.,7.]),np.array([0.,9.]))
        np.testing.assert_allclose(c['total'],[-3.,-1.])
        np.testing.assert_allclose(c['support'],[-2.,1.])
        np.testing.assert_allclose(c['numerical'],[-1.,-2.])
        np.testing.assert_allclose(c['total'],c['support']+c['numerical'])
        np.testing.assert_allclose(c['numerical_minus_support'],[1.,-3.])


if __name__ == '__main__':
    unittest.main()
