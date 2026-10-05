import unittest
import numpy as np
from scripts.bounded_site_sampling import bounded_weights, sample_indices


class BoundedSamplingTests(unittest.TestCase):
    def test_small_site_does_not_dominate(self):
        w=bounded_weights(['large']*1000+['small'])
        self.assertLessEqual(w.max(),2.)
        self.assertGreaterEqual(w.min(),.5)
        self.assertAlmostEqual(w.mean(),1.)
        self.assertEqual(w[-1],2.)

    def test_balanced_sites_and_single_site(self):
        for sites in (['a','a','b','b'],['a']*7):
            np.testing.assert_allclose(bounded_weights(sites),1.)

    def test_permutation(self):
        sites=np.array(['a']*19+['b']*3+['c'])
        order=np.random.default_rng(4).permutation(len(sites))
        np.testing.assert_allclose(bounded_weights(sites)[order],bounded_weights(sites[order]))

    def test_rng_reproducibility(self):
        w=bounded_weights(['a']*10+['b'])
        a=sample_indices(w,np.random.default_rng(7),100)
        b=sample_indices(w,np.random.default_rng(7),100)
        np.testing.assert_array_equal(a,b)

    def test_invalid(self):
        with self.assertRaises(ValueError):bounded_weights([])
        with self.assertRaises(ValueError):sample_indices([np.nan],np.random.default_rng(1),2)
