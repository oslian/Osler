import unittest
import numpy as np
from scripts.diagnose_kernel_metric import decompose
from scripts.train_kernel_probe import fit_map


class KernelMetricTests(unittest.TestCase):
    def test_age_only_cannot_be_counted_as_numeric_signal(self):
        x=np.zeros((40,265)); x[:,159]=np.arange(40); w=np.ones(40)
        mapping,_=fit_map(x,w,4); r=decompose(x,w,4,mapping['kernel'].gamma)
        self.assertEqual(r['numeric_fraction_median'],0.)
        self.assertEqual(r['median_fraction']['lag'],1.)
        self.assertEqual(r['numeric_removal_kernel_change_mean'],0.)

    def test_numeric_only_kernel_difference_is_nonzero(self):
        x=np.zeros((40,265)); x[:,0]=np.arange(40); w=np.ones(40)
        mapping,_=fit_map(x,w,4); r=decompose(x,w,4,mapping['kernel'].gamma)
        self.assertEqual(r['numeric_fraction_median'],1.)
        self.assertEqual(r['median_fraction']['lag'],0.)
        self.assertGreater(r['numeric_removal_kernel_change_mean'],0.)

    def test_invalid_gamma_and_empty_distance(self):
        x=np.zeros((4,265)); r=decompose(x,np.ones(4),2,1.)
        self.assertEqual(r['nonzero_pairs'],0)
        self.assertIsNone(r['numeric_fraction_median'])
        with self.assertRaises(ValueError): decompose(x,np.ones(4),2,2.)


if __name__=='__main__': unittest.main()
