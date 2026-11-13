import unittest
from scripts.summarize_kernel_ray_gap import gap_summary


class KernelRayGapTests(unittest.TestCase):
    def test_recovery_excludes_nonpositive_gap(self):
        r=gap_summary([2,4,1],[1.5,3,1],[1,2,1.5])
        self.assertEqual(r['current_harms'],2)
        self.assertEqual(r['oracle_ray_harms'],2)
        self.assertEqual(r['positive_gap_combinations'],2)
        self.assertEqual(r['median_gap_fraction_recovered'],.5)

    def test_invalid_and_no_gap(self):
        self.assertIsNone(gap_summary([1],[.9],[2])['median_gap_fraction_recovered'])
        with self.assertRaises(ValueError): gap_summary([1],[1.1],[1])
        with self.assertRaises(ValueError): gap_summary([1],[0],[-1])


if __name__=='__main__': unittest.main()
