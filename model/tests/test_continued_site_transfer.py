import unittest

from scripts.audit_continued_site_transfer import site_summary


class SiteSummaryTests(unittest.TestCase):
    def test_weighting_reversal(self):
        result = site_summary([-1.] * 10 + [2., 2.], ['a'] * 10 + ['b', 'c'])
        self.assertTrue(result['weighting_sign_disagreement'])
        self.assertEqual(result['improving_sites'], 1)
        self.assertFalse(result['all_leave_one_site_out_improve'])

    def test_uniform_gain(self):
        result = site_summary([-1., -2., -3.], ['a', 'b', 'c'])
        self.assertTrue(result['all_leave_one_site_out_improve'])
        self.assertAlmostEqual(result['effective_sites'], 3.)

    def test_invalid_and_single_site(self):
        with self.assertRaises(ValueError):
            site_summary([float('nan')], ['a'])
        self.assertIsNone(site_summary([1.], ['a'])['leave_one_site_out_range'])
