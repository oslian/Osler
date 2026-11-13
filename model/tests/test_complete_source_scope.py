import unittest
from scripts.audit_complete_source_scope import exposure_row


class ExposureScopeTest(unittest.TestCase):
    def test_partial_and_pending_are_distinct(self):
        r = exposure_row(10000, 3752, 4288, True)
        self.assertEqual(r['remaining_unseen'], 5712)
        self.assertAlmostEqual(r['final_coverage'], .4288)
        self.assertEqual(exposure_row(359, 0, 0, False)['status'], 'not_trained_in_matched_trial')

    def test_impossible_counts_rejected(self):
        for args in [(5, 6, 6, True), (5, 4, 3, True), (5, 1, 1, False)]:
            with self.assertRaises(ValueError):
                exposure_row(*args)
