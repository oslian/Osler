import unittest
from scripts.audit_bun_full_history_coverage import summarize


class FullHistorySummaryTests(unittest.TestCase):
    def test_patient_paired_error_difference(self):
        rows = [dict(patient=i, errors=dict(full=2., latest_each=1., last8_latest_each=3.),
                     events=dict(full=10, latest_each=3, last8_latest_each=8)) for i in range(3)]
        result = summarize(rows)
        self.assertEqual(result['minus_full']['latest_each']['mean'], -1.)
        self.assertEqual(result['minus_full']['last8_latest_each']['mean'], 1.)
        self.assertEqual(result['mean_events']['latest_each'], 3.)

    def test_duplicate_or_empty_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])
        with self.assertRaises(ValueError):
            summarize([dict(patient=1), dict(patient=1)])
