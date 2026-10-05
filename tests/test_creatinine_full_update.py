import unittest
from scripts.audit_creatinine_full_update import summarize


class FullUpdateSummaryTests(unittest.TestCase):
    def test_paired_summary_and_support(self):
        rows=[dict(delta=-1.,before_error=2.,after_error=1.,final_error=1.,last_variables=['bun'],gap=0.) for _ in range(30)]
        r=summarize(rows,['bun','map'])
        self.assertEqual(r['paired_change']['mean'],-1.)
        self.assertEqual(r['paired_change']['upper'],-1.)
        self.assertEqual(r['by_last_variable']['bun']['absent']['status'],'insufficient_support')
        self.assertEqual(r['by_last_variable']['map']['absent']['patients'],30)
        self.assertEqual(r['positive_final_gap'],0)
