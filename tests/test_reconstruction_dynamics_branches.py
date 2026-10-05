import unittest
from scripts.audit_reconstruction_dynamics_branches import summarize


class BranchSummaryTests(unittest.TestCase):
    def test_groups_targets_not_rows(self):
        rows = [dict(patient=1,target='a',errors=dict(full=2.,no_body=1.,no_organs=3.)),
                dict(patient=1,target='b',errors=dict(full=1.,no_body=3.,no_organs=2.))]
        r = summarize(rows,['a','b'])
        self.assertEqual(r['a']['versus_full']['no_body']['mean'],-1.)
        self.assertEqual(r['b']['versus_full']['no_body']['mean'],2.)

    def test_missing_or_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            summarize([],['a'])
        with self.assertRaises(ValueError):
            summarize([dict(patient=1,target='a'),dict(patient=1,target='a')],['a'])
