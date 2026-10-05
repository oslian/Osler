import unittest
from scripts.audit_broad_prior_increment import decompose


class DecompositionTest(unittest.TestCase):
    def test_only_prior_changes(self):
        r=decompose([2,4],[0,0],[2,2],[0,0],[2,2])
        self.assertEqual(r['total'],r['prior'])
        self.assertEqual(r['incremental_change'],0)

    def test_individual_improvement(self):
        r=decompose([2,4],[3,3],[2,4],[3,3],[3,3])
        self.assertEqual(r['prior'],0)
        self.assertEqual(r['incremental_change'],-1)

    def test_total_improves_while_increment_harms(self):
        r=decompose([2,4],[0,0],[1,1],[0,0],[3,3])
        self.assertLess(r['total'],0)
        self.assertGreater(r['incremental_change'],0)
