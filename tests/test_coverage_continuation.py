import unittest
from scripts.continue_source_coverage import required_epochs


class CoverageBudgetTest(unittest.TestCase):
    def test_dense_pool_sets_budget(self):
        self.assertEqual(required_epochs({'dense':range(10096),'sparse':range(90)}),19)
        self.assertEqual(required_epochs({'exact':range(1072)}),2)

    def test_invalid_pool_rejected(self):
        for pools in ({},{'empty':[]}):
            with self.assertRaises(ValueError): required_epochs(pools)
