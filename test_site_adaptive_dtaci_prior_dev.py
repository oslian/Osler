import unittest

from site_adaptive_dtaci_prior_dev import CANDIDATES, choose, label, robust_count


def cells_with(counts):
    """counts: {database: {theta: number of robust cells}} -> synthetic cell list."""
    cells = []
    for database, by_theta in counts.items():
        for i in range(max(by_theta.values())):
            cells.append({"databases": {database: {label(t): {"interval_robust": i < by_theta.get(t, 0)} for t in CANDIDATES}}})
    return cells


class PriorRuleTests(unittest.TestCase):
    def test_choose_maximizes_robust_count_and_breaks_ties_low(self):
        cells = cells_with({"a": {0.0: 1, 0.1: 3, 0.2: 3}, "b": {0.0: 2, 0.1: 2, 0.2: 2}})
        self.assertEqual(choose(cells, ["a", "b"]), 0.1)
        self.assertEqual(choose(cells, ["b"]), 0.0)

    def test_robust_count_skips_missing_databases(self):
        cells = cells_with({"a": {0.0: 2}})
        self.assertEqual(robust_count(cells, "a", 0.0), 2)
        self.assertEqual(robust_count(cells, "b", 0.0), 0)


if __name__ == "__main__":
    unittest.main()
