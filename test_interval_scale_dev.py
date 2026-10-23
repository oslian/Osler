import unittest

from interval_scale_dev import choose


def cells_with(counts):
    cells = []
    for database, by_scheme in counts.items():
        for i in range(max(by_scheme.values())):
            cells.append({"databases": {database: {s: {"interval_robust": i < by_scheme.get(s, 0)}
                                                    for s in ("quintile", "tail8", "tail8_change")}}})
    return cells


class SchemeRuleTests(unittest.TestCase):
    def test_most_robust_wins_and_ties_go_to_the_simpler_scheme(self):
        cells = cells_with({"a": {"quintile": 2, "tail8": 3, "tail8_change": 3}})
        self.assertEqual(choose(cells, ["a"]), "tail8")
        cells = cells_with({"a": {"quintile": 3, "tail8": 3, "tail8_change": 3}})
        self.assertEqual(choose(cells, ["a"]), "quintile")


if __name__ == "__main__":
    unittest.main()
