import unittest

from external_recipe_selection_dev import select

MPR, TCR, PLAIN, HURDLE = ("measurement_pure_ridge", "observed_treatment_context_ridge", "plain_ridge",
                           "target_specific_change_hurdle")


def summary(point, persistence):
    return {"point_robust": point, "persistence_robust": persistence}


class SelectionRuleTests(unittest.TestCase):
    def test_most_robust_databases_wins(self):
        candidates = {MPR: {"a": summary(True, True), "b": summary(False, True)},
                      HURDLE: {"a": summary(True, True), "b": summary(True, True)}}
        result = select(candidates, ["a", "b"])
        self.assertEqual(result["selected"], HURDLE)
        self.assertFalse(result["persistence_fallback"])

    def test_ties_go_to_simplest(self):
        candidates = {HURDLE: {"a": summary(True, True)}, PLAIN: {"a": summary(True, True)}, MPR: {"a": summary(True, True)}}
        self.assertEqual(select(candidates, ["a"])["selected"], MPR)

    def test_fallback_without_strict_majority(self):
        candidates = {MPR: {"a": summary(False, True), "b": summary(False, False)}}
        self.assertTrue(select(candidates, ["a", "b"])["persistence_fallback"])
        candidates = {MPR: {"a": summary(False, True), "b": summary(False, True), "c": summary(False, False)}}
        self.assertFalse(select(candidates, ["a", "b", "c"])["persistence_fallback"])


if __name__ == "__main__":
    unittest.main()
