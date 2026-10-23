import unittest

from recipe_shift_selection_dev import recipes_for, select


def outcome(*passes):
    return {name: {"full_pass": value} for name, value in zip(("S1_time", "S2_cardiac_units"), passes)}


class RecipeSelectionTests(unittest.TestCase):
    def test_simplest_recipe_passing_both_scenarios_wins(self):
        recipes = ("measurement_pure_ridge", "plain_ridge", "target_specific_change_hurdle")
        results = {
            "measurement_pure_ridge": outcome(True, True),
            "plain_ridge": outcome(True, True),
            "target_specific_change_hurdle": outcome(True, True),
        }
        self.assertEqual(select(results, recipes), "measurement_pure_ridge")

    def test_complex_recipe_wins_only_when_simpler_ones_fail(self):
        recipes = ("measurement_pure_ridge", "plain_ridge", "target_specific_change_hurdle")
        results = {
            "measurement_pure_ridge": outcome(True, False),
            "plain_ridge": outcome(False, False),
            "target_specific_change_hurdle": outcome(True, True),
        }
        self.assertEqual(select(results, recipes), "target_specific_change_hurdle")

    def test_hurdle_offered_only_with_a_change_threshold(self):
        self.assertNotIn("target_specific_change_hurdle", recipes_for("heart_rate"))
        self.assertIn("target_specific_change_hurdle", recipes_for("creatinine"))


if __name__ == "__main__":
    unittest.main()
