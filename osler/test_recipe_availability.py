import unittest

import numpy as np
import pandas as pd

from osler_jepa.recipe_availability import deployable_recipe, treatment_context_available


class RecipeAvailabilityTests(unittest.TestCase):
    def test_treatment_recipe_falls_back_without_treatment_records(self):
        site = pd.DataFrame({"glucose_t": [100.0, 120.0]})
        self.assertEqual(deployable_recipe("observed_treatment_context_ridge", site, "glucose")[0], "measurement_pure_ridge")

    def test_treatment_recipe_kept_when_permitted_treatment_is_recorded(self):
        site = pd.DataFrame({"glucose_t": [100.0, 120.0], "hist_insulin": [1.0, np.nan]})
        self.assertTrue(treatment_context_available(site, "glucose"))
        self.assertEqual(deployable_recipe("observed_treatment_context_ridge", site, "glucose"),
                         ("observed_treatment_context_ridge", "catalog"))

    def test_unrelated_treatment_columns_do_not_count(self):
        site = pd.DataFrame({"glucose_t": [100.0], "hist_vasopressor": [1.0]})
        self.assertFalse(treatment_context_available(site, "glucose"))

    def test_other_recipes_are_never_replaced(self):
        site = pd.DataFrame({"creatinine_t": [1.0]})
        for recipe in ("target_specific_change_hurdle", "plain_ridge", "measurement_pure_ridge"):
            self.assertEqual(deployable_recipe(recipe, site, "creatinine")[0], recipe)


if __name__ == "__main__":
    unittest.main()
