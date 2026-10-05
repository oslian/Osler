import unittest
from extractors.eicu_body_system_configs import get_body_system_config
from extractors.eicu_body_system_transition_extract import classify_action


class PotassiumBrandTests(unittest.TestCase):
    def test_observed_brand_spellings(self):
        config = get_body_system_config('electrolyte_acid_base')
        for name in ('KLOR-CON','K-Dur 20','K-DUR','KLOR-CON,K-TAB','potassium chloride','KCl'):
            with self.subTest(name=name):
                self.assertIn('potassium_repletion', classify_action(config,name))

    def test_unrelated_names(self):
        config = get_body_system_config('electrolyte_acid_base')
        for name in ('insulin','furosemide','sodium chloride',''):
            self.assertNotIn('potassium_repletion', classify_action(config,name))
