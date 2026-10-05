import unittest
from extractors.mimiciv_observation_transition_extract import LAB_ITEMIDS,EXTRA_PLAUSIBLE
from whole_body_jepa.contracts import VARIABLE_UNITS,ALIASES


class AnalyteIdentityTest(unittest.TestCase):
    def test_ntprobnp_is_not_bnp(self):
        self.assertEqual(LAB_ITEMIDS['nt_pro_bnp'],(50963,))
        self.assertNotIn(50963,LAB_ITEMIDS.get('bnp',()))
        self.assertIn('nt_pro_bnp',EXTRA_PLAUSIBLE)
        self.assertEqual(VARIABLE_UNITS['nt_pro_bnp'],'pg/mL')
        self.assertNotEqual(ALIASES.get('nt_pro_bnp'),'bnp')


if __name__=='__main__':unittest.main()
