import unittest

import zigong_observation_transition_extract as zigong
from zigong_observation_transition_extract import ICU_DEPARTMENT, map_lab_item


class ZigongMappingTests(unittest.TestCase):
    def test_glucose_sources_follow_the_harmonized_definition(self):
        self.assertEqual(map_lab_item("Biochemistry 2", "Glucose (Glu)", "mmol/L")[0], "glucose")
        self.assertEqual(map_lab_item("blood sugar", "Glucose (Glu)", "mmol/L")[0], "glucose")
        self.assertEqual(map_lab_item("Blood gas analysis combination", "Glucose (Glu)", "mmol/L")[0], "glucose_bg")
        self.assertEqual(map_lab_item("EICU blood gas analysis", "Glucose concentration (cglu)", "mmol/L")[0], "glucose_bg")
        self.assertAlmostEqual(map_lab_item("Biochemistry 2", "Glucose (Glu)", "mmol/L")[1], 18.016)

    def test_unit_conversions_to_mimic_units(self):
        self.assertAlmostEqual(88.42 * map_lab_item("Renal function (hospitalized)", "Creatinine (enzymatic method) (CRE)", "umol/L")[1], 1.0)
        self.assertEqual(map_lab_item("Blood analysis (five categories)", "Hemoglobin (Hgb)", "g/L"), ("hemoglobin", 0.1))
        self.assertEqual(map_lab_item("Blood analysis (five categories)", "Hematocrit (HCT)", None), ("hematocrit", 100.0))
        self.assertEqual(map_lab_item("Blood gas analysis combination", "Hematocrit (HCT)", "%"), ("hematocrit", 1.0))
        self.assertEqual(map_lab_item("Renal function (hospitalized)", "Urea (urea)", "mmol/L"), ("bun", 2.801))

    def test_non_comparable_analytes_are_not_mapped(self):
        self.assertIsNone(map_lab_item("Brain natriuretic peptide (BNP)", "Brain natriuretic peptide (BNP)", "pg/mL"))
        self.assertIsNone(map_lab_item("Myocardial zymogram", "Creatine kinase isoenzyme (CK-MB)", "U/L"))

    def test_nursing_chart_is_not_used(self):
        self.assertFalse(hasattr(zigong, "load_nursing"))
        self.assertFalse(hasattr(zigong, "NURSING_VITALS"))

    def test_icu_departments(self):
        for name in ("Emergency department ward (EICU)", "Intensive care unit (ICU) ward a", "Neurosurgery severe NICU",
                     "Respiratory intensive care unit RICU", "Vascular Surgery ICU"):
            self.assertTrue(ICU_DEPARTMENT.search(name), name)
        for name in ("Respiratory and critical illness ward B", "Neurosurgery Ward a", "Department of Nephrology"):
            self.assertFalse(ICU_DEPARTMENT.search(name), name)


if __name__ == "__main__":
    unittest.main()
