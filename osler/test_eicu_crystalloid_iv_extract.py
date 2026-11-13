import unittest

from eicu_crystalloid_iv_extract import classify


class CrystalloidClassificationTests(unittest.TestCase):
    def test_typed_isotonic_fluids(self):
        for label in ("NS IVF", "ns ", "volume (ml)-0.9 %  sodium chloride infusion", "volume (ml)-sodium chloride 0.9% infusion",
                      "ns  w/20 meq kcl 1000 ml", "volume (ml)-sodium chloride (normal saline) 0.9 % bolus 1,000 ml", "NS bolus"):
            self.assertEqual(classify(label), "saline", label)
        for label in ("lr ", "LR IVF", "volume (ml)-lactated ringers infusion", "plasmalyte ", "Normosol"):
            self.assertEqual(classify(label), "balanced", label)

    def test_excluded_and_untyped(self):
        for label in ("volume (ml)-piperacillin-tazobactam (zosyn) 3.375 g in sodium chloride 0.9 % 100 ml ivpb",
                      "saline flush (ml)", "ns kvo", "d5 0.45 ns ", "volume (ml)-0.45 % sodium chloride solution",
                      "hypertonic saline 3% ", "d5 lr ", "volume (ml)-dextrose 5 % in lactated ringers infusion",
                      "fentanyl (ns)-in", "heparin (ns 0.45%)-in", "1/2 ns"):
            self.assertIsNone(classify(label), label)
        self.assertEqual(classify("crystalloids"), "untyped")


if __name__ == "__main__":
    unittest.main()
