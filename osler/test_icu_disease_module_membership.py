import unittest

import pandas as pd

from icu_disease_module_membership import diagnosis_matches, kdigo_stays, split_patterns


class MembershipTests(unittest.TestCase):
    def test_short_acronyms_match_whole_words_only(self):
        titles = pd.Series(["Other medical complications", "Disseminated intravascular coagulation (DIC)"])
        hit = diagnosis_matches(titles, pd.Series(["", ""]), pd.Series([9, 9]), ("dic",))
        self.assertEqual(hit.tolist(), [False, True])

    def test_code_patterns_respect_icd_version_and_dots(self):
        codes, words, phrases = split_patterns(("584", "n17", "728.86", "heart failure"))
        self.assertEqual(codes, ["584", "n17", "72886"])
        self.assertEqual(phrases, ["heart failure"])
        hit = diagnosis_matches(
            pd.Series(["", "", ""]), pd.Series(["5849", "N179", "5849"]), pd.Series([9, 10, 10]), ("584", "n17")
        )
        self.assertEqual(hit.tolist(), [True, True, False])

    def test_kdigo_absolute_and_ratio_rules(self):
        frame = pd.DataFrame({
            "stay_id": [1, 1, 2, 2, 3, 3],
            "charttime": pd.to_datetime([
                "2100-01-01 00:00", "2100-01-02 00:00",   # +0.3 within 48h
                "2100-01-01 00:00", "2100-01-05 00:00",   # 1.5x within 7 days, beyond 48h
                "2100-01-01 00:00", "2100-01-02 00:00",   # +0.2 only
            ]),
            "valuenum": [1.0, 1.3, 1.0, 1.6, 1.0, 1.2],
        })
        self.assertEqual(kdigo_stays(frame), {1, 2})


if __name__ == "__main__":
    unittest.main()
