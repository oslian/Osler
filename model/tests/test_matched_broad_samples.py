import unittest
import pandas as pd
from scripts.build_matched_broad_samples import matched_people


class MatchedBroadTests(unittest.TestCase):
    def test_order_independent_counts(self):
        old=pd.DataFrame(dict(uniquepid=['a','b'],role=['fit','encoder_tune'],hospitalid=[1,2]))
        new=pd.DataFrame(dict(uniquepid=['c','d','e','f'],role=['fit','fit','encoder_tune','encoder_tune'],hospitalid=[1,1,2,2]))
        self.assertEqual(matched_people(old,new),matched_people(old.iloc[::-1],new.iloc[::-1]))
        self.assertEqual(len(matched_people(old,new)),2)

    def test_insufficient_and_overlap_rejected(self):
        old=pd.DataFrame(dict(uniquepid=['a'],role=['fit'],hospitalid=[1]))
        with self.assertRaises(ValueError):matched_people(old,old)
        new=pd.DataFrame(dict(uniquepid=['b'],role=['fit'],hospitalid=[2]))
        with self.assertRaises(ValueError):matched_people(old,new)
