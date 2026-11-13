import unittest
import pandas as pd
from scripts.diagnose_seen_domain_correction import query_people, matched_fits


class SeenTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(patient=f'p{c}_{i}',component=c,truth=i) for c in range(4) for i in range(100)])

    def test_outcome_and_order_independence(self):
        f=self.frame();a=query_people(f)
        f=f.iloc[::-1].copy();f['truth']=-999
        self.assertEqual(a,query_people(f))
        self.assertEqual(len(a),80)

    def test_same_queries_equal_size_no_patient_leakage(self):
        f=self.frame();q=query_people(f)
        seen,unseen,query=matched_fits(f,q,0)
        self.assertEqual(len(seen),len(unseen))
        self.assertEqual(len(query),20)
        self.assertTrue((seen.component==0).any())
        self.assertFalse((unseen.component==0).any())
        self.assertFalse(set(seen.patient)&q)
        self.assertFalse(set(unseen.patient)&q)


if __name__=='__main__':unittest.main()
