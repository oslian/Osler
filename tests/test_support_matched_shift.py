import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_support_matched_shift import strata,diagnose


class MatchingTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(component=np.repeat(['a','b','c'],90),real=np.tile(np.arange(90),3),
            truth=np.tile(np.arange(90),3),fresh_1h_variables=2.,median_latest_age_hours=1.,history_span_hours=12.))
    def test_uniform_shift_no_false_cure(self):
        a=self.frame();b=a.copy();b.truth+=5
        r=diagnose(a,b)
        self.assertEqual(r['source_retained_patient_fraction'],1.)
        self.assertAlmostEqual(r['matched_signed_shift'],5.)
        self.assertAlmostEqual(r['matched_absolute_bin_shift'],5.)
    def test_cuts_do_not_read_labels(self):
        a=self.frame();b=a.copy();b.truth=999
        self.assertEqual(strata(a,a)[1],strata(b,b)[1])
    def test_no_overlap_explicit(self):
        a=self.frame();b=a.copy();b.fresh_1h_variables=0
        r=diagnose(a,b)
        self.assertEqual(r['status'],'no_supported_overlap')
        self.assertEqual(r['outer_retained_patient_fraction'],0.)
        self.assertNotIn('matched_signed_shift',r)


if __name__=='__main__':unittest.main()
