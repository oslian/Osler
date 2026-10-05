import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_conditional_offset_shift import boundaries,diagnose


class ConditionalTests(unittest.TestCase):
    def make(self):
        return pd.DataFrame(dict(component=['a']*90+['b']*90+['c']*90,real=np.tile(np.arange(90),3),truth=np.tile(np.arange(90),3)+2))
    def test_uniform_within_shift(self):
        a=self.make();b=a.copy();b.truth+=5
        r=diagnose(a,b)['decomposition']
        self.assertAlmostEqual(r['composition'],0.)
        self.assertAlmostEqual(r['within_prediction_state'],5.)
        self.assertTrue(r['each_bin_has_three_supported_components'])
    def test_boundaries_ignore_truth(self):
        a=self.make();b=a.copy();b.truth*=100
        np.testing.assert_array_equal(boundaries(a),boundaries(b))
    def test_missing_bin_not_invented(self):
        a=self.make();b=a.loc[a.real<10]
        self.assertIsNone(diagnose(a,b)['decomposition'])


if __name__=='__main__':unittest.main()
