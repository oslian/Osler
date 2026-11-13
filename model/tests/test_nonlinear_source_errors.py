import unittest
import numpy as np
import pandas as pd

from scripts.diagnose_nonlinear_source_errors import ARMS, decompose_errors


def frame():
    y = np.array([0., .5, 2., 4.])
    f = pd.DataFrame(dict(patient=['a', 'b', 'c', 'd'], component=['x']*4,
        fold=0, target='t', role='selection', truth=y))
    for arm in ARMS:
        f[arm] = y
    f['body_probe'] = y+1
    return f


class SourceErrorTests(unittest.TestCase):
    def test_bands_reproduce_patient_weighting(self):
        a = frame(); b = a.iloc[:1].copy(); b['fold'] = 1; b['body_probe'] = 3.
        result = decompose_errors(pd.concat([a,b]))
        self.assertAlmostEqual(result['normalized_mae']['patient']['body_probe'], 1.25)
        self.assertAlmostEqual(sum(x['body_minus_native_contribution'] for x in result['bands'].values()), 1.25)
        self.assertEqual(result['bands']['tail']['mass_fraction'], .25)

    def test_matched_constant_bias_removed_for_all_arms(self):
        a = frame()
        for arm in ARMS: a[arm] = a.truth+2
        a['body_probe'] = a.truth+4
        result = decompose_errors(a)['matched_local_offset']
        self.assertIsNotNone(result)
        for arm in ARMS: self.assertAlmostEqual(result['after'][arm], 0.)

    def test_singleton_unknown_and_no_positive_regression(self):
        a = frame().iloc[:1].copy(); a['body_probe'] = a.truth
        result = decompose_errors(a)
        self.assertIsNone(result['matched_local_offset'])
        self.assertIsNone(result['bands']['tail']['share_of_positive_body_regression'])
        self.assertIsNone(result['bands']['tail']['normalized_mae']['native'])

    def test_outer_duplicate_and_nonfinite_rejected(self):
        a = frame()
        with self.assertRaises(ValueError): decompose_errors(pd.concat([a,a]))
        a['role'] = 'test'
        with self.assertRaises(ValueError): decompose_errors(a)
        a = frame(); a.loc[0,'truth'] = np.nan
        with self.assertRaises(ValueError): decompose_errors(a)


if __name__ == '__main__': unittest.main()
