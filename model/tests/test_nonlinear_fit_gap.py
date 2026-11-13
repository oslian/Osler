import unittest
import numpy as np

from scripts.diagnose_nonlinear_fit_gap import weighted_median, summarize, classify, original_population


class FitGapTests(unittest.TestCase):
    def test_original_population_preserves_float64_baseline(self):
        raw = np.array([2.1,2.2,3.4])
        cached = (raw.astype(np.float32).astype(float)-2.2)/1.1
        median,gap = original_population(raw,2.2,1.1,cached)
        self.assertEqual(median,0.)
        self.assertGreater(gap,0)
        with self.assertRaises(ValueError): original_population(raw,2.2,1.1,cached+.01)

    def test_weighted_median_uses_fit_weights(self):
        self.assertEqual(weighted_median([0,1,100],[1,5,1]),1.)
        with self.assertRaises(ValueError): weighted_median([1],[0])

    def test_bands_and_components_use_different_weighting(self):
        r = summarize([0,0,4],{'native':np.array([0,0,4]),'body_probe':np.array([1,1,7])},['a','a','b'])
        self.assertAlmostEqual(r['normalized_mae']['patient']['body_probe'],5/3)
        self.assertEqual(r['normalized_mae']['component']['body_probe'],2.)
        self.assertAlmostEqual(sum(b['body_minus_native_contribution'] for b in r['bands'].values()),5/3)
        self.assertIsNone(r['bands']['intermediate']['normalized_mae']['native'])

    def test_reversal_is_not_fit_failure(self):
        fit = summarize([0,0],{'native':np.ones(2),'body_probe':np.zeros(2)},['a','b'])
        query = summarize([0,0],{'native':np.zeros(2),'body_probe':np.ones(2)},['c','d'])
        self.assertEqual(classify(fit,query,'native','patient')['pattern'],'benefit_to_nonbenefit')

    def test_invalid_shapes_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): summarize([0],{'native':np.array([0]),'body_probe':np.array([np.nan])},['a'])
        with self.assertRaises(ValueError): summarize([0],{'native':np.array([0]),'body_probe':np.array([0])},['a','b'])


if __name__=='__main__': unittest.main()
