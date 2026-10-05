import unittest
import numpy as np
import pandas as pd
from scripts.try_patient_equal_correction import predict


class PatientEqualTest(unittest.TestCase):
    def test_query_labels_are_not_used(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=200)
        source = pd.DataFrame(dict(real=x, truth=x+1+rng.normal(size=200)*.2,
                                   component=np.repeat(np.arange(4), 50), fit_scale=1.))
        query = source.copy()
        a, status = predict(source, query)
        self.assertEqual(status, 'fitted')
        query['truth'] = -999.
        b, _ = predict(source, query)
        np.testing.assert_array_equal(a, b)

    def test_missing_support(self):
        source = pd.DataFrame(dict(real=[0.], truth=[1.], component=[0], fit_scale=[1.]))
        self.assertEqual(predict(source, source), (None, 'insufficient_support'))
