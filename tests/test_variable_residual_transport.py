import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_variable_residual_transport import variable_summary


class TransportTests(unittest.TestCase):
    def frame(self):
        x = np.tile(np.arange(40, dtype=float), 3)
        c = np.repeat(np.arange(3), 40)
        return pd.DataFrame(dict(component=c, value_x=x+100*c, age_x=0., observed_x=True,
                                 truth=2*x+1000*c, real=x+1000*c, process=1000*c))

    def test_within_component_not_between_component(self):
        f = self.frame()
        a = variable_summary(f, 'x')
        f['truth'] -= 10000*f.component
        f['real'] -= 10000*f.component
        f['process'] -= 10000*f.component
        b = variable_summary(f, 'x')
        self.assertEqual(a['correlations'], b['correlations'])
        self.assertAlmostEqual(a['correlations']['residual'], 1.)

    def test_zero_observed_and_missing(self):
        f = self.frame()
        a = variable_summary(f, 'x')
        self.assertEqual(a['observed_patients'], 120)
        f.loc[0, 'observed_x'] = False
        f.loc[0, 'value_x'] = np.nan
        self.assertEqual(variable_summary(f, 'x')['observed_patients'], 119)

    def test_insufficient_and_constant_explicit(self):
        f = self.frame()
        self.assertEqual(variable_summary(f.iloc[:60], 'x')['status'], 'insufficient_support')
        f['value_x'] = 0.
        self.assertIsNone(variable_summary(f, 'x')['correlations']['residual'])


if __name__ == '__main__':
    unittest.main()
