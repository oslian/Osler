import unittest
import pandas as pd
from scripts.diagnose_coverage_support import association


class SupportTests(unittest.TestCase):
    def test_small_sample(self):
        d = pd.DataFrame({'x':[1,2], 'advantage':[2,3], 'component':['a','a']})
        self.assertIsNone(association(d,'x')['within_component_rho'])

    def test_within_group_relation(self):
        d = pd.DataFrame({'x':list(range(20))*2,'advantage':list(range(20))*2,'component':['a']*20+['b']*20})
        self.assertAlmostEqual(association(d,'x')['within_component_rho'],1)

    def test_constant_support(self):
        d = pd.DataFrame({'x':[1]*40,'advantage':range(40),'component':['a']*40})
        self.assertIsNone(association(d,'x')['within_component_rho'])


if __name__=='__main__':
    unittest.main()
