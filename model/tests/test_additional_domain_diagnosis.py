import unittest
import pandas as pd
from scripts.diagnose_additional_domains import domain_summary


class DiagnosisTests(unittest.TestCase):
    def test_equal_components_and_leave_one_out(self):
        f=pd.DataFrame(dict(component=['a']*30+['b']*2+['c'],truth=0.,
            real=[1.]*30+[5.]*2+[-3.],process=2.,population=2.))
        r=domain_summary(f);s=r['comparisons']['process']
        self.assertEqual(s['all_component_mean'],1.)
        self.assertEqual(s['supported_component_mean'],-1.)
        self.assertEqual(s['leave_one_component_mean_range'],[0.,2.])
        self.assertEqual(s['harmed_components'],2)
        self.assertEqual(r['positive_median_bias_components'],2)
        self.assertEqual(r['negative_median_bias_components'],1)

    def test_single_small_component(self):
        f=pd.DataFrame(dict(component=['a'],truth=[0.],real=[0.],process=[1.],population=[1.]))
        s=domain_summary(f)['comparisons']['process']
        self.assertIsNone(s['supported_component_mean'])
        self.assertIsNone(s['leave_one_component_mean_range'])


if __name__=='__main__':unittest.main()
