import unittest
import torch
from scripts.extract_additional_observation_support import support


class SupportTests(unittest.TestCase):
    def example(self):
        # 3 values, 3 masks, 3 ages in days; target0 remains entirely hidden.
        return dict(tokens=torch.tensor([[0.,0.,0.,0.,1.,0.,7.,0.,7.],
            [0.,0.,8.,0.,0.,1.,7.,7.,.5/24]]),event_hours=torch.tensor([0.,5.],dtype=torch.float64),
            anchor_hour=6.,target_index=0,truth_normalized=999.)
    def test_zero_observed_and_age(self):
        e=self.example();r=support(e,3)
        self.assertEqual(r['variables_observed'],2)
        self.assertEqual(r['fresh_1h_variables'],0)
        self.assertEqual(r['fresh_6h_variables'],2)
        self.assertAlmostEqual(r['median_latest_age_hours'],3.75)
        e['truth_normalized']=-999.;e['tokens'][:,:3]*=100
        self.assertEqual(support(e,3),r)
    def test_future_or_target_rejected(self):
        e=self.example();e['event_hours'][1]=7
        with self.assertRaises(ValueError):support(e,3)
        e=self.example();e['tokens'][0,3]=1
        with self.assertRaises(ValueError):support(e,3)


if __name__=='__main__':unittest.main()
