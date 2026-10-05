import unittest
import pandas as pd
from scripts.fixed_reconstruction_monitor import monitor_keys,monitor_scores


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame(dict(patient=['a','b','c'],target=['x']*3,component=['1','1','2']))
        self.roles={p:'selection' for p in self.frame.patient}

    def test_identity_selection_is_order_independent(self):
        pd.testing.assert_frame_equal(monitor_keys(self.frame,self.roles,1),monitor_keys(self.frame.iloc[::-1],self.roles,1))

    def test_reject_fit_patient(self):
        self.roles['a']='fit'
        with self.assertRaises(ValueError):monitor_keys(self.frame,self.roles)

    def test_missing_prediction_rejected(self):
        p=self.frame.iloc[:2].assign(truth=0.,fit_scale=1.,prediction=0.)
        with self.assertRaises(ValueError):monitor_scores(self.frame,p)

    def test_scores_and_small_groups(self):
        p=self.frame.assign(truth=0.,fit_scale=1.,prediction=[0.,0.,3.])
        r=monitor_scores(self.frame,p)['x']
        self.assertEqual(r['normalized_patient_mae'],1)
        self.assertEqual(r['normalized_component_mae'],1.5)


if __name__=='__main__':unittest.main()
