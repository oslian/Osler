import unittest
import numpy as np
import pandas as pd
from scripts.train_potassium_matched_encoder import event_pools, sample_events


class EventBalancedSamplingTest(unittest.TestCase):
    def test_patient_event_balance_and_query_retention(self):
        f=pd.DataFrame({'subject_id':['a']*10+['b']*2+['c'],
                        'stay_id':['s1']*10+['s2','s3','s4']})
        ids=np.array(['x']*9+['y','x','x',''])
        fit=np.arange(13)<12
        pools=event_pools(f,fit,ids)
        self.assertEqual([len(p) for p in pools],[2,2])
        np.testing.assert_array_equal(np.sort(np.concatenate([g for p in pools for g in p])),np.arange(12))
        draws=sample_events(pools,np.random.default_rng(7),100000)
        self.assertTrue(set(draws)==set(range(12)))
        for mask in (draws<9,draws==9,draws==10,draws==11):
            self.assertAlmostEqual(mask.mean(),.25,delta=.01)
        np.testing.assert_array_equal(draws,sample_events(pools,np.random.default_rng(7),100000))

    def test_missing_or_cross_patient_event_rejected(self):
        f=pd.DataFrame({'subject_id':['a','b'],'stay_id':['s','s']})
        for ids in (['','x'],['x','x']):
            with self.assertRaises(ValueError):event_pools(f,np.ones(2,bool),ids)


if __name__=='__main__':unittest.main()
