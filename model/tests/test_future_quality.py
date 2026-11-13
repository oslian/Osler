import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.availability_supervision import future_pairs


class FutureQualityTest(unittest.TestCase):
    def test_nearest_invalid_not_replaced_by_farther_valid(self):
        t=pd.Timestamp('2100-01-01')
        events=pd.DataFrame(dict(stay_id=['a']*3,source_id=['1','2','3'],variable=['map']*3,
            measured_at=[t,t+pd.Timedelta(hours=6),t+pd.Timedelta(hours=6.5)],
            available_at=[t,t+pd.Timedelta(hours=6),t+pd.Timedelta(hours=6.5)],
            value=[80.,0.,90.],quality_valid=[True,False,True]))
        q=pd.DataFrame(dict(stay_id=['a'],t=[t]))
        label=future_pairs(q,events,['map'])
        self.assertFalse(label['paired'][0,0])
        self.assertTrue(np.isnan(label['future'][0,0]))
        events.loc[1,'quality_valid']=True
        label=future_pairs(q,events,['map'])
        self.assertTrue(label['paired'][0,0])
        self.assertEqual(label['future'][0,0],0.)


if __name__=='__main__':unittest.main()
