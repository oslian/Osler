import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.native_event_tokens import encode_group


class NativeEventTokenTest(unittest.TestCase):
    def test_bedside_reaches_encoder_and_preserves_zero(self):
        t = pd.Timestamp('2100-01-02')
        group = pd.DataFrame(dict(stay_id=['a'], source_kind=['bedside_available'],
                                 event_time=[t], measured_at=[t], variable=['respiratory_rate'], value=[0.]))
        token, event = encode_group(group, ['respiratory_rate'], [15.], [5.], [0], 1, 6.)
        self.assertEqual(token[0], -3.)
        self.assertEqual(token[1], 1.)
        self.assertEqual(token[-2], 1.)
        self.assertTrue(np.isnan(event[2]))

    def test_new_events_only_zero_and_age(self):
        t=pd.Timestamp('2100-01-02')
        g=pd.DataFrame({'stay_id':['a','a'],'source_kind':['lab_available','insulin_record'],
                        'event_time':[t,t],'measured_at':[t-pd.Timedelta(hours=2),pd.NaT],
                        'variable':['potassium','insulin_rate'],'value':[4.,0.]})
        token,event=encode_group(g,['potassium','glucose'],[4.,100.],[1.,20.],[0,1],2,6.)
        np.testing.assert_array_equal(token[2:4],[1.,0.])
        self.assertAlmostEqual(float(token[4]),2/24.)
        self.assertEqual(event[2],0.)
        self.assertEqual(token[-1],24/168.)

    def test_conflict_and_stale_not_observed(self):
        t=pd.Timestamp('2100-01-02')
        g=pd.DataFrame({'stay_id':['a','a'],'source_kind':['lab_available']*2,
                        'event_time':[t,t],'measured_at':[t,t],
                        'variable':['potassium']*2,'value':[4.,5.]})
        token,_=encode_group(g,['potassium'],[4.],[1.],[0],1,6.)
        self.assertEqual(token[1],0.)
        g=g.iloc[:1].copy();g['measured_at']=t-pd.Timedelta(hours=8)
        token,_=encode_group(g,['potassium'],[4.],[1.],[0],1,6.)
        self.assertEqual(token[1],0.)


if __name__=='__main__':unittest.main()
