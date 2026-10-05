import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.event_window_history import event_windows,select_event_features


class EventWindowsTest(unittest.TestCase):
    def test_invalid_quality_is_not_an_excursion(self):
        t = pd.Timestamp('2100-01-01')
        events = pd.DataFrame([
            ('a', 0, 0, 70., True), ('b', 1, 1, 900., False),
            ('c', 2, 2, 0., True), ('c', 2, 4, 0., False),
        ], columns=['source_id', 'measured_at', 'available_at', 'value', 'quality_valid'])
        events['stay_id'] = 'a'; events['variable'] = 'map'
        for key in ('measured_at', 'available_at'):
            events[key] = t + pd.to_timedelta(events[key], unit='h')
        q = pd.DataFrame(dict(stay_id=['a', 'a'],
                              t=[t + pd.Timedelta(hours=3), t + pd.Timedelta(hours=4)]))
        out = event_windows(q, events, ['map'], [0.], [1.])
        self.assertEqual(out[0, 0, 0, 3], 70.)
        self.assertEqual(out[0, 0, 0, 2], 0.)
        self.assertAlmostEqual(out[0, 0, 0, 6], np.log1p(2))
        self.assertEqual(out[1, 0, 0, 2], 70.)
        self.assertAlmostEqual(out[1, 0, 0, 6], np.log1p(1))
        np.testing.assert_array_equal(out[:1], event_windows(
            q.iloc[:1], events.loc[events.available_at <= q.t.iloc[0]], ['map'], [0.], [1.]))

    def test_agreeing_invalid_source_invalidates_timestamp(self):
        t = pd.Timestamp('2100-01-01')
        events = pd.DataFrame(dict(stay_id=['a', 'a'], source_id=['x', 'y'],
            variable=['map', 'map'], value=[70., 70.], measured_at=[t, t],
            available_at=[t, t], quality_valid=[True, False]))
        q = pd.DataFrame(dict(stay_id=['a'], t=[t]))
        self.assertTrue((event_windows(q, events, ['map'], [0.], [1.]) == 0).all())

    def test_excursion_revisions_and_identity(self):
        t=pd.Timestamp('2100-01-01')
        events=pd.DataFrame([dict(stay_id='a',source_id=str(i),variable='k',value=v,
            measured_at=t+pd.Timedelta(hours=h),available_at=t+pd.Timedelta(hours=h))
            for i,(h,v) in enumerate([(0,4.),(.5,8.),(1,4.)])])
        q=pd.DataFrame({'stay_id':['a'],'t':[t+pd.Timedelta(hours=1)]})
        old=event_windows(q,events,['k'],[0.],[1.])
        self.assertEqual(old[0,0,0,3],8.)
        self.assertAlmostEqual(old[0,0,0,6],np.log1p(3))
        revision=events.iloc[1:2].assign(value=99.,available_at=t+pd.Timedelta(hours=2))
        np.testing.assert_array_equal(old,event_windows(q,pd.concat([events,revision,events.iloc[:1]]),['k'],[0.],[1.]))
        q['stay_id']='b'
        self.assertTrue((event_windows(q,events,['k'],[0.],[1.])==0).all())

    def test_numeric_support_and_unknown_channels(self):
        x=np.ones((2,3,2,9));y=select_event_features(x,[True,True,False],0,'support').reshape(2,3,2,9)
        self.assertTrue((y[:,0]==1).all());self.assertTrue((y[:,1,:,:6]==0).all())
        self.assertTrue((y[:,1,:,6:]==1).all());self.assertTrue((y[:,2]==0).all())
        self.assertTrue((x==1).all())

    def test_conflicting_values_are_not_excursions(self):
        t=pd.Timestamp('2100-01-01')
        events=pd.DataFrame([('old',0,0,4.),('a',1,1,3.),('b',1,1,9.),('b',1,3,3.)],
            columns=['source_id','measured_at','available_at','value'])
        events['stay_id']='a';events['variable']='k'
        for key in ('measured_at','available_at'):
            events[key]=t+pd.to_timedelta(events[key],unit='h')
        q=pd.DataFrame(dict(stay_id=['a','a'],t=[t+pd.Timedelta(hours=2),t+pd.Timedelta(hours=3)]))
        out=event_windows(q,events,['k'],[0.],[1.])
        self.assertEqual(out[0,0,0,3],4.)
        self.assertAlmostEqual(out[0,0,0,6],np.log1p(1))
        self.assertEqual(out[1,0,0,2],3.)
        self.assertEqual(out[1,0,0,3],4.)
        self.assertAlmostEqual(out[1,0,0,6],np.log1p(3))
        np.testing.assert_array_equal(out[:1],event_windows(q.iloc[:1],events.loc[events.available_at<=q.t.iloc[0]],['k'],[0.],[1.]))
        conflicted=events.loc[events.source_id.ne('old') & (events.available_at<=q.t.iloc[0])]
        self.assertTrue((event_windows(q.iloc[:1],conflicted,['k'],[0.],[1.])==0).all())


if __name__=='__main__':unittest.main()
