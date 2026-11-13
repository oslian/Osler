import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.availability_supervision import future_pairs


class AuxiliaryLabelTest(unittest.TestCase):
    def test_availability_versions_ties_missing_and_zero(self):
        t = pd.Timestamp('2100-01-01')
        events = pd.DataFrame([
            ('a', 'zero', 'x', 0, 0, 0.),
            ('a', 'late', 'x', .5, 2, 99.),
            ('a', 'early', 'x', 6, 6, 2.),
            ('a', 'early', 'x', 6, 9, 3.),
            ('a', 'later', 'x', 8, 8, 4.),
            ('b', 'other', 'x', 0, 0, 100.),
        ], columns=['stay_id', 'source_id', 'variable', 'measured_at', 'available_at', 'value'])
        for col in ('measured_at', 'available_at'):
            events[col] = t + pd.to_timedelta(events[col], unit='h')
        queries = pd.DataFrame({'stay_id': ['a', 'b', 'a'],
                                't': [t+pd.Timedelta(hours=1), t, t+pd.Timedelta(hours=30)]})
        result = future_pairs(queries, events, ['x', 'missing'])
        self.assertEqual(result['current'][0, 0], 0)
        self.assertEqual(result['future'][0, 0], 3)
        self.assertEqual(result['actual_delta_t'][0, 0], 5)
        self.assertEqual(result['future_source_id'][0,0],'early')
        self.assertEqual(result['future_source_id'][1,0],'')
        self.assertFalse(result['paired'][1].any())
        self.assertFalse(result['paired'][2].any())
        self.assertFalse(result['paired'][:, 1].any())
        self.assertTrue(np.isnan(result['current'][2, 0]))
        shuffled = future_pairs(queries, events.sample(frac=1, random_state=7), ['x', 'missing'])
        for key in result:
            np.testing.assert_equal(result[key], shuffled[key])

    def test_invalid_query(self):
        q = pd.DataFrame({'stay_id': ['a', 'a'], 't': [pd.Timestamp('2100-01-01')]*2})
        with self.assertRaises(ValueError):
            future_pairs(q, pd.DataFrame(), ['x'])

    def test_conflicting_nearest_truth_does_not_choose_by_id_or_skip(self):
        t=pd.Timestamp('2100-01-01')
        events=pd.DataFrame(dict(stay_id=['a']*4,source_id=['start','a','z','farther'],
            variable=['x']*4,value=[1.,2.,5.,3.],
            measured_at=[t,t+pd.Timedelta(hours=6),t+pd.Timedelta(hours=6),t+pd.Timedelta(hours=6.5)]))
        events['available_at']=events.measured_at
        q=pd.DataFrame(dict(stay_id=['a'],t=[t]))
        for raw in (events,events.sample(frac=1,random_state=4)):
            out=future_pairs(q,raw,['x'])
            self.assertFalse(out['paired'][0,0])
            self.assertTrue(np.isnan(out['future'][0,0]))
            self.assertEqual(out['future_source_id'][0,0],'')
        events.loc[2,'value']=2.
        out=future_pairs(q,events,['x'])
        self.assertEqual(out['future'][0,0],2.)
        self.assertTrue(out['paired'][0,0])

    def test_equal_valued_remeasurement_has_distinct_identity(self):
        t=pd.Timestamp('2100-01-01')
        events=pd.DataFrame(dict(stay_id=['a']*3,source_id=['start','first','second'],
            variable=['x']*3,value=[1.,2.,2.],measured_at=[t,t+pd.Timedelta(hours=6),t+pd.Timedelta(hours=8)]))
        events['available_at']=events.measured_at
        q=pd.DataFrame(dict(stay_id=['a','a'],t=[t,t+pd.Timedelta(hours=2)]))
        out=future_pairs(q,events,['x'])
        np.testing.assert_array_equal(out['future'][:,0],[2.,2.])
        np.testing.assert_array_equal(out['future_source_id'][:,0],['first','second'])
