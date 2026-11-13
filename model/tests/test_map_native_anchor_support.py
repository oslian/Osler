import unittest
import numpy as np
import pandas as pd
from scripts.audit_map_native_anchor_support import summary
from scripts.train_map_native_anchor_tree import feature_view
from scripts.audit_map_native_outcome_weights import summarize
from scripts.audit_map_native_history_density import draw_counts
from scripts.audit_map_native_raw_density import raw_counts


class MapNativeSupportTests(unittest.TestCase):
    def test_raw_count_respects_arrival_conflict_and_equal_valued_remeasurement(self):
        origin = pd.Timestamp('2100-01-01')
        def time(h):
            return origin+pd.Timedelta(hours=h)
        events = pd.DataFrame([
            ('s','a',0,0,70.), ('s','b',1,2,70.), ('s','c',1,3,80.),
            ('other','d',1,1,60.),
        ],columns=['stay_id','source_id','measured_at','available_at','value'])
        for col in ('measured_at','available_at'):
            events[col] = events[col].map(time)
        events['variable'] = 'map'
        events['quality_valid'] = True
        queries = pd.DataFrame({'stay_id':['s']*4,'t':[time(h) for h in (1,2,3,8)]})
        np.testing.assert_array_equal(raw_counts(queries,events,['map'])[:,0],[1,2,1,0])

    def test_carried_draws_and_actual_window(self):
        mask = np.ones((1,3,1))
        # At lag 6 the draw is 7h old relative to anchor; exclude it.
        ages = np.array([[[1/24],[0],[1/24]]])
        self.assertEqual(draw_counts(mask,ages,[6,1,0])[0,0],2)
        self.assertEqual(draw_counts(mask,ages,[6,1,0],6)[0,0],1)
        # A fresh remeasurement is distinct even without seeing any values.
        ages[0,2,0] = 0
        self.assertEqual(draw_counts(mask,ages,[6,1,0],6)[0,0],2)

    def test_reused_future_event_gets_one_weight(self):
        frame = pd.DataFrame({'subject_id':['p']*3, 'stay_id':['s']*3,
                              'future_id':['a','a','b'], 'truth':[0.,0.,0.]})
        points = {k:np.array([0.,0.,9.]) for k in ('own','full','support','persistence')}
        self.assertEqual(summarize(frame, points, False)['mae']['full'], 3.)
        self.assertEqual(summarize(frame, points, True)['mae']['full'], 4.5)
        frame.loc[1,'truth'] = 1.
        with self.assertRaises(ValueError):
            summarize(frame, points, True)

    def test_feature_modes_preserve_own_history_and_remove_only_requested_values(self):
        x = np.arange(36, dtype=np.float32).reshape(2, 18)
        allowed = np.array([True, True, False])
        full = feature_view(x, 3, 1, allowed, 'full').reshape(2, 3, 2, 3)
        support = feature_view(x, 3, 1, allowed, 'support').reshape(2, 3, 2, 3)
        own = feature_view(x, 3, 1, allowed, 'own')
        np.testing.assert_array_equal(full[:, 0, :, 2], 0)
        np.testing.assert_array_equal(full[:, 1, :, 2], 0)
        np.testing.assert_array_equal(full[:, 2, :, 2], 7)
        np.testing.assert_array_equal(support[:, 1:], full[:, 1:])
        np.testing.assert_array_equal(support[:, 0, :, 0], 0)
        np.testing.assert_array_equal(support[:, :, :, 1].reshape(2, -1), own)
        np.testing.assert_array_equal(x, np.arange(36).reshape(2, 18))

    def test_patient_and_future_event_counts_are_not_row_counts(self):
        q = pd.DataFrame({'stay_id': ['a', 'a', 'b', 'b'],
                          'uniquepid': ['p', 'p', 'r', 'r']})
        labels = dict(paired=np.array([[True], [True], [True], [False]]),
                      future_source_id=np.array([['x'], ['x'], ['x'], ['z']]),
                      measurement_age=np.array([[0.], [1.], [2.], [np.nan]]),
                      actual_delta_t=np.array([[6.], [5.], [7.], [np.nan]]))
        report = summary(q, labels)
        self.assertEqual(report['pairs'], 3)
        self.assertEqual(report['patients'], 2)
        self.assertEqual(report['unique_future_events'], 2)
        self.assertEqual(report['age_hours'][2], 1.)

    def test_empty_support_is_explicit(self):
        q = pd.DataFrame({'stay_id': ['a'], 'uniquepid': ['p']})
        labels = dict(paired=np.array([[False]]), future_source_id=np.array([['']]),
                      measurement_age=np.array([[np.nan]]), actual_delta_t=np.array([[np.nan]]))
        report = summary(q, labels)
        self.assertEqual(report['patients'], 0)
        self.assertEqual(report['age_hours'], [])


if __name__ == '__main__':
    unittest.main()
