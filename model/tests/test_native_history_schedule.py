import unittest
import pandas as pd
from whole_body_jepa.native_history_schedule import native_history_schedule, conflict_checked_history_schedule


class NativeHistoryScheduleTest(unittest.TestCase):
    def test_bedside_conflicts_and_source_namespaces(self):
        t = pd.Timestamp('2100-01-01')
        events = pd.DataFrame(dict(stay_id=['a', 'a'], source_kind=['lab_available', 'bedside_available'],
            source_id=['1', '1'], event_time=[t, t], measured_at=[t, t],
            variable=['map', 'map'], value=[80., 90.]))
        queries = pd.DataFrame(dict(stay_id=['a'], t=[t]))
        _, groups = conflict_checked_history_schedule(events, queries)
        self.assertEqual(len(groups[0]), 0)
        events['value'] = 80.
        _, groups = conflict_checked_history_schedule(events, queries)
        self.assertEqual(len(groups[0][0]), 2)
        events.loc[1, 'measured_at'] = t + pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError):
            conflict_checked_history_schedule(events, queries)

    def test_identity_time_patient_and_no_query_update(self):
        base=pd.Timestamp('2020-01-01')
        records=[('a','lab','1',0,4.),('a','lab','1',0,4.),
                 ('a','lab','2',1,4.),('a','rate','2',1,0.),
                 ('a','lab','3',3,6.),('b','lab','4',1,9.)]
        e=pd.DataFrame(records,columns=['stay_id','source_kind','source_id','hour','value'])
        e['event_time']=base+pd.to_timedelta(e.pop('hour'),unit='h')
        q=pd.DataFrame({'stay_id':['a','a','a'],'t':[base+pd.Timedelta(hours=h) for h in (2,2,0)]})
        canonical,schedules=native_history_schedule(e,q,2.)
        self.assertEqual(len(canonical),5)
        self.assertEqual(len(schedules[0]),1)
        self.assertEqual(len(schedules[0][0]),2)
        self.assertEqual(schedules[0][0].tolist(),schedules[1][0].tolist())
        self.assertEqual(canonical.loc[schedules[0][0],'value'].tolist(),[4.,0.])
        self.assertEqual(len(schedules[2][0]),1)

    def test_conflicts_and_missing_time_fail(self):
        e=pd.DataFrame({'stay_id':['a','a'],'source_kind':['lab','lab'],
                        'source_id':['1','1'],'event_time':['2020-01-01']*2,'value':[1.,2.]})
        q=pd.DataFrame({'stay_id':['a'],'t':['2020-01-02']})
        with self.assertRaises(ValueError):native_history_schedule(e,q)

    def test_query_replay_conflict_resolution_and_equal_remeasurement(self):
        base = pd.Timestamp('2020-01-01')
        e = pd.DataFrame([
            ('a', 'lab_available', '1', 0, 0, 2.),
            ('a', 'lab_available', '2', 1, 0, 5.),
            ('a', 'lab_available', '2', 2, 0, 2.),
            ('a', 'lab_available', '3', 3, 3, 2.),
            ('a', 'lab_available', '4', 4, 4, 0.),
            ('a', 'insulin_record', 'r', 1, None, 0.),
        ], columns=['stay_id', 'source_kind', 'source_id', 'event_time', 'measured_at', 'value'])
        e['variable'] = 'potassium'
        for col in ('event_time', 'measured_at'):
            e[col] = base + pd.to_timedelta(e[col], unit='h')
        q = pd.DataFrame({'stay_id': ['a'] * 5, 't': [base + pd.Timedelta(hours=h) for h in range(5)]})
        canonical, schedule = conflict_checked_history_schedule(e, q)
        def lab_values(frame, groups):
            ids = [int(i) for g in groups for i in g]
            rows = frame.loc[ids]
            return rows.loc[rows.source_kind.eq('lab_available'), 'value'].tolist()
        self.assertEqual([lab_values(canonical, s) for s in schedule],
                         [[2.], [], [2., 2.], [2., 2., 2.], [2., 2., 2., 0.]])
        for i in range(5):
            prefix, s = conflict_checked_history_schedule(e.loc[e.event_time <= q.t.iloc[i]], q.iloc[[i]])
            self.assertEqual(lab_values(prefix, s[0]), lab_values(canonical, schedule[i]))
        # Earlier source remains relevant to conflict resolution outside lookback.
        _, short = conflict_checked_history_schedule(e, q.iloc[[1]], .5)
        self.assertEqual(lab_values(canonical, short[0]), [])
        self.assertEqual(canonical.loc[short[0][0], 'source_kind'].tolist(), ['insulin_record'])
        e=e.iloc[:1].copy();e['event_time']=pd.NaT
        with self.assertRaises(ValueError):native_history_schedule(e,q)


if __name__=='__main__':unittest.main()
