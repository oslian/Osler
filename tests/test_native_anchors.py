import unittest
import pandas as pd

from scripts.audit_potassium_native_anchors import native_queries


class NativeAnchorTest(unittest.TestCase):
    def test_available_not_measured_and_repeated_equal_values(self):
        t=pd.Timestamp('2100-01-01')
        events=pd.DataFrame([dict(stay_id='a',source_id=str(i),variable='potassium',value=4.,
            measured_at=t+pd.Timedelta(hours=i),available_at=t+pd.Timedelta(hours=i+1)) for i in range(2)])
        identities=pd.DataFrame({'stay_id':['a'],'subject_id':['person']})
        q=native_queries(pd.concat([events,events.iloc[:1]]),identities)
        self.assertEqual(list(q.t),[t+pd.Timedelta(hours=1),t+pd.Timedelta(hours=2)])
        self.assertEqual(set(q.subject_id),{'person'})
        # No future glucose value or future outcome eligibility is consulted.
        altered=events.assign(variable='glucose',source_id=['g1','g2'])
        pd.testing.assert_frame_equal(q,native_queries(pd.concat([events,altered]),identities))

    def test_ambiguous_identity_rejected(self):
        with self.assertRaises(ValueError):
            native_queries(pd.DataFrame(),pd.DataFrame({'stay_id':['a','a'],'subject_id':['x','y']}))


if __name__=='__main__':unittest.main()
