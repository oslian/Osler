import unittest
import pandas as pd
from whole_body_jepa.availability_anchors import rebuild


class AvailabilityAnchorsTest(unittest.TestCase):
    def events(self, rows):
        origin = pd.Timestamp('2100-01-01')
        return pd.DataFrame([dict(stay_id='s', source_id=i, measured_at=origin+pd.Timedelta(hours=m),
                                 available_at=origin+pd.Timedelta(hours=a), value=v) for i,m,a,v in rows])

    def test_new_anchor_requires_new_label(self):
        f = rebuild(self.events([('a',0,2,100),('b',6,7,120),('c',8,9,130)]))
        self.assertEqual(f.iloc[0].truth,130)
        self.assertEqual(f.iloc[0].actual_delta_t,6)
        self.assertEqual(f.iloc[0].current,100)
        self.assertEqual(f.iloc[0].mean_24,100)

    def test_late_old_measurement_does_not_replace_current(self):
        f = rebuild(self.events([('a',0,4,80),('b',2,3,100)]))
        self.assertEqual(f.iloc[-1].current,100)

    def test_duplicate_and_equal_remeasurement(self):
        raw = self.events([('a',0,1,100),('b',2,3,100)])
        pd.testing.assert_frame_equal(rebuild(raw), rebuild(pd.concat([raw,raw])))
        self.assertEqual(len(rebuild(raw)),2)
        self.assertEqual(rebuild(raw).iloc[-1].history_valid_6,1)

    def test_revision_not_visible_early_and_unscored_query_retained(self):
        f = rebuild(self.events([('a',0,1,100),('a',0,4,150)]))
        self.assertEqual(f.iloc[0].current,100)
        self.assertEqual(f.iloc[-1].current,150)
        self.assertTrue(f.truth.isna().all())


if __name__ == '__main__': unittest.main()
