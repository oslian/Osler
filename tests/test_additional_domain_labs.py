import unittest
import pandas as pd
from scripts.extract_additional_domain_labs import fresh_unambiguous


class AdditionalLabTests(unittest.TestCase):
    def test_conflicting_latest_values_excluded(self):
        f=pd.DataFrame(dict(stay_id=['a']*3,variable=['x']*3,value=[1.,2.,3.],
            available_at=pd.to_datetime(['2020-01-01 03:00']*3),
            measured_at=pd.to_datetime(['2020-01-01 01:00','2020-01-01 02:30','2020-01-01 02:30'])))
        self.assertTrue(fresh_unambiguous(f).empty)

    def test_latest_draw_and_zero_retained(self):
        f=pd.DataFrame(dict(stay_id=['a']*2,variable=['x']*2,value=[9.,0.],
            available_at=pd.to_datetime(['2020-01-01 03:00']*2),
            measured_at=pd.to_datetime(['2020-01-01 01:00','2020-01-01 02:30'])))
        r=fresh_unambiguous(f)
        self.assertEqual(r.value.tolist(),[0.])

    def test_stale_label_not_counted(self):
        f=pd.DataFrame(dict(stay_id=['a'],variable=['x'],value=[1.],
            available_at=pd.to_datetime(['2020-01-01 03:00']),
            measured_at=pd.to_datetime(['2020-01-01 01:00'])))
        self.assertTrue(fresh_unambiguous(f).empty)
