import unittest
import pandas as pd
from scripts.audit_potassium_treatment_support import window_support


class TreatmentSupportTests(unittest.TestCase):
    def test_time_and_patient_boundary(self):
        q=pd.DataFrame(dict(stay_id=['a','b','a'],offset=[360,360,361]))
        e=pd.DataFrame(dict(stay_id=['a','a'],offset=[0,361]))
        self.assertEqual(window_support(q,e).tolist(),[False,False,True])

    def test_zero_rate_event_is_not_missing(self):
        q=pd.DataFrame(dict(stay_id=['a'],offset=[10]))
        e=pd.DataFrame(dict(stay_id=['a'],offset=[10],rate=[0]))
        self.assertEqual(window_support(q,e).tolist(),[True])
