import unittest
import pandas as pd
from scripts.audit_additional_domain_insulin_support import rate_change_stays, support_counts


class InsulinSupportTests(unittest.TestCase):
    def rates(self, rows):
        return pd.DataFrame(rows, columns=['stay_id', 'label', 'offset', 'value'])

    def test_change_and_six_hour_boundary(self):
        rows=[('a','x',0,1),('a','x',360,2),('b','x',0,1),('b','x',361,2),
              ('c','x',0,1),('c','x',30,1)]
        self.assertEqual(rate_change_stays(self.rates(rows)), {'a'})

    def test_conflict_and_simultaneous_labels_excluded(self):
        rows=[('a','x',0,1),('a','x',30,2),('a','x',30,3),
              ('b','x',0,1),('b','x',30,2),('b','y',0,3),('b','y',30,4)]
        self.assertEqual(rate_change_stays(self.rates(rows)), set())

    def test_empty_and_repeat_documentation(self):
        self.assertEqual(rate_change_stays(self.rates([])), set())
        rows=[('a','x',0,1),('a','x',0,1),('a','x',30,2)]
        self.assertEqual(rate_change_stays(self.rates(rows)), {'a'})

    def test_unique_patient_not_event_weighting(self):
        frame=pd.DataFrame(dict(variable=['bun']*3,uniquepid=['a','a','b'],component=['x']*3))
        self.assertEqual(support_counts(frame, {'a'})['bun']['patients'], 1)
