import unittest
import pandas as pd
from scripts.split_insulin_potassium_cohort import split_roles, held_hospital


class ExpandedSplitTests(unittest.TestCase):
    def test_cross_hospital_patient_is_purged_from_source(self):
        held=next(i for i in range(100) if held_hospital(i))
        source=next(i for i in range(100) if not held_hospital(i))
        q=pd.DataFrame(dict(subject_id=['same','same']+[f'p{i}' for i in range(20)],hospitalid=[held,source]+[source]*20))
        roles=split_roles(q,7)
        self.assertTrue(roles['hospital'][0])
        self.assertTrue(roles['excluded'][1])
        for name in ('fit','tune','evaluation'):
            self.assertFalse(roles[name][:2].any())
