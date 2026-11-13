import unittest
import pandas as pd
from whole_body_jepa.reconstruction_partitions import patient_roles


class PartitionTest(unittest.TestCase):
    def setUp(self):
        self.q=pd.DataFrame({'uniquepid':['a','a','b','c','d']})
        self.p={'fit':[0,1],'tune':[2],'evaluation':[3],'hospital':[4]}

    def test_patient_not_old_shard_role(self):
        p=dict(self.p,fit=[2],tune=[0,1])
        self.assertEqual(patient_roles(self.q,p),{'b':'fit','a':'tune','c':'evaluation','d':'hospital'})

    def test_overlap_is_rejected(self):
        with self.assertRaises(ValueError):patient_roles(self.q,dict(self.p,tune=[1,2]))

    def test_missing_patient_is_rejected(self):
        with self.assertRaises(ValueError):patient_roles(self.q,dict(self.p,hospital=[]))
