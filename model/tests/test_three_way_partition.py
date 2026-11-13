import unittest
import numpy as np
from scripts.train_potassium_matched_encoder import three_way_partition


class ThreeWayPartitionTest(unittest.TestCase):
    def test_disjoint_patient_roles_and_order_invariance(self):
        ids=np.repeat(np.arange(100).astype(str),3)
        valid=np.ones(len(ids),bool);valid[::3]=False
        masks=three_way_partition(ids,valid,7)
        self.assertEqual([len(set(ids[m])) for m in masks],[60,20,20])
        np.testing.assert_array_equal(np.sum(masks,axis=0),valid.astype(int))
        for i,a in enumerate(masks):
            for b in masks[i+1:]:self.assertFalse(set(ids[a]) & set(ids[b]))
        reverse=three_way_partition(ids[::-1],valid[::-1],7)
        for a,b in zip(masks,reverse):np.testing.assert_array_equal(a,b[::-1])

    def test_missing_support(self):
        with self.assertRaises(ValueError):three_way_partition(['a'],[True],7)


if __name__=='__main__':unittest.main()
