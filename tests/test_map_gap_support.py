import unittest
import numpy as np
from scripts.audit_map_derived_gap_support import support_groups


class SupportTests(unittest.TestCase):
    def test_partition_uses_only_known_masks(self):
        d=np.zeros((3,3,2))
        d[0,1,-1]=1
        d[1,1,0]=1
        g=support_groups(d)
        np.testing.assert_array_equal(g['current'],[True,False,False])
        np.testing.assert_array_equal(g['history_only'],[False,True,False])
        np.testing.assert_array_equal(g['none'],[False,False,True])
        d[:,0]=999
        for k,v in g.items():
            np.testing.assert_array_equal(support_groups(d)[k],v)


if __name__=='__main__':
    unittest.main()
