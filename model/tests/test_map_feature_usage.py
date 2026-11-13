import unittest
from types import SimpleNamespace
import numpy as np
from scripts.audit_map_feature_usage import split_usage


class UsageTests(unittest.TestCase):
    def test_only_split_nodes_contribute(self):
        nodes = np.array([(0,1,3.),(0,1,2.),(1,0,99.)],
                         dtype=[('is_leaf','u1'),('feature_idx','u4'),('gain','f8')])
        model = SimpleNamespace(_predictors=[[SimpleNamespace(nodes=nodes)]])
        count,gain = split_usage(model,3)
        np.testing.assert_array_equal(count,[0,2,0])
        np.testing.assert_array_equal(gain,[0.,5.,0.])


if __name__ == '__main__':
    unittest.main()
