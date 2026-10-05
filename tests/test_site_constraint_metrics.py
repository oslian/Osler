import unittest
import numpy as np
from scripts.audit_site_constraint_metrics import metric_changes


class MetricTests(unittest.TestCase):
    def test_smooth_improvement_need_not_improve_mae(self):
        result=metric_changes(np.array([0.,2.]),np.array([.9,1.2]))
        self.assertGreater(result['mae'],0)
        self.assertLess(result['smooth_l1'],0)

    def test_identical_residuals_zero(self):
        residual=np.array([-3.,0.,1.])
        self.assertEqual(metric_changes(residual,residual),{'mae':0.,'smooth_l1':0.})
