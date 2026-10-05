import unittest
import numpy as np
import torch
from scripts.extract_additional_variable_snapshot import snapshot


class SnapshotTests(unittest.TestCase):
    def example(self):
        return dict(tokens=torch.tensor([[0.,0.,8.,0.,1.,1.,7.,0.,0.],
                    [0.,3.,9.,0.,1.,1.,7.,0.,.5/24]], dtype=torch.float64),
                    event_hours=torch.tensor([0.,5.], dtype=torch.float64),
                    anchor_hour=6., target_index=0, truth_normalized=999.)

    def test_latest_measurement_and_latency(self):
        value, age, mask = snapshot(self.example(), 3)
        np.testing.assert_array_equal(mask, [False,True,True])
        np.testing.assert_allclose(value, [np.nan,3.,9.], equal_nan=True)
        np.testing.assert_allclose(age, [np.nan,1.,1.5], equal_nan=True)

    def test_zero_and_late_older_result(self):
        e = self.example()
        e['tokens'][1,7] = 6/24
        value, age, mask = snapshot(e, 3)
        self.assertEqual(value[1], 0.)
        self.assertTrue(mask[1])
        self.assertEqual(age[1], 6.)

    def test_truth_invariance_and_revision_tie(self):
        e = self.example()
        e['tokens'][1,7] = 5/24
        a = snapshot(e, 3)
        self.assertEqual(a[0][1], 3.)
        e['truth_normalized'] = -999.
        for old, new in zip(a, snapshot(e, 3)):
            np.testing.assert_array_equal(old, new)

    def test_hidden_target(self):
        e = self.example()
        e['tokens'][1,3] = 1
        with self.assertRaises(ValueError):
            snapshot(e, 3)


if __name__ == '__main__':
    unittest.main()
