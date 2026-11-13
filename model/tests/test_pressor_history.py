import unittest
import numpy as np
from scripts.build_map_pressor_history import documented_change


class PressorHistory(unittest.TestCase):
    def snapshot(self, rate, age):
        x = np.array([[[rate]]],float)
        return {'rate':x, 'known':np.isfinite(x), 'age_hours':np.array([[age]],float)}

    def test_carried_same_record_is_not_new_measurement(self):
        delta, known, _ = documented_change(self.snapshot(2,2), self.snapshot(2,1),1)
        self.assertFalse(known.any()); self.assertTrue(np.isnan(delta).all())

    def test_recorded_zero_change_is_valid_when_time_differs(self):
        delta, known, gap = documented_change(self.snapshot(0,0), self.snapshot(0,0),1)
        self.assertTrue(known.all()); self.assertEqual(delta.item(),0); self.assertEqual(gap.item(),1)

    def test_decrease_and_missing(self):
        d, k, _ = documented_change(self.snapshot(1,0), self.snapshot(3,0),1)
        self.assertEqual(d.item(),-2)
        _, k, _ = documented_change(self.snapshot(1,0), self.snapshot(np.nan,0),1)
        self.assertFalse(k.any())


if __name__ == '__main__': unittest.main()
