import unittest
import numpy as np
from whole_body_jepa.native_clock import restore_clock


class NativeClockTest(unittest.TestCase):
    def test_only_clocks_change_and_late_gaps_survive(self):
        x=np.zeros((2,6),np.float32);x[:,3]=7;x[:,-1]=1
        y=restore_clock(x,np.array([0,1,0]),np.array([192.,204.,192.]),1)
        np.testing.assert_array_equal(y[:,[0,1,2,4]],x[:,[0,1,2,4]])
        self.assertAlmostEqual(float((y[1,-1]-y[0,-1])*168),12,places=4)
        np.testing.assert_array_equal(x[:,3],[7,7])

    def test_conflicting_or_missing_times_fail(self):
        x=np.zeros((2,6),np.float32)
        for ids,hours in [([0,0,1],[1,2,3]),([0],[1]),([0,1],[1,np.nan])]:
            with self.assertRaises(ValueError):restore_clock(x,np.array(ids),np.array(hours),1)

    def test_pre_admission_time_is_legal(self):
        y=restore_clock(np.zeros((2,6),np.float32),np.array([0,1]),np.array([-2.,0.]),1)
        self.assertAlmostEqual(float(y[0,-1]*168),-2.,places=5)
