import unittest

import numpy as np
import pandas as pd

from teacher_anchored_joint_jepa_external_gate import (
    _split_train_calibration_with_fraction,
)


class ExternalCalibrationSplitTests(unittest.TestCase):
    def test_split_is_patient_disjoint_and_respects_training_rows(self):
        frame = pd.DataFrame(
            {
                "subject_id": np.repeat([f"p{i}" for i in range(20)], 2),
                "value": np.arange(40),
            }
        )
        train_rows = np.arange(36, dtype=np.int64)
        fit, calibration = _split_train_calibration_with_fraction(
            frame, train_rows, seed=7, fraction=0.30
        )
        self.assertFalse(set(fit).intersection(set(calibration)))
        self.assertTrue(set(fit).union(set(calibration)).issubset(set(train_rows)))
        fit_patients = set(frame.iloc[fit].subject_id)
        calibration_patients = set(frame.iloc[calibration].subject_id)
        self.assertFalse(fit_patients.intersection(calibration_patients))

    def test_rejects_invalid_fraction(self):
        frame = pd.DataFrame({"subject_id": ["p1", "p2"]})
        with self.assertRaises(ValueError):
            _split_train_calibration_with_fraction(
                frame, np.arange(2), seed=1, fraction=0.5
            )


if __name__ == "__main__":
    unittest.main()
