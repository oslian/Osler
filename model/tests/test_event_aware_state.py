import unittest

import numpy as np
import pandas as pd

from whole_body_jepa.event_aware_state import (
    patient_equal_mae,
    patient_equal_weights,
    select_apply_threshold,
)


class EventAwareStateTest(unittest.TestCase):
    def test_patient_weights_equalize_total_mass(self) -> None:
        subjects = pd.Series(["a", "a", "b"])
        weights = patient_equal_weights(subjects)
        self.assertAlmostEqual(weights[:2].sum(), weights[2:].sum())

    def test_threshold_can_select_mixed_application(self) -> None:
        subjects = pd.Series(["a", "a", "b", "b"])
        truth = np.array([1.0, 1.0, 0.0, 0.0])
        frozen = np.zeros(4)
        always = np.ones(4)
        threshold, board = select_apply_threshold(
            np.array([-1.0, -1.0, 1.0, 1.0]), subjects, truth, frozen, always
        )
        chosen = np.where(np.array([-1.0, -1.0, 1.0, 1.0]) < threshold, always, frozen)
        self.assertEqual(patient_equal_mae(subjects, truth, chosen), 0.0)
        self.assertGreater(len(board), 2)


if __name__ == "__main__":
    unittest.main()

