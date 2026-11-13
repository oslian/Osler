import unittest

import numpy as np
import pandas as pd

from whole_body_jepa.cross_variable_history import causal_wrong_patient_rows


class CausalWrongPatientRowsTest(unittest.TestCase):
    def test_donor_never_comes_from_the_future_or_same_patient(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["a", "a", "b", "b", "c", "c"],
                "_time_hours": [1.0, 5.0, 2.0, 8.0, 3.0, 9.0],
            }
        )
        donors = causal_wrong_patient_rows(frame, seed=7)
        present = donors >= 0

        self.assertTrue(np.all(frame._time_hours.to_numpy()[donors[present]]
                               <= frame._time_hours.to_numpy()[present]))
        self.assertTrue(np.all(frame.subject_id.to_numpy()[donors[present]]
                               != frame.subject_id.to_numpy()[present]))

    def test_missing_early_donor_is_explicit(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["a", "a", "b", "b"],
                "_time_hours": [0.0, 1.0, 5.0, 6.0],
            }
        )
        donors = causal_wrong_patient_rows(frame, seed=19)
        self.assertIn(-1, donors)


if __name__ == "__main__":
    unittest.main()
