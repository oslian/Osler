import unittest

import pandas as pd

from patient_disjoint_event_holdout import build_patient_disjoint_holdout


class PatientDisjointEventHoldoutTests(unittest.TestCase):
    def test_removes_every_stay_for_selected_patients(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["p1", "p1", "p2", "p3"],
                "stay_id": ["s1", "s2", "s3", "s4"],
                "value": [1, 2, 3, 4],
            }
        )
        holdout, report = build_patient_disjoint_holdout(
            frame, sampled_stays=1, seed=2026
        )
        excluded = set(frame.subject_id) - set(holdout.subject_id)
        self.assertTrue(excluded)
        self.assertTrue(frame[frame.subject_id.isin(excluded)].stay_id.nunique() >= 1)
        self.assertEqual(report["patient_overlap"], 0)
        self.assertEqual(
            report["holdout_patients"], holdout.subject_id.nunique()
        )


if __name__ == "__main__":
    unittest.main()
