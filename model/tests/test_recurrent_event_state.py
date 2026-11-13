import unittest

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.recurrent_event_state import (
    FeatureStandardizer,
    RecurrentSignedResidual,
    patient_sequences,
    predict_recurrent,
)


class RecurrentEventStateTest(unittest.TestCase):
    def test_sequences_never_cross_patients_and_are_chronological(self) -> None:
        frame = pd.DataFrame(
            {
                "subject_id": ["b", "a", "a"],
                "t": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
                "_time_hours": [0.0, 24.0, 0.0],
            }
        )
        rows = patient_sequences(frame)
        self.assertEqual([list(item) for item in rows], [[2, 1], [0]])

    def test_recurrent_state_differs_from_anchor_reset(self) -> None:
        torch.manual_seed(4)
        frame = pd.DataFrame(
            {
                "subject_id": ["a", "a"],
                "t": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "_time_hours": [0.0, 24.0],
            }
        )
        features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        standardizer = FeatureStandardizer.fit(features)
        model = RecurrentSignedResidual(2, hidden_width=4)
        with torch.no_grad():
            model.readout[-1].weight.fill_(0.25)
        recurrent = predict_recurrent(model, standardizer, frame, features)
        reset = predict_recurrent(model, standardizer, frame, features, reset_each_anchor=True)
        self.assertAlmostEqual(float(recurrent[0]), float(reset[0]), places=6)
        self.assertNotAlmostEqual(float(recurrent[1]), float(reset[1]), places=6)


if __name__ == "__main__":
    unittest.main()
