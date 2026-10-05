import unittest

import numpy as np
import pandas as pd
import torch

from scripts.audit_update_invariance import probe_stream
from whole_body_jepa.recurrent_event_state import FeatureStandardizer, RecurrentSignedResidual, predict_recurrent


class UpdateInvarianceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = RecurrentSignedResidual(5)
        torch.nn.init.normal_(self.model.readout[-1].weight, std=.1)
        self.x = np.zeros((4, 5), dtype=np.float32)
        self.norm = FeatureStandardizer(np.zeros(5), np.ones(5))
        t = pd.Timestamp('2020-01-01')
        self.frame = pd.DataFrame({'subject_id': ['a']*4, 't': [t+pd.Timedelta(hours=h) for h in (0, 2, 6, 8)],
            '_time_hours': [0, 2, 6, 8], 'truth': [140., np.nan, np.nan, np.nan],
            'frozen': [100.]*4, 'future_event_time': [t+pd.Timedelta(hours=6), pd.NaT, pd.NaT, pd.NaT],
            'treatment_changed': [False, True, False, False]})

    def test_parity_and_local_feedback(self):
        pred, withheld, meta = probe_stream(self.model, self.norm, self.frame, self.x, 10.)
        production = 100 + predict_recurrent(self.model, self.norm, self.frame, self.x, feedback_scale=10.)*10
        np.testing.assert_allclose(pred, production, atol=1e-5)
        np.testing.assert_allclose(pred[:2], withheld[:2])
        self.assertEqual(meta[:, 0].tolist(), [0, 0, 1, 0])
        self.assertEqual(meta[2, 1], 1)
        self.assertGreater(abs(pred[2]-withheld[2]), 1e-6)
        self.assertEqual(pred[3], withheld[3])

    def test_density_only_affects_later_predictions(self):
        first, _, _ = probe_stream(self.model, self.norm, self.frame, self.x, 10.)
        repeated, _, meta = probe_stream(self.model, self.norm, self.frame, self.x, 10., 4)
        self.assertEqual(first[0], repeated[0])
        self.assertGreater(abs(first[1]-repeated[1]), 1e-6)
        self.assertEqual(meta[:, 0].sum(), 1)

    def test_patient_isolation(self):
        original, _, _ = probe_stream(self.model, self.norm, self.frame, self.x, 10.)
        other = self.frame.assign(subject_id='b', truth=900.)
        frame = pd.concat([self.frame, other], ignore_index=True)
        pred, _, _ = probe_stream(self.model, self.norm, frame, np.vstack([self.x, self.x+5]), 10.)
        np.testing.assert_array_equal(pred[:4], original)


if __name__ == '__main__':
    unittest.main()
