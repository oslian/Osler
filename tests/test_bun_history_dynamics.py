import unittest
import torch
from tests import test_native_body_reconstruction as fixture
from scripts.audit_bun_history_dynamics import replay_factors


class HistoryDynamicsTests(unittest.TestCase):
    setUp = fixture.BodyReconstructionTest.setUp

    def test_full_replay_and_single_event_no_old_update(self):
        with torch.inference_mode():
            for anchor in (1., 3.):
                result = replay_factors(self.model, self.tokens, self.times, anchor, 'map')
                self.assertAlmostEqual(result['full'], float(self.model(self.tokens, self.times, anchor, 'map')[0]), places=7)
            x = self.tokens.clone(); x[1,4:6] = 0
            r = replay_factors(self.model, x, self.times, 1., 'map')
            self.assertEqual(r['full'], r['latest_updates_fixed_clock'])
            self.assertEqual(r['no_dynamics'], r['latest_no_dynamics'])
