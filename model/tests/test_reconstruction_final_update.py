import unittest
import torch
from tests.test_native_body_reconstruction import BodyReconstructionTest
from scripts.audit_reconstruction_final_update import trace_final


class FinalUpdateTraceTests(unittest.TestCase):
    setUp=BodyReconstructionTest.setUp

    def test_exact_final_at_anchor_and_after_gap(self):
        with torch.inference_mode():
            for anchor in (1.,3.):
                stages,metadata=trace_final(self.model,self.tokens,self.times,anchor,'map')
                expected=float(self.model(self.tokens,self.times,anchor,'map')[0])
                self.assertAlmostEqual(stages['final'],expected,places=7)
                self.assertEqual(metadata['last_gap_hours'],anchor-1.)

    def test_target_only_last_event_stays_masked(self):
        tokens=self.tokens.clone();tokens[1,4:6]=0
        with torch.inference_mode():
            stages,metadata=trace_final(self.model,tokens,self.times,1.,'map')
            self.assertEqual(metadata['events'],1)
            self.assertEqual(metadata['last_gap_hours'],1.)
            self.assertAlmostEqual(stages['final'],float(self.model(tokens,self.times,1.,'map')[0]),places=7)
