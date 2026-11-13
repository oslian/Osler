import unittest
import numpy as np
import pandas as pd
import torch

from scripts.probe_assimilation_content import trace_content, assess, ARMS
from tests.test_native_body_reconstruction import BodyReconstructionTest
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations


class AssimilationContentTests(unittest.TestCase):
    setUp = BodyReconstructionTest.setUp

    def test_exact_final_and_last_observed_snapshot(self):
        with torch.inference_mode():
            for anchor in (1., 3.):
                result = trace_content(self.model, self.tokens, self.times, anchor, 'map')
                prediction = self.model.readout(result['final'][None])[0, 0]
                torch.testing.assert_close(prediction, self.model(self.tokens, self.times, anchor, 'map')[0])
                np.testing.assert_array_equal(result['positions'], [-1, 1, 1])
                torch.testing.assert_close(result['post'][1], result['post'][2], atol=0, rtol=0)
                self.assertEqual(float(result['post'][0].abs().sum()), 0.)
            result = trace_content(self.model, self.tokens, self.times, 1., 'map')
            torch.testing.assert_close(result['post'][1], result['final'], atol=0, rtol=0)

    def test_later_unrelated_update_does_not_change_earlier_snapshot(self):
        x = self.tokens.clone()
        x[1, 4] = 0  # Creatinine last measured at the first event, including observed zero.
        with torch.inference_mode():
            first = trace_content(self.model, x, self.times, 3., 'map')
            x[1, 2] = 9.
            second = trace_content(self.model, x, self.times, 3., 'map')
        self.assertEqual(first['positions'][1], 0)
        torch.testing.assert_close(first['post'][1], second['post'][1], atol=0, rtol=0)
        self.assertGreater(float((first['post'][2]-second['post'][2]).abs().sum()), 0.)

    def test_event_reference_uses_original_unique_batch(self):
        with torch.inference_mode():
            result = trace_content(self.model, self.tokens, self.times, 3., 'map')
            variables = self.model.trajectory.variables
            clean, _ = mask_native_target(self.tokens, self.times, variables, 'map')
            unique = np.unique(result['positions'][result['present']])
            encoded, _, _ = encode_native_organ_observations(self.model.trajectory.jepa, variables, clean[unique])
            expected = encoded[np.searchsorted(unique, result['positions'][result['present']])]
            torch.testing.assert_close(result['event'][result['present']], expected, atol=0, rtol=0)

    def test_invalid_time_rejected_before_mask(self):
        x = self.tokens.clone()
        x[1, 4:6] = 0
        for times in (torch.tensor([0., 0.]), torch.tensor([0., 2.]), torch.tensor([0., float('nan')])):
            with self.assertRaises(ValueError): trace_content(self.model, x, times, 1., 'map')

    def test_patient_equal_scoring(self):
        frame = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0],
            component=['x', 'x', 'y'], variable=['glucose']*3, truth=[0.]*3))
        for arm in ARMS: frame[arm] = [1., 3., 6.]
        result = assess(frame, 1, repetitions=20)
        self.assertEqual(result['normalized_mae']['patient']['ridge_post'], 4.)
        self.assertFalse(any(x['screen_pass'] for x in result['comparisons'].values()))
        with self.assertRaises(ValueError): assess(pd.concat([frame, frame.iloc[:1]]), 1, repetitions=20)


if __name__ == '__main__': unittest.main()
