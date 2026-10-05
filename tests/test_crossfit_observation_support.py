import unittest
import torch

from scripts.audit_crossfit_observation_support import observation_summary


class ObservationSupportTests(unittest.TestCase):
    def example(self):
        return dict(tokens=torch.tensor([[0., 0., 1., 0., .5/24, 7.],
                                         [0., 2., 0., 1., 7., 0.]]),
                    event_hours=torch.tensor([1., 5.]), anchor_hour=6.)

    def test_age_includes_anchor_gap_and_zero_is_observed(self):
        r = observation_summary(self.example(), 2)
        self.assertEqual(r['variables_seen'], 2)
        self.assertEqual(r['variables_fresh_1h'], 1)
        self.assertEqual(r['variables_fresh_6h'], 2)
        self.assertAlmostEqual(r['latest_age_median_hours'], 3.25)
        self.assertEqual(r['observed_entries'], 2)

    def test_missing_age_sentinel_does_not_count(self):
        e = self.example(); e['tokens'][:, 3] = 0
        r = observation_summary(e, 2)
        self.assertEqual(r['variables_seen'], 1)
        self.assertEqual(r['variables_fresh_1h'], 0)

    def test_future_event_rejected(self):
        e = self.example(); e['anchor_hour'] = 4.
        with self.assertRaises(ValueError):
            observation_summary(e, 2)
