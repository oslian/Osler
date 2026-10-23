import unittest

import numpy as np

from osler_jepa.deployment_clock import deployment_clock


class DeploymentClockTests(unittest.TestCase):
    def setUp(self):
        stays, onset = [], []
        for stay in range(100):
            for hour in range(0, 48, 6):
                stays.append(f"s{stay}")
                onset.append(float(hour))
        self.stays, self.onset = np.array(stays), np.array(onset)

    def test_stays_overlap_and_are_chronological_within_a_stay(self):
        clocked = deployment_clock(self.stays, self.onset, seed=0, census=12)
        ordered = self.stays[clocked.order]
        changes = int(np.sum(ordered[1:] != ordered[:-1]))
        self.assertGreater(changes, 3 * 100)                        # interleaved, not one stay after another
        for stay in ("s0", "s50"):
            rows = np.flatnonzero(self.stays == stay)
            np.testing.assert_allclose(np.diff(clocked.clock_hours[rows]), 6.0)

    def test_census_controls_concurrency(self):
        for census in (4, 20):
            clocked = deployment_clock(self.stays, self.onset, seed=1, census=census)
            starts = {s: clocked.clock_hours[self.stays == s].min() for s in np.unique(self.stays)}
            ends = {s: clocked.clock_hours[self.stays == s].max() + 1 for s in starts}
            probe = np.linspace(min(starts.values()) + 50, max(ends.values()) - 50, 200)
            mean_census = np.mean([sum(starts[s] <= t < ends[s] for s in starts) for t in probe])
            self.assertAlmostEqual(mean_census, census, delta=0.35 * census)

    def test_warmup_is_the_earliest_arrivals(self):
        clocked = deployment_clock(self.stays, self.onset, seed=2)
        warm = set(self.stays[clocked.warmup])
        self.assertEqual(len(warm), 25)                              # min(200, 25% of 100)
        first_arrival = {s: clocked.clock_hours[self.stays == s].min() for s in np.unique(self.stays)}
        self.assertLess(max(first_arrival[s] for s in warm), min(first_arrival[s] for s in first_arrival if s not in warm))


if __name__ == "__main__":
    unittest.main()
