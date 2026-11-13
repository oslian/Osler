import unittest
import numpy as np
from scripts.audit_map_expanded_exposure import replay, summarize


class MapExposureTest(unittest.TestCase):
    def test_matches_training_list_comprehension(self):
        pools = [np.array([1, 2]), np.array([7]), np.array([9, 10, 11])]
        rng = np.random.default_rng(19)
        counts = {}
        snapshots = replay(pools, 19, epochs=3, steps=4, batch_size=5)
        for epoch in range(3):
            for _ in range(4):
                chosen = np.array([rng.choice(pools[i]) for i in rng.integers(len(pools), size=5)])
                for row in chosen:
                    counts[int(row)] = counts.get(int(row), 0) + 1
            self.assertEqual(snapshots[epoch], counts)
        self.assertEqual(summarize(counts, pools)['total_draws'], 60)

    def test_coverage_counts_distinct_rows_and_patients(self):
        result = summarize({1: 4}, [np.array([1, 2]), np.array([3])])
        self.assertAlmostEqual(result['row_coverage'], 1 / 3)
        self.assertEqual(result['patient_coverage'], .5)
        self.assertEqual(result['total_draws'], 4)


if __name__ == '__main__':
    unittest.main()
