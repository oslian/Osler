import unittest
import numpy as np
from scripts.audit_map_expanded_exposure import replay


class MapSamplerReplayTest(unittest.TestCase):
    def test_matches_training_list_comprehension_at_each_epoch(self):
        pools = [np.array([1, 5, 8]), np.array([9]), np.array([11, 14])]
        for seed in (7, 19):
            expected = replay(pools, seed, epochs=4, steps=3, batch_size=5)
            rng = np.random.default_rng(seed)
            counts = {}
            for epoch in range(4):
                for _ in range(3):
                    chosen = np.array([rng.choice(pools[i]) for i in rng.integers(len(pools), size=5)])
                    for row in chosen:
                        counts[int(row)] = counts.get(int(row), 0) + 1
                self.assertEqual(counts, expected[epoch])


if __name__ == '__main__':
    unittest.main()
