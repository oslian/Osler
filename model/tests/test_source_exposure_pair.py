from collections import Counter
from copy import deepcopy
import unittest
import numpy as np
from scripts.train_source_exposure_pair import epoch_plan, shuffled_labels


class ExposurePairTest(unittest.TestCase):
    def test_group_target_budget_and_resume(self):
        groups = {g: {c: [(f"{g}{i}", c) for i in range(10)] for c in (0, 1)} for g in ("a", "b")}
        queues = {g: {c: {} for c in d} for g, d in groups.items()}
        rng = np.random.default_rng(19)
        first = epoch_plan(groups, queues, rng, draws=6)
        self.assertEqual(Counter((g, k[1]) for g, k in first),
                         Counter({(g, c): 6 for g in groups for c in (0, 1)}))
        saved = deepcopy(queues)
        rng2 = np.random.default_rng()
        rng2.bit_generator.state = deepcopy(rng.bit_generator.state)
        second = epoch_plan(groups, queues, rng, draws=4)
        self.assertEqual(second, epoch_plan(groups, saved, rng2, draws=4))
        self.assertEqual(len(set(first + second)), 40)

    def test_placebo_preserves_target_labels(self):
        class Store:
            groups = {0: [(i, 0) for i in range(5)], 1: [(i, 1) for i in range(3)]}
            labels = {(i, c): float(i + c * 100) for c, n in ((0, 5), (1, 3)) for i in range(n)}
        result = shuffled_labels(Store())
        for keys in Store.groups.values():
            self.assertEqual(sorted(result[k] for k in keys), sorted(Store.labels[k] for k in keys))
            self.assertTrue(all(result[k] != Store.labels[k] for k in keys))
