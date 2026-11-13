import unittest
from collections import Counter
import numpy as np
from scripts.train_matched_broad_reconstruction import fixed_schedule
from scripts.train_native_reconstruction_expanded import balanced_schedule


class ScheduleTests(unittest.TestCase):
    def test_complete_equal_draws(self):
        groups={0:[('a',0),('b',0)],1:[('a',1)]}
        schedule=fixed_schedule(groups,np.random.default_rng(19),draws=4)
        self.assertEqual(Counter(k[1] for k in schedule),{0:4,1:4})
        self.assertEqual(set(schedule),set(sum(groups.values(),[])))

    def test_matches_original_when_max_equals_budget(self):
        groups={0:[('a',0),('b',0)],1:[('a',1)]}
        self.assertEqual(fixed_schedule(groups,np.random.default_rng(19),draws=2),balanced_schedule(groups,np.random.default_rng(19)))

    def test_insufficient_budget_rejected(self):
        with self.assertRaises(ValueError):fixed_schedule({0:[1,2]},np.random.default_rng(1),draws=1)
