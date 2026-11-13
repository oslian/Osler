import unittest
import numpy as np
from scripts.train_native_reconstruction_expanded import balanced_schedule, placebo_labels


class ExpandedSamplingTest(unittest.TestCase):
    def test_every_pair_and_equal_target_exposure(self):
        groups={0:[('a',0),('b',0),('c',0)],1:[('a',1),('b',1)]}
        s=balanced_schedule(groups,np.random.default_rng(7))
        self.assertEqual(set(s),set(sum(groups.values(),[])))
        self.assertEqual([sum(k[1]==c for k in s) for c in groups],[3,3])
        self.assertEqual(s,balanced_schedule(groups,np.random.default_rng(7)))

    def test_placebo_deranges_within_target(self):
        groups={c:[(p,c) for p in range(4)] for c in range(2)}
        data={k:dict(truth_normalized=10*k[1]+k[0]) for g in groups.values() for k in g}
        labels=placebo_labels(groups,data,np.random.default_rng(7))
        for c,keys in groups.items():
            self.assertEqual(sorted(labels[k] for k in keys),sorted(data[k]['truth_normalized'] for k in keys))
            self.assertTrue(all(labels[k]!=data[k]['truth_normalized'] for k in keys))

    def test_empty_and_single_patient_fail(self):
        with self.assertRaises(ValueError):balanced_schedule({},np.random.default_rng(7))
        with self.assertRaises(ValueError):placebo_labels({0:[('a',0)]},{},np.random.default_rng(7))
