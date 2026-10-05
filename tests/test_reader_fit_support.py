import sqlite3
import unittest
from types import SimpleNamespace
from contextlib import closing
import copy

import numpy as np
import pandas as pd

from scripts.audit_reader_fit_support import ARMS, assess, require_roles, summarize, verify_subset
from scripts.cache_nonlinear_probe import encode_features
from scripts.verify_reader_fit_support import leaf_counts, assessment_parity


class ReaderFitSupportTests(unittest.TestCase):
    def test_complexity_counts_leaves_not_total_nodes(self):
        nodes = np.array([(80, 0), (40, 1), (40, 1)], dtype=[('count', 'i8'), ('is_leaf', 'i1')])
        tree = SimpleNamespace(nodes=nodes)
        model = SimpleNamespace(_predictors=[[tree] for _ in range(100)])
        self.assertEqual(leaf_counts(model), [2]*100)
        model._predictors.pop()
        with self.assertRaises(ValueError): leaf_counts(model)

    def test_roles_reject_overlap_and_missing_training_support(self):
        roles = dict(a='fit', b='fit', c='selection')
        require_roles(['a', 'b'], ['c'], ['a'], roles)
        for f, q, s in [(['a'], ['a'], ['a']), (['a'], ['c'], ['b']),
                        (['c'], ['a'], ['c']), (['a', 'a'], ['c'], ['a'])]:
            with self.assertRaises(ValueError): require_roles(f, q, s, roles)

    def test_subset_requires_actual_same_features_truth_and_role(self):
        with closing(sqlite3.connect(':memory:')) as db:
            db.execute('CREATE TABLE features(patient TEXT,target INTEGER,payload BLOB,digest TEXT)')
            body, raw = np.zeros(64, dtype=np.float32), np.zeros(265)
            payload, digest = encode_features(body, raw, 0., 1.)
            db.execute('INSERT INTO features VALUES (?,?,?,?)', ('a', 0, payload, digest))
            data = dict(patient=np.array(['a']), body=body[None], raw=raw[None],
                        truth=np.array([0.]), native=np.array([1.]))
            verify_subset(data, [0], db, 0, {'a': 'fit'}, 'fit')
            with self.assertRaises(ValueError): verify_subset(data, [0], db, 0, {'a': 'selection'}, 'fit')
            with self.assertRaises(ValueError): verify_subset(data, [0], db, 1, {'a': 'fit'}, 'fit')
            data['raw'][0, 0] = 1.
            with self.assertRaises(AssertionError): verify_subset(data, [0], db, 0, {'a': 'fit'}, 'fit')

    def frame(self):
        frame = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0],
                                 variable=['map']*3, component=['x', 'x', 'y'], truth=[0.]*3))
        for arm in ARMS: frame[arm] = [1., 3., 6.]
        return frame

    def test_repeated_patients_equalized_before_scoring(self):
        result = assess(self.frame(), 1, 20)
        self.assertEqual(result['normalized_mae']['patient']['full_raw'], 4.)
        self.assertFalse(any(s['screen_pass'] for s in result['comparisons'].values()))
        frame = self.frame()
        with self.assertRaises(ValueError): assess(pd.concat([frame, frame.iloc[:1]]), 1, 20)
        frame.loc[0, 'full_raw'] = np.nan
        with self.assertRaises(ValueError): assess(frame, 1, 20)

    def test_end_to_end_summary_retains_unsupported(self):
        targets, summary = summarize(self.frame(), ['map', 'bun'], 20)
        self.assertEqual(targets['bun']['status'], 'insufficient_support')
        self.assertEqual(summary['patients'], 2)
        self.assertEqual(summary['variables'], 1)
        self.assertEqual(len(summary['screens']), 3)
        self.assertTrue(all(not v for v in summary['screens'].values()))
        report = dict(targets=copy.deepcopy(targets), summary=copy.deepcopy(summary))
        report['summary']['fit_support'] = {}
        self.assertEqual(assessment_parity(targets, summary, report), 0.)
        report['targets']['map']['patients'] = 3
        with self.assertRaises(ValueError): assessment_parity(targets, summary, report)


if __name__ == '__main__': unittest.main()
