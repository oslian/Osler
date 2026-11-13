import unittest
import numpy as np
from scripts.audit_nested_readout_cv import nested_predictions


class NestedReadoutTests(unittest.TestCase):
    def test_outer_labels_cannot_change_own_fold_prediction(self):
        x = np.arange(40, dtype=float).reshape(-1, 1)
        sites = np.repeat(np.arange(10), 4).astype(str)
        groups = np.tile(['a', 'b'], 20)
        y = np.sin(x[:, 0]) + 2

        def solver(x, y, w, oldw, oldb, strength):
            return np.zeros(1), float(np.average(y, weights=w)/(1+strength))

        from sklearn.model_selection import GroupKFold
        _, valid = next(GroupKFold(n_splits=5).split(x, y, sites))
        first, _ = nested_predictions(x, y, groups, sites, np.zeros(1), 0., solver)
        changed = y.copy(); changed[valid] += 1000
        second, _ = nested_predictions(x, changed, groups, sites, np.zeros(1), 0., solver)
        np.testing.assert_array_equal(first[valid], second[valid])

    def test_perfect_original_is_preserved(self):
        x = np.arange(40, dtype=float).reshape(-1, 1)
        sites = np.repeat(np.arange(10), 4).astype(str)
        groups = np.tile(['a', 'b'], 20)
        y = x[:, 0] * 2 + 1
        def solver(*args):
            return np.zeros(1), 0.
        pred, folds = nested_predictions(x, y, groups, sites, np.array([2.]), 1., solver)
        np.testing.assert_array_equal(pred, y)
        self.assertTrue(all(f['choice']=='unchanged' for f in folds))
