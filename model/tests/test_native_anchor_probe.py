import unittest

import numpy as np
import pandas as pd

from scripts.score_native_anchor_probe import ARMS, losses
from scripts.train_native_anchor_probe import NEW_ARMS, PARAMETERS, fit_heads, inputs, labels, predict


class ConstantModel:
    def predict(self, x):
        return np.full(len(x), 2.)


class NativeAnchorTests(unittest.TestCase):
    def test_truth_objective_is_exact_and_not_teacher_imitation(self):
        y, native, r = np.array([5., -1., 0.]), np.array([2., 4., 0.]), np.array([1., -3., 0.])
        np.testing.assert_array_equal(abs(r-labels(y, native, 'anchored_residual')), abs(native+r-y))
        np.testing.assert_array_equal(labels(y, native, 'absolute_anchor'), y)
        np.testing.assert_array_equal(labels(y+7, native+7, 'anchored_residual'), y-native)

    def test_permutation_preserves_own_native_and_inputs(self):
        b = np.arange(192.).reshape(3, 64); n = np.array([10., 20., 30.]); original = b.copy()
        x = inputs(b, n, 'shuffled_anchor', np.array([2, 0, 1]))
        np.testing.assert_array_equal(x[:, :64], b[[2, 0, 1]])
        np.testing.assert_array_equal(x[:, -1], n)
        np.testing.assert_array_equal(b, original)
        p = inputs(b, n, 'anchor_only')
        self.assertTrue((p[:, :64] == 0).all())
        np.testing.assert_array_equal(p[:, -1], n)

    def test_residual_added_once_without_clipping(self):
        b = np.zeros((2, 64)); n = np.array([-10., 100.])
        np.testing.assert_array_equal(predict(ConstantModel(), b, n, 'anchored_residual'), n+2)
        np.testing.assert_array_equal(predict(ConstantModel(), b, n, 'absolute_anchor'), [2., 2.])

    def test_invalid_inputs_rejected(self):
        b = np.zeros((2,64)); n = np.zeros(2)
        with self.assertRaises(ValueError): inputs(b, n, 'shuffled_anchor', [0, 0])
        with self.assertRaises(ValueError): inputs(b, n, 'absolute_anchor', [1, 0])
        with self.assertRaises(ValueError): inputs(b, [np.nan, 0], 'anchor_only')
        with self.assertRaises(ValueError): labels([1, 2], [0], 'anchored_residual')

    def test_all_heads_same_fixed_budget_real_labels(self):
        rng = np.random.default_rng(8)
        b = rng.normal(size=(80,64)); n = b[:,0]; y = n + b[:,1]
        models, singleton = fit_heads(b, n, y, [str(i) for i in range(80)], ['a']*40+['b']*40, 7)
        self.assertEqual(singleton, 0)
        self.assertEqual(tuple(models), NEW_ARMS)
        for arm, model in models.items():
            self.assertEqual(model.n_features_in_, 65)
            self.assertEqual(model.n_iter_, 100)
            self.assertEqual({k:model.get_params()[k] for k in PARAMETERS}, PARAMETERS)
            self.assertTrue(np.isfinite(predict(model, b, n, arm)).all())

    def test_patient_equalized_not_fold_occurrence_weighted(self):
        frame = pd.DataFrame(dict(patient=['a','a','b'], component=['c','c','d'],
            target=['x']*3, fold=[0,1,0], truth=[0.]*3, **{a:[2.,2.,0.] for a in ARMS}))
        people, groups = losses(frame)
        self.assertEqual(people.anchored_residual.mean(), 1.)
        self.assertEqual(groups.anchored_residual.mean(), 1.)
        with self.assertRaises(ValueError): losses(pd.concat([frame,frame.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()
