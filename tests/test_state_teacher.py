import unittest

import numpy as np
import pandas as pd

from scripts.probe_state_teacher import ARMS, PARAMETERS, STUDENTS, assess, fit_students, process_input, unit_errors


class StateTeacherTests(unittest.TestCase):
    def test_process_input_does_not_leak_numbers_or_mutate(self):
        raw = np.arange(3*265, dtype=float).reshape(3, 265)
        before = raw.copy(); x = process_input(raw)
        np.testing.assert_array_equal(x[:, :106], np.zeros((3, 106)))
        np.testing.assert_array_equal(x[:, 106:], raw[:, 106:])
        np.testing.assert_array_equal(raw, before)
        with self.assertRaises(ValueError): process_input(raw[:, :-1])

    def test_fixed_budget_and_same_dimension_null(self):
        rng = np.random.default_rng(9)
        body = rng.normal(size=(50, 64)); raw = rng.normal(size=(50, 265))
        models, center, singles = fit_students(body, raw, body[:, 0],
            [str(i) for i in range(50)], ['a']*49+['b'], 19)
        self.assertEqual(singles, 1)
        self.assertTrue(np.isfinite(center))
        for arm, width in zip(STUDENTS, (64, 265, 64)):
            self.assertEqual(models[arm].n_features_in_, width)
            self.assertEqual(models[arm].n_iter_, 100)
            for k, v in PARAMETERS.items(): self.assertEqual(models[arm].get_params()[k], v)
        with self.assertRaises(ValueError):
            fit_students(body, raw, np.full(50, np.nan), [str(i) for i in range(50)], ['a']*50, 19)

    def test_imitation_is_not_truth_accuracy(self):
        n = 36
        frame = pd.DataFrame(dict(patient=[str(i) for i in range(n)], fold=[0]*n,
            target=['x']*n, component=np.repeat(['a', 'b', 'c'], 12), truth=np.zeros(n)))
        for arm in ARMS: frame[arm] = 0.
        frame['state_student'] = 2.; frame['raw_probe'] = 2.
        teacher = assess(frame, 'raw_probe', ('body_probe',), 19, 20)
        truth = assess(frame, 'truth', ('body_probe',), 19, 20)
        self.assertTrue(teacher['full_screen'])
        self.assertFalse(truth['full_screen'])
        self.assertEqual(teacher['normalized_mae']['patient']['state_student'], 0.)
        self.assertEqual(truth['normalized_mae']['patient']['state_student'], 2.)

    def test_patient_loss_aggregation_and_identity_guard(self):
        frame = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0], target=['x']*3,
                                 component=['c', 'c', 'd'], truth=[0., 0., 0.]))
        for arm in ARMS: frame[arm] = [-2., 2., 0.]
        people, _ = unit_errors(frame, 'truth')
        self.assertEqual(people.loc['a', 'state_student'], 2.)
        with self.assertRaises(ValueError): unit_errors(pd.concat([frame, frame.iloc[:1]]), 'truth')
        frame.loc[1, 'component'] = 'd'
        with self.assertRaises(ValueError): unit_errors(frame, 'truth')

    def test_conditional_median_teacher_need_not_preserve_truth_optimum(self):
        # The raw-informed teacher is optimal in each group, but its point
        # predictions lose the distribution needed by a less-informed student.
        group_a = np.r_[np.zeros(306), np.full(294, 100.)]
        group_b = np.full(400, 100.)
        truth = np.r_[group_a, group_b]
        teacher = np.r_[np.full(600, np.median(group_a)), np.full(400, np.median(group_b))]
        student, body_optimum = np.median(teacher), np.median(truth)
        self.assertEqual(student, 0.)
        self.assertEqual(body_optimum, 100.)
        self.assertAlmostEqual(np.mean(abs(teacher-truth)), 29.4)
        self.assertAlmostEqual(np.mean(abs(body_optimum-truth)), 30.6)
        self.assertAlmostEqual(np.mean(abs(student-truth)), 69.4)
        self.assertLess(np.mean(abs(student-teacher)), np.mean(abs(body_optimum-teacher)))


if __name__ == '__main__':
    unittest.main()
