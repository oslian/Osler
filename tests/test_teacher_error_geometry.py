import itertools
import unittest

import numpy as np
import pandas as pd

from scripts.diagnose_teacher_error_geometry import aggregate, decompose


class TeacherErrorGeometryTests(unittest.TestCase):
    def test_exhaustive_absolute_loss_identity(self):
        x = np.asarray(list(itertools.product((-2., -1., 0., 1., 2.), repeat=4)))
        r = decompose(*x.T)
        np.testing.assert_allclose(r.net_truth_delta, abs(x[:, 2]-x[:, 3])-abs(x[:, 0]-x[:, 3]))
        np.testing.assert_allclose(r.gross_harm-r.gross_benefit, r.net_truth_delta)
        keep = (r.teacher_quality == 'better') & (r.teacher_position == 'within_teacher_segment')
        self.assertTrue((r.loc[keep, 'net_truth_delta'] <= 0).all())

    def test_better_teacher_but_wrong_student(self):
        r = decompose([0., 0., 0.], [2., 2., 2.], [-1., 7., 1.], [2., 2., 2.])
        self.assertEqual(r.teacher_position.tolist(), ['away_teacher', 'beyond_teacher', 'within_teacher_segment'])
        self.assertEqual(r.net_truth_delta.tolist(), [1., 3., -1.])

    def test_following_bad_teacher_hurts(self):
        r = decompose([0.], [4.], [2.], [0.])
        self.assertEqual(r.teacher_quality.iloc[0], 'worse')
        self.assertEqual(r.teacher_position.iloc[0], 'within_teacher_segment')
        self.assertLess(r.net_imitation_delta.iloc[0], 0)
        self.assertGreater(r.net_truth_delta.iloc[0], 0)

    def test_truth_crossing_and_stationary_teacher(self):
        r = decompose([0., 0., 0.], [3., 0., 4.], [3., 2., 0.], [1., 1., 1.])
        self.assertEqual(r.truth_position.tolist(), ['crossing_truth', 'crossing_truth', 'unchanged'])
        self.assertEqual(r.teacher_position.iloc[1], 'teacher_stationary')

    def test_patient_equalized_contributions_not_row_counts(self):
        f = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0], target=['x']*3,
            component=['c', 'c', 'd'], body_probe=[0.]*3, raw_probe=[3.]*3,
            state_student=[2., 2., 0.], truth=[0.]*3))
        r = aggregate(f)
        self.assertEqual(r['total']['patient']['net_truth_delta'], 1.)
        self.assertEqual(r['partitions']['teacher_position']['within_teacher_segment']['patient']['fraction'], .5)
        self.assertEqual(r['partitions']['teacher_position']['within_teacher_segment']['occurrences'], 2)
        with self.assertRaises(ValueError): aggregate(pd.concat([f, f.iloc[:1]]))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): decompose([0.], [1.], [np.nan], [2.])


if __name__ == '__main__':
    unittest.main()
