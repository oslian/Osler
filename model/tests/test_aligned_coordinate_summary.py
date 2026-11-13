import unittest

from scripts.summarize_aligned_coordinates import summarize


class AlignmentSummaryTests(unittest.TestCase):
    def fixture(self):
        targets = [str(i) for i in range(47)]
        r = dict(targets={}, fit_checks={}, retention={}, summary=dict(normalized_target_macro={}))
        old = dict(folds={})
        for level in ('patient', 'component'):
            r['summary']['normalized_target_macro'][level] = dict(aligned_body=.9, body_probe=1.)
        for t in targets:
            r['targets'][t] = dict(normalized_mae={level: dict(
                aligned_body=.9, body_probe=1., raw_probe=.8, random_axes=.9, shuffled_axes=.95)
                for level in ('patient', 'component')})
        for f in map(str, range(5)):
            r['fit_checks'][f], r['retention'][f], old['folds'][f] = {}, {}, {}
            for t in targets:
                r['fit_checks'][f][t] = {a: dict(fit_mae=x) for a, x in
                    (('aligned_body', .7), ('random_axes', .8), ('shuffled_axes', .9))}
                r['retention'][f][t] = {role: {b: dict(median_mse_ratio=.6)
                    for b in ('latest', 'mean', 'process')} for role in ('fit', 'selection')}
                old['folds'][f][t] = dict(fit=dict(normalized_mae=dict(
                    patient=dict(body_probe=.8, raw_probe=.6))))
        return r, old

    def test_fit_and_selection_are_not_mixed(self):
        r, old = self.fixture()
        s = summarize(r, old)
        self.assertEqual(s['fit']['body_probe']['improves'], 235)
        self.assertEqual(s['fit']['raw_probe']['worsens'], 235)
        self.assertEqual(len(s['selection']['random_axes']['patient']['mean_better_targets']), 0)
        self.assertEqual(s['retention']['selection']['latest']['target_fold_summaries'], 235)
        self.assertAlmostEqual(s['macro_relative_improvement_vs_body']['patient'], .1)

    def test_incomplete_family_rejected(self):
        r, old = self.fixture()
        del r['fit_checks']['3']['2']
        with self.assertRaises(ValueError):
            summarize(r, old)


if __name__ == '__main__':
    unittest.main()
