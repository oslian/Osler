import copy
import unittest

from scripts.summarize_nonlinear_probe import compare


def fixture(native, body, passed=False):
    means = {level: dict(native=native, body_probe=body) for level in ('patient', 'component')}
    return dict(normalized_mae=means,
        comparisons={'body_probe_minus_native': {'all_8_contrasts_47_screen': passed}},
        per_component=[dict(patients=100, normalized_mae=means['patient'])])


class ProbeSummaryTests(unittest.TestCase):
    def test_equal_target_weighting_and_explicit_sensitivity(self):
        outer = dict(alt=fixture(10, 5, True), bun=fixture(1, 2))
        source = copy.deepcopy(outer)
        result = compare(source, outer, 'body_probe')
        self.assertEqual(result['mean_improvement_both_levels'], ['alt'])
        self.assertEqual(result['mean_harm_at_either_level'], ['bun'])
        self.assertEqual(result['levels']['patient']['native_macro'], 5.5)
        self.assertEqual(result['levels']['patient']['excluding_alt_ast']['candidate_macro'], 2)
        self.assertEqual(result['component_harms']['bun']['harms_with_at_least_30_patients'], 1)

    def test_source_outer_screens_are_not_merged_as_replication(self):
        source = dict(bun=fixture(2, 1, True), calcium=fixture(1, 2))
        outer = dict(bun=fixture(1, 2), calcium=fixture(2, 1, True))
        result = compare(source, outer, 'body_probe')
        self.assertEqual(result['source_and_outer_incremental_screen'], [])
        self.assertEqual(result['source_only_incremental_screen'], ['bun'])
        self.assertEqual(result['outer_only_incremental_screen'], ['calcium'])
        self.assertEqual(result['levels']['component']['source_outer_sign_disagreements'], ['bun', 'calcium'])

    def test_family_drift_rejected(self):
        with self.assertRaises(ValueError):
            compare(dict(bun=fixture(2, 1)), dict(alt=fixture(2, 1)), 'body_probe')


if __name__ == '__main__':
    unittest.main()
