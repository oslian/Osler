import unittest

import numpy as np

from osler_jepa.instrumental_variable import (
    cluster_bootstrap,
    first_stage_f,
    standardized_mean_difference,
    two_stage_least_squares,
)


def confounded_data(n_clusters=200, per_cluster=40, effect=-2.0, seed=0):
    """Hospital preference instrument; an unmeasured severity confounds treatment and outcome."""

    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    preference = rng.uniform(0.05, 0.95, n_clusters)[cluster]
    severity = rng.normal(size=len(cluster))                      # unmeasured
    covariate = rng.normal(size=len(cluster))                     # measured
    treated = (rng.uniform(size=len(cluster)) < np.clip(preference - 0.25 * severity, 0, 1)).astype(float)
    outcome = effect * treated + 1.5 * severity + 0.8 * covariate + rng.normal(size=len(cluster))
    return outcome, treated, preference, covariate[:, None], cluster


class InstrumentalVariableTests(unittest.TestCase):
    def test_2sls_recovers_the_effect_where_ols_is_confounded(self):
        y, d, z, x, _ = confounded_data()
        estimate = two_stage_least_squares(y, d, z, x)
        naive = np.linalg.lstsq(np.column_stack([d, np.ones_like(d), x]), y, rcond=None)[0][0]
        self.assertAlmostEqual(estimate, -2.0, delta=0.25)
        self.assertLess(naive, -2.4)                                   # confounded by severity

    def test_cluster_bootstrap_covers_the_truth_and_first_stage_is_strong(self):
        y, d, z, x, cluster = confounded_data(seed=1)
        result = cluster_bootstrap(lambda rows: two_stage_least_squares(y[rows], d[rows], z[rows], x[rows]),
                                   cluster, replicates=200, seed=0)
        self.assertLess(result.low, -2.0)
        self.assertGreater(result.high, -2.0)
        self.assertGreater(first_stage_f(d, z, x), 10.0)

    def test_null_effect_interval_includes_zero(self):
        y, d, z, x, cluster = confounded_data(effect=0.0, seed=2)
        result = cluster_bootstrap(lambda rows: two_stage_least_squares(y[rows], d[rows], z[rows], x[rows]),
                                   cluster, replicates=200, seed=0)
        self.assertLess(result.low, 0.0)
        self.assertGreater(result.high, 0.0)

    def test_standardized_mean_difference(self):
        self.assertAlmostEqual(standardized_mean_difference([1.0, 2.0, 3.0, 2.0, 3.0, 4.0], [0, 0, 0, 1, 1, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
