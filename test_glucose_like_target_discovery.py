import unittest

import numpy as np

from glucose_like_target_discovery import rank_glucose_like_targets


class GlucoseLikeTargetDiscoveryTests(unittest.TestCase):
    def test_ranking_recovers_target_with_glucose_like_descriptor(self):
        descriptors = {
            "glucose": {"density": 0.8, "delta": 0.4, "corr": 0.7},
            "similar": {"density": 0.78, "delta": 0.42, "corr": 0.69},
            "slow": {"density": 0.1, "delta": 0.01, "corr": 0.99},
        }
        support = {
            name: {"future_rows": 500, "future_subjects": 100}
            for name in descriptors
        }
        selected, ranking, features = rank_glucose_like_targets(
            descriptors, support, top_k=2
        )
        self.assertEqual(selected, ["glucose", "similar"])
        self.assertEqual(features, ["corr", "delta", "density"])
        self.assertEqual(ranking[0]["target"], "glucose")

    def test_low_support_target_is_not_selected(self):
        descriptors = {
            "glucose": {"density": 0.8},
            "unsupported": {"density": 0.8},
        }
        support = {
            "glucose": {"future_rows": 500, "future_subjects": 100},
            "unsupported": {"future_rows": 10, "future_subjects": 3},
        }
        selected, _, _ = rank_glucose_like_targets(
            descriptors, support, top_k=2
        )
        self.assertEqual(selected, ["glucose"])


if __name__ == "__main__":
    unittest.main()
