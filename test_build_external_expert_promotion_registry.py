import unittest

from build_external_expert_promotion_registry import tier


def lockbox(full=True, evaluable=True):
    return {"evaluable": evaluable, "metrics": {"value_conditional_full_pass": full}}


def samples(full, point=1.0, evaluable=True):
    return {"evaluable": evaluable, "full_pass_share": full, "point_pass_share": point}


class PromotionTierTests(unittest.TestCase):
    def test_tiers_follow_multi_sample_robustness(self):
        self.assertEqual(tier(lockbox(), samples(0.8), samples(0.8)), "A")
        self.assertEqual(tier(lockbox(), samples(1.0), samples(0.6)), "B")
        self.assertEqual(tier(lockbox(), samples(0.6, point=1.0), samples(1.0)), "C")
        self.assertIsNone(tier(lockbox(), samples(0.6, point=0.6), samples(1.0)))

    def test_lockbox_failure_is_never_promoted(self):
        self.assertIsNone(tier(lockbox(full=False), samples(1.0), samples(1.0)))
        self.assertIsNone(tier(lockbox(evaluable=False), samples(1.0), samples(1.0)))

    def test_unevaluable_eicu_cell_cannot_reach_tier_a(self):
        self.assertEqual(tier(lockbox(), samples(1.0), samples(0.0, evaluable=False)), "B")


if __name__ == "__main__":
    unittest.main()
