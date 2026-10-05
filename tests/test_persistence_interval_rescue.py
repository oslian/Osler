import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.audit_persistence_interval_rescue import audit_conditional_interval, scale_features
from scripts.promote_interval_rescue import build_selector
from whole_body_jepa.source_selection import MULTI, PERSISTENCE


class PersistenceIntervalRescueTests(unittest.TestCase):
    def test_scale_features_use_no_future_truth_and_keep_zero(self):
        frame = pd.DataFrame({"current": [0., 2.], MULTI: [1., 1.], "truth": [10., 3.]})
        expected = np.asarray([[0., 1.], [2., 1.]])
        np.testing.assert_array_equal(scale_features(frame, MULTI), expected)
        frame["truth"] = -9999
        np.testing.assert_array_equal(scale_features(frame, MULTI), expected)

    def test_external_outcomes_cannot_change_scale_or_quantile(self):
        n = 1200
        frame = pd.DataFrame({"subject_id": [f"mimic:{i}" for i in range(n)], "stay_id": np.arange(n),
                              "t": pd.date_range("2020-01-01", periods=n, freq="h"),
                              "careunit": ["a", "b"] * (n // 2), "current": np.linspace(0, 5, n),
                              MULTI: np.linspace(0, 5, n) + .1})
        frame["truth"] = frame[MULTI] + np.linspace(.1, 1, n)
        external = frame.iloc[:400].copy()
        external["subject_id"] = "nwicu:" + external["subject_id"]
        selection = {"selected_source": MULTI, "incumbent_source": PERSISTENCE}
        points = {db: {"point": {"pass": True}} for db in ("mimic_test", "nwicu")}
        with patch("scripts.audit_persistence_interval_rescue.validate_selected", return_value=points):
            before = audit_conditional_interval(frame, external, selection, "test@1h")
            external["truth"] += 1000
            after = audit_conditional_interval(frame, external, selection, "test@1h")
        self.assertEqual(before["normalized_radius"], after["normalized_radius"])
        self.assertEqual(before["scale_model"], after["scale_model"])
        self.assertFalse(after["pass"])

    def test_promotion_cannot_bypass_point_or_coverage_gate(self):
        gates = {k: {"pass": True} for k in ("point", "coverage", "careunit_coverage", "late_coverage")}
        parent = {"release_binding": {}, "cells": {"x@1h": {
            "target": "x", "can_move": False, "incumbent_source": PERSISTENCE,
            "source_model": PERSISTENCE, "source_changed": False, "selected_source": MULTI,
            "selection": {"scores": {MULTI: {"eligible_challenger": True}}},
            "validation": {"mimic_test": gates, "nwicu": gates, "promotion_allowed": False},
        }}}
        result = {"selected_source": MULTI, "pass": True, "failure_reasons": [],
                  "mimic_test": gates, "nwicu": gates, "method": "normalized_current_predicted_delta",
                  "normalized_radius": 1., "scale_model": {}}
        report = {"release_binding": {}, "passed_cells": ["x@1h"], "cells": {"x@1h": result}, "parent_selector_sha256": "frozen"}
        promoted = build_selector(parent, report)
        self.assertTrue(promoted["cells"]["x@1h"]["can_move"])
        self.assertFalse(parent["cells"]["x@1h"]["can_move"])
        broken = copy.deepcopy(report)
        broken["cells"]["x@1h"]["nwicu"]["coverage"]["pass"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_selector(parent, broken)
        broken = copy.deepcopy(report)
        broken["cells"]["x@1h"]["selected_source"] = "some_other_head"
        with self.assertRaisesRegex(ValueError, "selected source"):
            build_selector(parent, broken)


if __name__ == "__main__":
    unittest.main()
