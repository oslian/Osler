import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from whole_body_jepa.source_selection import (
    BASE, HEAD, MULTI, PERSISTENCE, BOUND_FILES, load_selector, patient_roles,
    predict_head_values, select_source, validate_selected,
)
from whole_body_jepa.multisource_intervals import sha256


class SourceSelectionTests(unittest.TestCase):
    def test_patient_roles_never_split_repeated_stays(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(200), 4)})
        roles = patient_roles(frame)
        self.assertTrue(np.all(sum(roles.values()) == 1))
        groups = [set(frame.loc[mask, "subject_id"]) for mask in roles.values()]
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                self.assertFalse(left & right)

    def test_shared_head_slices_and_source_normalization(self):
        head = {"feature_view": "shared", "feature_width": 257, "family": "ridge",
                "feature_median": np.zeros(257), "feature_scale": np.ones(257),
                "runtime_feature_median": np.ones(257), "runtime_feature_scale": np.ones(257),
                "coefficient": np.ones(257) / 257, "intercept": 0,
                "normalized_delta_limit": 10, "target_scale": 2, "shrinkage": 1}
        features = np.full((2, 311), 2.)
        features[:, 257:] = 9999
        np.testing.assert_allclose(predict_head_values(head, features, np.zeros(2), runtime=False), 4)
        np.testing.assert_allclose(predict_head_values(head, features, np.zeros(2), runtime=True), 2)
        with self.assertRaisesRegex(ValueError, "311"):
            predict_head_values(head, features[:, :257], np.zeros(2), runtime=False)

    def test_original_runtime_normalization_keeps_float64_arithmetic(self):
        rng = np.random.default_rng(4)
        features = rng.normal(size=(3, 311)).astype(np.float32)
        median = rng.normal(size=311).astype(np.float32)
        scale = rng.uniform(.1, 1, 311).astype(np.float32)
        coefficient = rng.normal(size=311)
        head = {"feature_width": 311, "family": "ridge", "runtime_feature_median": median,
                "runtime_feature_scale": scale, "coefficient": coefficient, "intercept": .31,
                "normalized_delta_limit": 1e6, "target_scale": 2., "shrinkage": .5}
        expected = ((features - median.astype(float)) / scale.astype(float)) @ coefficient + .31
        np.testing.assert_array_equal(predict_head_values(head, features, np.zeros(3), runtime=True), expected)

    def test_selection_uses_only_development_not_external_ranking(self):
        frame = pd.DataFrame({"subject_id": np.arange(500).astype(str), "truth": 2.,
                              PERSISTENCE: 0., BASE: 1., HEAD: 1.5, MULTI: 1.9})
        result = select_source(frame, BASE, "synthetic@1h")
        self.assertEqual(result["selected_source"], MULTI)
        frame[MULTI] = 0
        self.assertEqual(select_source(frame, BASE, "synthetic@1h")["selected_source"], HEAD)

    def test_no_significant_gain_retains_incumbent(self):
        frame = pd.DataFrame({"subject_id": np.arange(250).astype(str), "truth": 1.,
                              PERSISTENCE: 0., BASE: .5, MULTI: .5})
        self.assertEqual(select_source(frame, BASE, "same@1h")["selected_source"], BASE)

    def test_support_failure_cannot_activate_source(self):
        frame = pd.DataFrame({"subject_id": ["a", "b"], "truth": [1., 2.], BASE: [1., 1.]})
        result = validate_selected(frame, pd.DataFrame({"subject_id": ["external"]}),
                                   {"selected_source": BASE, "incumbent_source": BASE}, "a@1h")
        self.assertFalse(result["promotion_allowed"])
        self.assertIsNone(result["radius"])

    def test_selector_binding_and_validation_cannot_be_bypassed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in BOUND_FILES:
                (root / name).write_bytes(b"frozen")
            payload = {"schema": "whole_body_jepa.unified_source_selector.v1",
                       "release_binding": {name: sha256(root / name) for name in BOUND_FILES},
                       "cells": {"x": {"source_changed": True, "use_selector_interval": True,
                                         "interval_method": "constant",
                                         "source_model": MULTI, "incumbent_source": BASE, "can_move": True,
                                         "radius": 1., "validation": {"promotion_allowed": True}}}}
            path = root / "selector.json"
            path.write_text(json.dumps(payload))
            load_selector(path, root)
            payload["cells"]["x"]["validation"]["promotion_allowed"] = False
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "unvalidated"):
                load_selector(path, root)
            payload["cells"] = {}
            path.write_text(json.dumps(payload))
            (root / BOUND_FILES[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_selector(path, root)


if __name__ == "__main__":
    unittest.main()
