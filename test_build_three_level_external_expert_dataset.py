import unittest

import numpy as np
import pandas as pd

from build_three_level_external_expert_dataset import (
    canonicalize_cell,
    validated_cells,
)


class ThreeLevelExternalExpertDatasetTests(unittest.TestCase):
    def test_change_hurdle_value_cell_remains_continuous_anchor(self):
        cells = validated_cells(
            {
                "cells": [
                    {
                        "cell": "hemoglobin@12h",
                        "target": "hemoglobin",
                        "horizon_hours": 12,
                        "task_type": "continuous_value_with_change_hurdle",
                        "source_model": "measurement_pure_ridge",
                        "full_external_pass": True,
                    }
                ]
            }
        )
        self.assertEqual(cells[0]["role"], "three_level_value_anchor")

    def test_canonical_cell_only_registers_measurements_with_age(self):
        frame = pd.DataFrame(
            {
                "subject_id": ("M:1", "E:1"),
                "stay_id": ("M:S", "E:S"),
                "anchor_time": pd.to_datetime(("2026-01-01", "2026-01-02")),
                "database": ("MIMIC-IV", "eICU"),
                "heart_rate_t": (80.0, 90.0),
                "heart_rate_age_hr": (0.0, 1.0),
                "aki_active_t": (1.0, 0.0),
                "future_heart_rate": (82.0, 88.0),
            }
        )
        output = canonicalize_cell(
            frame,
            target="heart_rate",
            horizon=6,
            anchor_column="expert__heart_rate__6h",
            prediction=np.asarray((81.0, 89.0)),
        )
        self.assertEqual(set(output["_source_horizon"]), {6.0})
        self.assertIn("_observed_heart_rate", output)
        self.assertNotIn("future_aki_active", output)


if __name__ == "__main__":
    unittest.main()
