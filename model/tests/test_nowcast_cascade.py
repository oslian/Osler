import unittest

import numpy as np
import pandas as pd

from scripts.audit_nowcast_forecast_cascade import _select_residual_shrinkage


class CascadeShrinkageTests(unittest.TestCase):
    def test_selects_full_residual_when_it_recovers_truth(self) -> None:
        rows = pd.DataFrame(
            {
                "subject_id": [f"p{index}" for index in range(30)],
                "truth": np.linspace(10.0, 20.0, 30),
            }
        )
        nowcast = rows["truth"].to_numpy() + 2.0
        raw = {
            "rows": rows,
            "nowcast_persistence": nowcast,
            "nowcast_fed_jepa": rows["truth"].to_numpy(),
        }
        weight, evaluation = _select_residual_shrinkage(raw)
        self.assertEqual(weight, 1.0)
        self.assertTrue(evaluation.any())

    def test_rejects_residual_when_nowcast_is_already_correct(self) -> None:
        rows = pd.DataFrame(
            {
                "subject_id": [f"p{index}" for index in range(30)],
                "truth": np.linspace(10.0, 20.0, 30),
            }
        )
        nowcast = rows["truth"].to_numpy()
        raw = {
            "rows": rows,
            "nowcast_persistence": nowcast,
            "nowcast_fed_jepa": nowcast + 2.0,
        }
        weight, _ = _select_residual_shrinkage(raw)
        self.assertEqual(weight, 0.0)


if __name__ == "__main__":
    unittest.main()
