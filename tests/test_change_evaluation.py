import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_change_behavior import frozen_outputs, label_mask
from whole_body_jepa.change_evaluation import evaluate_change, estimate, interval_score
from whole_body_jepa.runtime import load_release, parse_patient_csv, _model_frame, _build_inputs, predict_patient


class ChangeEvaluationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"subject_id": ["a", "b", "c", "d"],
            "current": [0., 10., 10., 10.], "truth": [0., 10., 20., 0.],
            "point": [0., 20., 20., np.nan], "lower": [np.nan, 18., 18., np.nan],
            "upper": [np.nan, 22., 22., np.nan]})

    def test_false_moves_and_abstention_denominators(self):
        r = evaluate_change(self.frame(), 5, repetitions=20)
        self.assertEqual(r["stable"]["false_move_rate_on_stable"]["mean"], .5)
        self.assertEqual(r["changed"]["signed_change_recall_including_abstention"]["mean"], .5)
        self.assertEqual(r["changed"]["coverage_when_interval_available"]["mean"], 1)
        self.assertEqual(r["changed"]["covered_fraction_all_evaluable"]["mean"], .5)
        self.assertEqual(r["stable"]["rows"], 2)  # physical zero is observed
        self.assertIsNone(r["stable"]["mae"]["ci95"])
        self.assertFalse(r["promotion_allowed"])

    def test_persistence_not_awarded_perfect_change_performance(self):
        f = self.frame()
        f["point"] = f.current
        f[["lower", "upper"]] = np.nan
        r = evaluate_change(f, 5, repetitions=20)
        self.assertEqual(r["stable"]["false_move_rate_on_stable"]["mean"], 0)
        self.assertEqual(r["changed"]["signed_change_recall_including_abstention"]["mean"], 0)
        self.assertIsNone(r["all"]["coverage_when_interval_available"]["mean"])

    def test_patient_equalization(self):
        f = pd.DataFrame({"subject_id": ["a"] * 20 + ["b"]})
        self.assertEqual(estimate(f, [0.] * 20 + [10.])["mean"], 5.)

    def test_interval_score_penalizes_width_and_misses(self):
        np.testing.assert_allclose(interval_score([0., 3.], [-1., -1.], [1., 1.]), [2., 42.])
        self.assertGreater(interval_score([0.], [-100.], [100.])[0], 2)

    def test_future_strata_do_not_modify_predictions(self):
        f = self.frame()
        before = f.copy(deep=True)
        evaluate_change(f, 5, repetitions=20)
        pd.testing.assert_frame_equal(f, before)

    def test_bad_intervals_rejected(self):
        f = self.frame()
        f.loc[1, "lower"] = 30
        with self.assertRaises(ValueError):
            evaluate_change(f, 5)

    def test_future_labels_fail_closed(self):
        f = pd.DataFrame({"t": pd.to_datetime(["2020-01-01"] * 4),
            "x_t": [0.] * 4, "future_x": [1.] * 4,
            "future_label_quality_x": ["strict", "imputed", "strict", "strict"],
            "future_event_time_x": ["2020-01-02", "2020-01-02", "NaT", "2020-01-01"],
            "future_delta_t_hr_x": [24.] * 4})
        np.testing.assert_array_equal(label_mask(f, "x"), [True, False, False, False])
        self.assertFalse(label_mask(f.drop(columns="future_delta_t_hr_x"), "x").any())
        f["future_x"] = [None, "", "not_measured", None]
        self.assertFalse(label_mask(f, "x").any())

    def test_frozen_batch_matches_runtime_all_numeric_cells(self):
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]
        assets = load_release(root / "models")
        text = (root / "examples/patient_2.csv").read_text()
        variables = tuple(assets.checkpoint["variables"])
        frame = _model_frame(parse_patient_csv(text, variables), variables, assets.max_age_hours, "test")
        results = predict_patient(text, model_dir=root / "models")
        for cell in results["cells"]:
            if cell["target"] == "hypoxemia_event":
                continue
            arrays = _build_inputs(frame, assets, cell["horizon_hours"])
            out = frozen_outputs(assets, arrays, frame, np.array([len(frame) - 1]), cell["cell"]).iloc[0]
            for a, b in (("point", "forecast_point"), ("lower", "lower"), ("upper", "upper")):
                expected = np.nan if cell[b] is None else cell[b]
                np.testing.assert_allclose(out[a], expected, atol=1e-5, rtol=1e-6, equal_nan=True,
                                           err_msg=cell["cell"] + ":" + a)


if __name__ == "__main__":
    unittest.main()
