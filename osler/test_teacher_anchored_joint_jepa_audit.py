import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from teacher_anchored_joint_jepa_audit import (
    DEFAULT_VARIANTS,
    TeacherAnchoredResidualAdapter,
    _adapter_batches,
    _adaptive_patient_cluster_conformal_quantile,
    _adaptive_conformal_quantile,
    _crossfit_adaptive_patient_cluster_conformal_quantile,
    _crossfit_normalized_conformal,
    _compose_coupling_path_prediction,
    _distributional_adaptive_widths,
    _distributional_mondrian_widths,
    _distributional_asymmetric_widths,
    _distributional_asymmetric_adaptive_widths,
    _distributional_shape_adaptive_widths,
    _future_label_support,
    _fixed_expert_anchor_prediction,
    _feature_batch,
    _latent_svd_stability,
    _patient_cluster_conformal_quantile,
    _patient_coverage_diagnostics,
    _pcgrad,
    _sparsemax,
    _select_targets,
    _split_rows_for_mode,
    _summarize_organ_coupling,
    _train_distributional_scale_heads,
)


class TeacherAnchoredJointJEPAAuditTests(unittest.TestCase):
    def test_adapter_domain_balancing_equalizes_group_sampling(self):
        rows = np.arange(1000, dtype=np.int64)
        labels = np.asarray(["large"] * 950 + ["small"] * 50)
        batches = list(
            _adapter_batches(
                rows,
                batch_size=100,
                rng=np.random.default_rng(17),
                balance_labels=labels,
            )
        )
        sampled = np.concatenate(batches)
        small_fraction = float((labels[sampled] == "small").mean())
        self.assertGreater(small_fraction, 0.42)
        self.assertLess(small_fraction, 0.58)

    def test_compressed_feature_batch_restores_float32(self):
        features = {
            "context": torch.tensor(
                [[1.125, -2.25], [3.5, 4.75]], dtype=torch.float16
            ),
            "presence": torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8),
        }
        context = _feature_batch(features, "context", np.asarray([1, 0]))
        presence = _feature_batch(features, "presence", np.asarray([0, 1]))
        self.assertEqual(context.dtype, torch.float32)
        self.assertEqual(presence.dtype, torch.float32)
        torch.testing.assert_close(
            context,
            torch.tensor([[3.5, 4.75], [1.125, -2.25]]),
        )
        torch.testing.assert_close(
            presence,
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

    def test_canonical_default_uses_only_adjacent_three_level(self):
        self.assertEqual(DEFAULT_VARIANTS, ("adjacent_three_level",))

    def test_external_split_modes_isolate_requested_provenance(self):
        frame = pd.DataFrame(
            {
                "subject_id": np.repeat(np.arange(12), 4),
                "hospitalid": np.repeat(np.arange(4), 12),
                "careunit": np.tile(np.repeat(("micu", "sicu"), 2), 12),
                "stay_id": np.repeat(np.arange(12), 4),
                "_time_hours": np.tile((0.0, 1.0, 2.0, 3.0), 12),
            }
        )
        for mode, column in (
            ("subject", "subject_id"),
            ("hospital", "hospitalid"),
            ("careunit", "careunit"),
        ):
            train, test = _split_rows_for_mode(frame, mode, seed=7)
            self.assertFalse(
                set(frame.iloc[train][column])
                & set(frame.iloc[test][column])
            )

        train, test = _split_rows_for_mode(frame, "time", seed=7)
        for stay_id in frame["stay_id"].unique():
            train_time = frame.iloc[train]
            train_time = train_time[train_time["stay_id"] == stay_id]
            test_time = frame.iloc[test]
            test_time = test_time[test_time["stay_id"] == stay_id]
            self.assertLess(
                float(train_time["_time_hours"].max()),
                float(test_time["_time_hours"].min()),
            )

        with self.assertRaises(ValueError):
            _split_rows_for_mode(frame, "invalid", seed=7)

    def test_database_split_isolates_databases_and_patients(self):
        frame = pd.DataFrame(
            {
                "subject_id": ("M:1", "M:2", "E:1", "E:2"),
                "database": ("MIMIC-IV", "MIMIC-IV", "eICU", "eICU"),
            }
        )
        train, test = _split_rows_for_mode(
            frame,
            "database",
            seed=7,
            train_database="MIMIC-IV",
            test_database="eICU",
        )
        self.assertEqual(set(train), {0, 1})
        self.assertEqual(set(test), {2, 3})
        self.assertFalse(
            set(frame.iloc[train]["subject_id"])
            & set(frame.iloc[test]["subject_id"])
        )

    def test_fixed_expert_anchor_is_exact_target_horizon_only(self):
        frame = pd.DataFrame(
            {
                "expert__heart_rate__6h": (80.0, 90.0, 100.0),
                "expert__hypoxemia_event__1h": (0.1, 0.2, 0.3),
            }
        )
        arrays = [None] * 9
        arrays[0] = SimpleNamespace(
            medians=np.asarray([50.0, 4.0]),
            scales=np.asarray([10.0, 1.0]),
        )
        arrays[8] = np.asarray([6.0, 12.0, 6.0], dtype=np.float32)
        registry = {
            "cells": [
                {
                    "cell": "heart_rate@6h",
                    "target": "heart_rate",
                    "horizon_hours": 6,
                    "anchor_column": "expert__heart_rate__6h",
                    "task_type": "continuous_value",
                    "source_model": "measurement_pure_ridge",
                },
                {
                    "cell": "heart_rate@12h",
                    "target": "heart_rate",
                    "horizon_hours": 12,
                    "anchor_column": "expert__heart_rate__6h",
                    "task_type": "continuous_value_with_change_hurdle",
                    "source_model": "measurement_pure_ridge",
                },
                {
                    "cell": "hypoxemia_event@1h",
                    "target": "hypoxemia_event",
                    "horizon_hours": 1,
                    "anchor_column": "expert__hypoxemia_event__1h",
                    "task_type": "binary_event_probability",
                },
            ]
        }
        output, metadata = _fixed_expert_anchor_prediction(
            frame,
            ("heart_rate", "potassium"),
            tuple(arrays),
            np.arange(3),
            registry,
        )
        np.testing.assert_allclose(output[[0, 2], 0], (3.0, 5.0))
        self.assertAlmostEqual(float(output[1, 0]), 4.0)
        self.assertTrue(np.isnan(output[:, 1]).all())
        self.assertEqual(set(metadata), {"heart_rate@6h", "heart_rate@12h"})

    def test_all_target_selection_preserves_canonical_variable_order(self):
        variables = ("glucose", "creatinine", "heart_rate")
        self.assertEqual(_select_targets("all", variables), variables)
        self.assertEqual(
            _select_targets("heart_rate,missing,glucose", variables),
            ("heart_rate", "glucose"),
        )

    def test_future_label_support_records_unsupported_cells(self):
        frame = pd.DataFrame(
            {
                "subject_id": np.repeat(np.arange(25), 2),
                "horizon_hours": np.tile([3, 6], 25),
                "future_glucose": np.arange(50, dtype=float),
                "future_cortisol": np.nan,
            }
        )
        report = _future_label_support(
            frame, ("glucose", "cortisol"), (3, 6), minimum_rows=20
        )
        self.assertEqual(report["glucose"]["evaluable_horizons"], ["3h", "6h"])
        self.assertEqual(
            report["cortisol"]["support_limited_horizons"], ["3h", "6h"]
        )

    def test_organ_coupling_summary_reports_support_and_signal(self):
        weights = np.zeros((2, 3, 3), dtype=np.float32)
        weights[0, 0, 1] = 0.75
        weights[0, 1, 0] = 1.0
        presence = np.asarray(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32
        )
        signal = np.zeros((2, 1, 4), dtype=np.float32)
        signal[0, 0, 0] = 0.5
        report = _summarize_organ_coupling(
            weights,
            presence,
            signal,
            np.asarray(
                [[[0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0]]],
                dtype=np.float32,
            ),
            ("map",),
            (
                "heart",
                "kidneys",
                "brain",
            ),
            coupling_gates=np.asarray([0.4, 0.6, 0.5]),
        )
        self.assertTrue(report["available"])
        self.assertEqual(len(report["edges"]), 1)
        edge = report["edges"][0]
        self.assertEqual(edge["coobserved_rows"], 1)
        self.assertAlmostEqual(edge["coobserved_fraction"], 0.5)
        self.assertAlmostEqual(edge["left_receives_from_right"], 0.75)
        self.assertAlmostEqual(edge["right_receives_from_left"], 1.0)
        self.assertAlmostEqual(
            report["target_signal"]["map"]["nonzero_fraction"], 0.5
        )

    def test_coupling_path_composition_uses_cell_specific_selection(self):
        arrays = (None,) * 8 + (np.asarray([3.0, 12.0, 3.0, 12.0]),)
        paths = {
            "system_only": np.zeros((4, 1), dtype=np.float32),
            "direct_organ": np.ones((4, 1), dtype=np.float32),
            "body_mediated": np.full((4, 1), 2.0, dtype=np.float32),
            "full": np.full((4, 1), 3.0, dtype=np.float32),
        }
        output = _compose_coupling_path_prediction(
            np.arange(4),
            arrays,
            ("creatinine",),
            paths,
            {
                "creatinine@3h": {"selected_path": "direct_organ"},
                "creatinine@12h": {"selected_path": "body_mediated"},
            },
        )
        np.testing.assert_array_equal(
            output[:, 0], np.asarray([1.0, 2.0, 1.0, 2.0])
        )

    def test_distributional_mondrian_uses_calibration_state_only(self):
        subjects = np.repeat(np.arange(60), 4)
        frame = pd.DataFrame({"subject_id": subjects})
        calibration_rows = np.arange(180)
        test_rows = np.arange(180, 240)
        current = np.linspace(-2.0, 2.0, 240)
        residual = 0.5 + np.abs(current[:180])
        widths, metadata = _distributional_mondrian_widths(
            frame,
            calibration_rows,
            test_rows,
            residual,
            np.ones(180),
            np.ones(60),
            current[:180],
            current[180:],
            np.ones(180, dtype=bool),
            np.ones(60, dtype=bool),
        )
        self.assertEqual(widths.shape, (60,))
        self.assertTrue(np.all(widths > 0.0))
        self.assertTrue(metadata["cutpoints_fit_on_calibration_only"])
        self.assertFalse(metadata["test_labels_used"])
        self.assertGreaterEqual(len(metadata["strata"]), 1)

    def test_distributional_adaptive_uses_disjoint_calibration_patients(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(90), 4)})
        rows = np.arange(360)
        rng = np.random.default_rng(41)
        residual = np.abs(rng.normal(0.0, 1.0, size=360))
        widths, metadata = _distributional_adaptive_widths(
            frame,
            rows,
            residual,
            np.ones(360),
            np.linspace(0.8, 1.2, 24),
            seed=43,
        )
        self.assertEqual(widths.shape, (24,))
        self.assertTrue(np.all(widths > 0.0))
        self.assertTrue(metadata["patient_disjoint_final_calibration"])
        self.assertFalse(metadata["test_labels_used"])
        self.assertFalse(metadata["scale_head_test_labels_used"])
        self.assertGreaterEqual(metadata["selected_nominal_coverage"], 0.75)
        self.assertLessEqual(metadata["selected_nominal_coverage"], 0.90)

    def test_distributional_asymmetric_calibrates_tails_separately(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(30), 4)})
        rows = np.arange(120)
        lower, upper, metadata = _distributional_asymmetric_widths(
            frame,
            rows,
            np.linspace(-0.2, 0.8, 120),
            np.linspace(-0.1, 2.0, 120),
            np.ones(12),
        )
        self.assertEqual(lower.shape, (12,))
        self.assertEqual(upper.shape, (12,))
        self.assertTrue(np.all(upper > lower))
        self.assertFalse(metadata["test_labels_used"])
        self.assertAlmostEqual(metadata["tail_coverage"], 0.95)

    def test_distributional_asymmetric_adaptive_uses_patient_crossfit(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(80), 4)})
        rows = np.arange(320)
        rng = np.random.default_rng(29)
        error = rng.standard_t(df=4, size=320) + 0.35
        lower, upper, metadata = (
            _distributional_asymmetric_adaptive_widths(
                frame,
                rows,
                error,
                np.ones(40),
                seed=31,
            )
        )
        self.assertEqual(lower.shape, (40,))
        self.assertEqual(upper.shape, (40,))
        self.assertTrue(np.all(lower > 0.0))
        self.assertTrue(np.all(upper > 0.0))
        self.assertTrue(metadata["all_calibration_rows_used_after_crossfit_selection"])
        self.assertFalse(metadata["test_labels_used"])
        self.assertGreaterEqual(metadata["selected_nominal_coverage"], 0.80)
        self.assertLessEqual(metadata["selected_nominal_coverage"], 0.91)
        self.assertAlmostEqual(
            metadata["tail_coverage"],
            1.0
            - (1.0 - metadata["selected_nominal_coverage"]) / 2.0,
        )

    def test_distributional_shape_selection_keeps_final_patients_disjoint(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(80), 4)})
        rows = np.arange(320)
        rng = np.random.default_rng(43)
        error = rng.normal(0.4, 0.6, size=320)
        lower, upper, metadata = _distributional_shape_adaptive_widths(
            frame,
            rows,
            error,
            np.ones(320),
            np.ones(20),
            seed=47,
        )
        self.assertEqual(lower.shape, (20,))
        self.assertEqual(upper.shape, (20,))
        self.assertTrue(metadata["selection_final_patient_disjoint"])
        self.assertFalse(metadata["test_labels_used"])
        self.assertIn(metadata["selected_shape"], ("symmetric", "asymmetric"))

    def _adapter(
        self,
        dynamic=True,
        sparse=False,
        distributional=False,
        target_specific=False,
    ):
        return TeacherAnchoredResidualAdapter(
            target_indices=(0, 1),
            target_module_membership=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            latent_dim=4,
            organ_dim=3,
            module_count=3,
            adapter_dim=8,
            dynamic_organ_route=dynamic,
            sparse_experts=sparse,
            distributional_scale=distributional,
            target_specific_projections=target_specific,
        )

    def _inputs(self):
        return {
            "teacher_anchor": torch.tensor([[0.2, -0.4], [0.3, -0.2]]),
            "context_latent": torch.randn(2, 4),
            "target_latent": torch.randn(2, 2, 4),
            "organ_latents": torch.randn(2, 3, 3),
            "organ_presence": torch.ones(2, 3),
            "current": torch.randn(2, 2),
            "current_mask": torch.ones(2, 2),
            "ages": torch.zeros(2, 2),
            "horizon": torch.tensor([6.0, 12.0]),
        }

    def test_zero_initialized_adapter_is_exact_teacher_anchor(self):
        adapter = self._adapter(dynamic=True)
        output = adapter(**self._inputs())
        torch.testing.assert_close(
            output["value"], self._inputs()["teacher_anchor"]
        )
        torch.testing.assert_close(
            output["residual"], torch.zeros_like(output["residual"])
        )

    def test_dynamic_route_excludes_targets_own_organ(self):
        adapter = self._adapter(dynamic=True)
        output = adapter(**self._inputs())
        self.assertTrue(
            torch.all(output["route_weights"][:, 0, 0] == 0.0)
        )
        self.assertTrue(
            torch.all(output["route_weights"][:, 1, 1] == 0.0)
        )
        torch.testing.assert_close(
            output["route_weights"].sum(dim=-1),
            torch.ones(2, 2),
        )

    def test_disabling_route_preserves_teacher_at_initialization(self):
        adapter = self._adapter(dynamic=True)
        output = adapter(**self._inputs(), disable_organ_route=True)
        torch.testing.assert_close(
            output["value"], self._inputs()["teacher_anchor"]
        )
        torch.testing.assert_close(
            output["route_weights"],
            torch.zeros_like(output["route_weights"]),
        )

    def test_sparsemax_produces_exact_sparse_simplex_weights(self):
        weights = _sparsemax(
            torch.tensor([[3.0, 1.0, -2.0], [0.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
        self.assertTrue(torch.any(weights == 0.0))
        self.assertTrue(torch.all(weights >= 0.0))

    def test_sparse_experts_are_target_specific_and_normalized(self):
        adapter = self._adapter(dynamic=False, sparse=True)
        output = adapter(**self._inputs())
        torch.testing.assert_close(
            output["expert_weights"].sum(dim=-1), torch.ones(2, 2)
        )
        self.assertTrue(torch.all(output["expert_weights"] >= 0.0))
        torch.testing.assert_close(
            output["value"], self._inputs()["teacher_anchor"]
        )

    def test_sparse_expert_masks_missing_own_organ(self):
        adapter = self._adapter(dynamic=False, sparse=True)
        inputs = self._inputs()
        inputs["organ_presence"][:, 0] = 0.0
        output = adapter(**inputs)
        self.assertTrue(
            torch.all(output["expert_weights"][:, 0, 2] == 0.0)
        )

    def test_target_specific_projection_has_matched_disable_path(self):
        adapter = self._adapter(
            dynamic=False,
            sparse=True,
            target_specific=True,
        )
        inputs = self._inputs()
        initial = adapter(**inputs)
        disabled = adapter(
            **inputs, disable_target_specific_projections=True
        )
        torch.testing.assert_close(initial["value"], disabled["value"])
        with torch.no_grad():
            adapter.target_fast_adapters[0][-1].bias.fill_(0.5)
            adapter.residual_heads[0][-1].weight.fill_(0.1)
        enabled = adapter(**inputs)
        disabled = adapter(
            **inputs, disable_target_specific_projections=True
        )
        self.assertFalse(torch.equal(enabled["value"], disabled["value"]))

    def test_disabling_sparse_experts_restores_hierarchical_mixture(self):
        adapter = self._adapter(dynamic=False, sparse=True)
        output = adapter(**self._inputs(), disable_sparse_experts=True)
        torch.testing.assert_close(
            output["expert_weights"][..., 0]
            + output["expert_weights"][..., 1],
            torch.ones(2, 2),
        )
        self.assertTrue(
            torch.all(output["expert_weights"][..., 2] == 0.0)
        )

    def test_distributional_head_starts_positive_without_moving_teacher(self):
        adapter = self._adapter(
            dynamic=False, sparse=True, distributional=True
        )
        inputs = self._inputs()
        output = adapter(**inputs)
        torch.testing.assert_close(output["value"], inputs["teacher_anchor"])
        self.assertTrue(torch.all(output["scale"] > 0.0))
        torch.testing.assert_close(
            output["scale"],
            torch.full_like(output["scale"], 0.50),
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    def test_distributional_loss_does_not_backprop_into_scale_inputs(self):
        adapter = self._adapter(
            dynamic=False, sparse=True, distributional=True
        )
        inputs = self._inputs()
        inputs["context_latent"].requires_grad_(True)
        output = adapter(**inputs)
        output["scale"].sum().backward()
        gradient = inputs["context_latent"].grad
        self.assertTrue(gradient is None or torch.all(gradient == 0.0))

    def test_scale_stage_does_not_change_point_predictor(self):
        torch.manual_seed(59)
        adapter = self._adapter(
            dynamic=False, sparse=True, distributional=True
        )
        rows = 24
        arrays = (
            None,
            None,
            np.zeros((rows, 2), dtype=np.float32),
            np.ones((rows, 2), dtype=np.float32),
            np.zeros((rows, 2), dtype=np.float32),
            np.random.default_rng(61).normal(size=(rows, 2)).astype(
                np.float32
            ),
            np.ones((rows, 2), dtype=np.float32),
            None,
            np.full(rows, 6.0, dtype=np.float32),
        )
        features = {
            "context": torch.randn(rows, 4),
            "target": torch.randn(rows, 2, 4),
            "organ": torch.randn(rows, 3, 3),
            "presence": torch.ones(rows, 3),
        }
        teacher = np.zeros((rows, 2), dtype=np.float32)
        point_before = {
            name: value.detach().clone()
            for name, value in adapter.named_parameters()
            if not name.startswith("scale_heads.")
        }
        scale_before = {
            name: value.detach().clone()
            for name, value in adapter.named_parameters()
            if name.startswith("scale_heads.")
        }
        report = _train_distributional_scale_heads(
            adapter,
            features,
            arrays,
            teacher,
            np.arange(rows),
            (0, 1),
            epochs=2,
            batch_size=8,
            seed=67,
        )
        self.assertTrue(report["point_predictor_frozen"])
        for name, value in adapter.named_parameters():
            if name in point_before:
                torch.testing.assert_close(value, point_before[name])
        self.assertTrue(
            any(
                not torch.equal(value, scale_before[name])
                for name, value in adapter.named_parameters()
                if name in scale_before
            )
        )

    def test_pcgrad_projects_a_direct_conflict(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        left = parameter[0] + parameter[1]
        right = -parameter[0] + 0.25 * parameter[1]
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        before, after = _pcgrad(
            [left, right],
            [parameter],
            np.random.default_rng(7),
        )
        self.assertTrue(any(value < 0.0 for value in before))
        self.assertTrue(all(value >= -1e-6 for value in after))
        self.assertIsNotNone(parameter.grad)

    def test_latent_svd_detects_stable_noncollapsed_modes(self):
        rng = np.random.default_rng(11)
        latent = rng.normal(size=(240, 3))
        projection = rng.normal(size=(3, 12))
        values = latent @ projection + rng.normal(scale=0.01, size=(240, 12))
        features = {
            "context": torch.from_numpy(values[:, :4]).float(),
            "target": torch.from_numpy(values[:, 4:].reshape(240, 2, 4)).float(),
        }
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(60), 4)})
        report = _latent_svd_stability(
            frame,
            features,
            np.arange(0, 120),
            np.arange(120, 240),
            max_rows=80,
            seed=11,
        )
        self.assertTrue(report["pass"])
        self.assertGreater(report["fit_effective_rank"], 2.0)
        self.assertGreater(report["mean_top_subspace_cosine"], 0.5)
        self.assertLessEqual(report["fit_sample_rows"], 80)
        self.assertTrue(report["patient_equalized_sampling"])

    def test_adaptive_conformal_uses_patient_disjoint_calibration(self):
        subject_ids = np.repeat(np.arange(30), 5)
        frame = pd.DataFrame({"subject_id": subject_ids})
        residuals = (
            np.linspace(0.05, 2.0, len(subject_ids))
            + (subject_ids % 3) * 0.01
        )
        quantile, rank, metadata = _adaptive_conformal_quantile(
            frame,
            np.arange(len(frame)),
            residuals,
            seed=17,
        )
        self.assertGreater(quantile, 0.0)
        self.assertGreater(rank, 0)
        self.assertEqual(
            metadata["method"],
            "patient_grouped_cv_adaptive_split_conformal",
        )
        self.assertTrue(
            metadata["patient_disjoint_final_calibration"]
        )
        self.assertFalse(metadata["test_labels_used"])
        self.assertGreater(metadata["tuning_subjects"], 0)
        self.assertGreater(
            metadata["final_calibration_subjects"], 0
        )
        self.assertLessEqual(
            metadata["selected_nominal_coverage"], 0.90
        )

    def test_adaptive_conformal_falls_back_when_support_is_small(self):
        frame = pd.DataFrame({"subject_id": np.arange(20)})
        quantile, rank, metadata = _adaptive_conformal_quantile(
            frame,
            np.arange(20),
            np.linspace(0.1, 1.0, 20),
            seed=19,
        )
        self.assertGreater(quantile, 0.0)
        self.assertGreater(rank, 0)
        self.assertEqual(
            metadata["method"], "fixed_split_conformal_fallback"
        )

    def test_normalized_conformal_crossfits_scale_by_patient(self):
        rng = np.random.default_rng(23)
        subject_ids = np.repeat(np.arange(40), 5)
        feature = rng.normal(size=(len(subject_ids), 3))
        residuals = np.abs(
            rng.normal(
                scale=0.5 + 0.8 * np.abs(feature[:, 0]),
                size=len(subject_ids),
            )
        )
        test_features = np.asarray(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        widths, metadata = _crossfit_normalized_conformal(
            pd.DataFrame({"subject_id": subject_ids}),
            np.arange(len(subject_ids)),
            residuals,
            feature,
            test_features,
            seed=29,
        )
        self.assertEqual(
            metadata["method"],
            "patient_grouped_crossfit_normalized_conformal",
        )
        self.assertEqual(widths.shape, (2,))
        self.assertTrue(np.all(widths > 0.0))
        self.assertFalse(metadata["test_labels_used"])
        self.assertTrue(metadata["patient_grouped_scale_crossfit"])

    def test_patient_cluster_conformal_equalizes_long_stays(self):
        frame = pd.DataFrame(
            {
                "subject_id": (
                    ["dense"] * 100
                    + ["sparse_1"] * 2
                    + ["sparse_2"] * 2
                    + ["sparse_3"] * 2
                )
            }
        )
        residuals = np.asarray(
            [0.1] * 100 + [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        )
        quantile, rank, metadata = (
            _patient_cluster_conformal_quantile(
                frame,
                np.arange(len(frame)),
                residuals,
                coverage=0.75,
            )
        )
        self.assertGreaterEqual(quantile, 1.2)
        self.assertGreater(rank, 0)
        self.assertEqual(metadata["calibration_subjects"], 4)
        self.assertTrue(metadata["equal_total_weight_per_patient"])

    def test_patient_coverage_is_not_row_dominated(self):
        frame = pd.DataFrame(
            {"subject_id": ["dense"] * 100 + ["sparse"] * 2}
        )
        covered = np.asarray([True] * 100 + [False, False])
        report = _patient_coverage_diagnostics(
            frame,
            np.arange(len(frame)),
            covered,
            seed=31,
            bootstrap_samples=100,
        )
        self.assertGreater(report["row_coverage"], 0.95)
        self.assertAlmostEqual(
            report["patient_equalized_coverage"], 0.5
        )
        self.assertEqual(report["test_subjects"], 2)

    def test_adaptive_patient_cluster_uses_disjoint_final_patients(self):
        rng = np.random.default_rng(43)
        subjects = np.repeat(np.arange(60), 4)
        frame = pd.DataFrame({"subject_id": subjects})
        residuals = np.abs(rng.normal(size=len(frame)))
        quantile, rank, metadata = (
            _adaptive_patient_cluster_conformal_quantile(
                frame,
                np.arange(len(frame)),
                residuals,
                seed=47,
            )
        )
        self.assertGreater(quantile, 0.0)
        self.assertGreater(rank, 0)
        self.assertEqual(
            metadata["method"],
            "patient_grouped_cv_adaptive_patient_cluster_conformal",
        )
        self.assertTrue(metadata["patient_disjoint_final_calibration"])
        self.assertFalse(metadata["test_labels_used"])

    def test_crossfit_adaptive_patient_cluster_uses_full_calibration(self):
        frame = pd.DataFrame({"subject_id": np.repeat(np.arange(80), 4)})
        rows = np.arange(320)
        residuals = np.linspace(0.05, 2.0, len(rows))
        quantile, rank, metadata = (
            _crossfit_adaptive_patient_cluster_conformal_quantile(
                frame,
                rows,
                residuals,
                seed=41,
            )
        )
        self.assertGreater(quantile, 0.0)
        self.assertGreater(rank, 0)
        self.assertEqual(metadata["calibration_rows"], len(rows))
        self.assertTrue(
            metadata["all_calibration_rows_used_after_crossfit_selection"]
        )
        self.assertFalse(metadata["test_labels_used"])
        self.assertLessEqual(metadata["selected_nominal_coverage"], 0.90)


if __name__ == "__main__":
    unittest.main()
