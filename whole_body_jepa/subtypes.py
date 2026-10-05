"""Measurement-pure physiological subtypes for few-observation adaptation.

The subtype model is deliberately separate from the promoted forecast runtime.
It learns soft memberships from pre-anchor whole-body representations, then
estimates target-cell residual priors for each subtype.  A patient with only a
few resolved forecast errors borrows strength from that prior; as their own
outcomes accumulate, hierarchical shrinkage moves toward the personal mean.

No future target, target-specific future latent, hospital identifier, or care
unit identifier is used by the subtype router.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from osler_jepa.shared_release_features import FEATURE_SLICES, FEATURE_WIDTH


SUBTYPE_FEATURE_BLOCKS = (
    "shared_context",
    "patient_belief",
    "measurement_pure_state",
)
SUBTYPE_BASE_WIDTH = sum(
    FEATURE_SLICES[name].stop - FEATURE_SLICES[name].start
    for name in SUBTYPE_FEATURE_BLOCKS
)


def subtype_features(
    feature_matrix: np.ndarray,
    phenotype_covariates: np.ndarray | None = None,
) -> np.ndarray:
    """Select pre-anchor blocks and optional measurement-pure covariates."""

    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_WIDTH:
        raise ValueError(f"feature_matrix must have shape [rows, {FEATURE_WIDTH}]")
    if not np.isfinite(matrix).all():
        raise ValueError("feature_matrix must contain only finite values")
    selected = np.concatenate(
        [matrix[:, FEATURE_SLICES[name]] for name in SUBTYPE_FEATURE_BLOCKS],
        axis=1,
    )
    if phenotype_covariates is None:
        return selected
    covariates = np.asarray(phenotype_covariates, dtype=np.float64)
    if covariates.ndim == 1:
        covariates = covariates[None, :]
    if covariates.ndim != 2 or covariates.shape[0] != len(matrix):
        raise ValueError("phenotype_covariates must align with feature rows")
    if not np.isfinite(covariates).all():
        raise ValueError("phenotype_covariates must contain only finite values")
    return np.concatenate((selected, covariates), axis=1)


def causal_history_covariates(
    normalized_values: np.ndarray,
    *,
    observed_mask: np.ndarray,
    measurement_ages: np.ndarray,
    subject_ids: Sequence[object],
    time_hours: Sequence[float],
    window_hours: float,
) -> np.ndarray:
    """Build pre-anchor level, slope, variability, density, and age features."""

    return causal_history_covariates_at_anchors(
        normalized_values,
        observed_mask=observed_mask,
        measurement_ages=measurement_ages,
        history_group_ids=subject_ids,
        history_times=time_hours,
        anchor_group_ids=subject_ids,
        anchor_times=time_hours,
        window_hours=window_hours,
    )


def causal_history_covariates_at_anchors(
    normalized_values: np.ndarray,
    *,
    observed_mask: np.ndarray,
    measurement_ages: np.ndarray,
    history_group_ids: Sequence[object],
    history_times: Sequence[float],
    anchor_group_ids: Sequence[object],
    anchor_times: Sequence[float],
    window_hours: float,
) -> np.ndarray:
    """Summarize a complete pre-anchor stay history at selected anchors.

    Training rows may be patient-balanced after extraction.  History must be
    summarized before that sampling step; otherwise two retained rows are
    mistaken for the patient's full physiological trajectory.
    """

    values = np.asarray(normalized_values, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    ages = np.asarray(measurement_ages, dtype=np.float64)
    groups = np.asarray(history_group_ids, dtype=object)
    times = np.asarray(history_times, dtype=np.float64)
    anchor_groups = np.asarray(anchor_group_ids, dtype=object)
    anchors = np.asarray(anchor_times, dtype=np.float64)
    if values.ndim != 2 or observed.shape != values.shape or ages.shape != values.shape:
        raise ValueError("history values, mask, and ages must share a two-dimensional shape")
    if groups.shape != (len(values),) or times.shape != (len(values),):
        raise ValueError("history groups and times must align with history rows")
    if anchor_groups.shape != anchors.shape or anchor_groups.ndim != 1:
        raise ValueError("anchor groups and times must align")
    if window_hours <= 0.0 or not np.isfinite(times).all() or not np.isfinite(anchors).all():
        raise ValueError("window_hours must be positive and all times finite")

    width = values.shape[1]
    output = np.zeros((len(anchors), width * 5 + 2), dtype=np.float64)
    history_by_group: dict[object, tuple[np.ndarray, np.ndarray]] = {}
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(times[rows], kind="stable")]
        history_by_group[group] = (rows, times[rows])
    for row, (group, anchor_time) in enumerate(zip(anchor_groups, anchors)):
        group_history = history_by_group.get(group)
        if group_history is None:
            continue
        rows, ordered_time = group_history
        start = np.searchsorted(
            ordered_time,
            float(anchor_time) - float(window_hours),
            side="left",
        )
        stop = np.searchsorted(ordered_time, float(anchor_time), side="right")
        history_rows = rows[start:stop]
        if not len(history_rows):
            continue
        history_time = times[history_rows]
        history_values = values[history_rows]
        history_observed = observed[history_rows] & np.isfinite(history_values)
        level = np.zeros(width, dtype=np.float64)
        slope = np.zeros(width, dtype=np.float64)
        variability = np.zeros(width, dtype=np.float64)
        density = history_observed.mean(axis=0)
        for column in range(width):
            valid = history_observed[:, column]
            if not valid.any():
                continue
            series = history_values[valid, column]
            series_time = history_time[valid]
            level[column] = float(np.median(series))
            variability[column] = float(
                np.median(np.abs(series - level[column]))
            )
            if len(series) >= 2 and series_time[-1] > series_time[0]:
                slope[column] = float(
                    (series[-1] - series[0])
                    / (series_time[-1] - series_time[0])
                )
        last = history_rows[-1]
        current_age = np.where(
            observed[last] & np.isfinite(ages[last]),
            np.clip(ages[last] / float(window_hours), 0.0, 1.0),
            1.0,
        )
        output[row] = np.concatenate(
            (
                level,
                slope,
                variability,
                density,
                current_age,
                np.asarray(
                    [
                        np.log1p(len(history_rows)),
                        min(
                            1.0,
                            (history_time[-1] - history_time[0])
                            / float(window_hours),
                        ),
                    ]
                ),
            )
        )
    return output.astype(np.float32)


def causal_trajectory_features(
    feature_matrix: np.ndarray,
    *,
    subject_ids: Sequence[object],
    time_hours: Sequence[float],
    window_hours: float = 48.0,
    current_weight: float = 0.0,
) -> np.ndarray:
    """Blend subtype blocks with a causal within-patient trajectory median.

    At row ``t`` the aggregate can only use rows from the same patient in
    ``[t - window_hours, t]``. Target-specific and future blocks are preserved
    exactly, so this transformation changes subtype routing only. The current
    row contributes ``current_weight`` and the trailing median contributes the
    remaining weight.
    """

    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_WIDTH:
        raise ValueError(f"feature_matrix must have shape [rows, {FEATURE_WIDTH}]")
    subjects = np.asarray(subject_ids, dtype=object)
    times = np.asarray(time_hours, dtype=np.float64)
    if subjects.shape != (len(matrix),) or times.shape != (len(matrix),):
        raise ValueError("subject_ids and time_hours must align with feature rows")
    if not np.isfinite(times).all() or window_hours <= 0.0:
        raise ValueError("time_hours must be finite and window_hours positive")
    if not 0.0 <= current_weight <= 1.0:
        raise ValueError("current_weight must be between zero and one")

    output = matrix.copy()
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        rows = rows[np.argsort(times[rows], kind="stable")]
        ordered_time = times[rows]
        for position, row in enumerate(rows):
            start = np.searchsorted(
                ordered_time,
                ordered_time[position] - float(window_hours),
                side="left",
            )
            history_rows = rows[start : position + 1]
            for block in SUBTYPE_FEATURE_BLOCKS:
                block_slice = FEATURE_SLICES[block]
                trajectory = np.median(matrix[history_rows, block_slice], axis=0)
                output[row, block_slice] = (
                    float(current_weight) * matrix[row, block_slice]
                    + (1.0 - float(current_weight)) * trajectory
                )
    return output.astype(np.float32)


def _patient_balanced_indices(
    subject_ids: Sequence[object],
    *,
    max_rows_per_subject: int,
) -> np.ndarray:
    subjects = np.asarray(subject_ids, dtype=object)
    if subjects.ndim != 1:
        raise ValueError("subject_ids must be one-dimensional")
    selected: list[int] = []
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        if len(rows) <= max_rows_per_subject:
            selected.extend(rows.tolist())
            continue
        positions = np.linspace(0, len(rows) - 1, max_rows_per_subject)
        selected.extend(rows[np.round(positions).astype(int)].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _patient_equal_weights(
    subject_ids: Sequence[object],
    balance_group_ids: Sequence[object] | None = None,
) -> np.ndarray:
    subjects = np.asarray(subject_ids, dtype=object)
    unique, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    if not len(unique):
        raise ValueError("subject_ids cannot be empty")
    patient_weight = 1.0 / counts[inverse].astype(np.float64)
    if balance_group_ids is None:
        return patient_weight
    groups = np.asarray(balance_group_ids, dtype=object)
    if groups.shape != subjects.shape:
        raise ValueError("balance_group_ids must align with subject_ids")
    memberships = np.asarray(
        [f"{group}\x1f{subject}" for subject, group in zip(subjects, groups)],
        dtype=object,
    )
    _, membership_inverse, membership_rows = np.unique(
        memberships, return_inverse=True, return_counts=True
    )
    membership_weight = 1.0 / membership_rows[membership_inverse].astype(np.float64)
    group_memberships = {
        group: len(set(memberships[groups == group])) for group in np.unique(groups)
    }
    return membership_weight * np.asarray(
        [1.0 / group_memberships[group] for group in groups], dtype=np.float64
    )


@dataclass(frozen=True)
class SubtypeCandidatePrediction:
    """Candidate-only result; it must not bypass the runtime promotion gate."""

    cell: str
    base_point: float
    subtype_point: float
    personalized_point: float
    subtype_correction: float
    patient_correction: float
    patient_weight: float
    resolved_personal_errors: int
    subtype_probabilities: tuple[float, ...]
    supported_probability: float
    promotion_status: str = "candidate_only"
    causal_claim_allowed: bool = False


class SoftPhysiologySubtypeModel:
    """Soft subtype residual priors with patient-level partial pooling.

    This is a mixture-of-experts *adapter*, not a replacement for JEPA.  The
    frozen whole-body prediction remains ``base_point``.  Subtypes can only
    suggest a residual correction, and a separate external gate must decide
    whether an exact target-horizon cell may use that correction.
    """

    schema = "whole_body_jepa.soft_physiology_subtype.v2"
    architecture_contract = {
        "router_inputs": SUBTYPE_FEATURE_BLOCKS,
        "future_features_used": False,
        "target_specific_future_latent_used": False,
        "hospital_or_careunit_used": False,
        "assignment": "soft_posterior",
        "phenotype_fusion": "separate_scaled_latent_and_background_embeddings",
        "target_adapters": ("subtype_prior", "ridge", "direct_ridge"),
        "adaptation": "subtype_prior_plus_resolved_patient_error_shrinkage",
        "runtime_status": "candidate_only_until_external_gate",
    }

    def __init__(
        self,
        *,
        subtype_count: int = 6,
        min_effective_subjects: float = 30.0,
        subtype_prior_strength: float = 20.0,
        patient_prior_strength: float = 4.0,
        max_rows_per_subject: int = 4,
        embedding_dim: int = 16,
        random_state: int = 17,
        target_adapter: str = "subtype_prior",
        target_adapter_alpha: float = 100.0,
    ) -> None:
        if subtype_count < 2:
            raise ValueError("subtype_count must be at least two")
        if min_effective_subjects <= 0 or subtype_prior_strength <= 0:
            raise ValueError("subtype support and prior strength must be positive")
        if patient_prior_strength <= 0 or max_rows_per_subject < 1:
            raise ValueError("patient prior and row limit must be positive")
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least two")
        if target_adapter not in {"subtype_prior", "ridge", "direct_ridge"}:
            raise ValueError(
                "target_adapter must be subtype_prior, ridge, or direct_ridge"
            )
        if target_adapter_alpha <= 0.0:
            raise ValueError("target_adapter_alpha must be positive")
        self.subtype_count = int(subtype_count)
        self.min_effective_subjects = float(min_effective_subjects)
        self.subtype_prior_strength = float(subtype_prior_strength)
        self.patient_prior_strength = float(patient_prior_strength)
        self.max_rows_per_subject = int(max_rows_per_subject)
        self.embedding_dim = int(embedding_dim)
        self.random_state = int(random_state)
        self.target_adapter = str(target_adapter)
        self.target_adapter_alpha = float(target_adapter_alpha)
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.embedding: PCA | None = None
        self.phenotype_scaler: RobustScaler | None = None
        self.phenotype_embedding: PCA | None = None
        self.mixture = BayesianGaussianMixture(
            n_components=self.subtype_count,
            covariance_type="diag",
            weight_concentration_prior_type="dirichlet_process",
            reg_covar=1e-4,
            max_iter=500,
            n_init=3,
            random_state=self.random_state,
        )
        self.cell_keys: tuple[str, ...] = ()
        self.residual_prior = np.empty((0, 0), dtype=np.float64)
        self.effective_subjects = np.empty((0, 0), dtype=np.float64)
        self.supported = np.empty((0, 0), dtype=bool)
        self.target_models: tuple[Ridge | None, ...] = ()
        self.phenotype_covariate_width = 0
        self.fitted = False

    def fit(
        self,
        feature_matrix: np.ndarray,
        *,
        base_points: np.ndarray,
        observed_targets: np.ndarray,
        observed_mask: np.ndarray,
        subject_ids: Sequence[object],
        cell_keys: Sequence[str],
        responsibility_permutation_seed: int | None = None,
        balance_group_ids: Sequence[object] | None = None,
        phenotype_covariates: np.ndarray | None = None,
    ) -> "SoftPhysiologySubtypeModel":
        """Fit on discovery patients only; future values never enter routing.

        ``responsibility_permutation_seed`` creates the capacity-matched
        placebo used by the external gate. It keeps the fitted router and
        parameter count fixed, but assigns each discovery patient another
        patient's mean subtype responsibility before residual-prior fitting.
        ``balance_group_ids`` may balance residual-prior fitting across
        discovery hospitals; group identity is never an encoder or router
        feature.
        """

        combined_features = subtype_features(feature_matrix, phenotype_covariates)
        features = combined_features[:, :SUBTYPE_BASE_WIDTH]
        covariates = combined_features[:, SUBTYPE_BASE_WIDTH:]
        base = np.asarray(base_points, dtype=np.float64)
        target = np.asarray(observed_targets, dtype=np.float64)
        mask = np.asarray(observed_mask, dtype=bool)
        subjects = np.asarray(subject_ids, dtype=object)
        if base.ndim == 1:
            base = base[:, None]
        if target.ndim == 1:
            target = target[:, None]
        if mask.ndim == 1:
            mask = mask[:, None]
        expected = (len(features), len(cell_keys))
        if base.shape != expected or target.shape != expected or mask.shape != expected:
            raise ValueError("base_points, targets, and mask must match rows x cells")
        if subjects.shape != (len(features),):
            raise ValueError("subject_ids must align with feature rows")
        if len(np.unique(subjects)) < max(20, self.subtype_count * 3):
            raise ValueError("too few discovery patients for physiological subtyping")

        fit_rows = _patient_balanced_indices(
            subjects,
            max_rows_per_subject=self.max_rows_per_subject,
        )
        self.scaler.fit(features[fit_rows])
        normalized = self.scaler.transform(features)
        component_count = min(
            self.embedding_dim,
            normalized.shape[1],
            max(2, len(fit_rows) - 1),
        )
        # Correlated physiological movement survives this train-only
        # compression, while independent high-dimensional noise cannot vote
        # 213 times in the subtype distance.
        self.embedding = PCA(
            n_components=component_count,
            whiten=False,
            random_state=self.random_state,
        )
        embedded_fit = self.embedding.fit_transform(normalized[fit_rows])
        embedded = self.embedding.transform(normalized)
        direct_adapter_features = embedded
        if covariates.shape[1]:
            self.phenotype_scaler = RobustScaler(quantile_range=(10.0, 90.0))
            self.phenotype_scaler.fit(covariates[fit_rows])
            normalized_covariates = self.phenotype_scaler.transform(covariates)
            phenotype_components = min(
                8,
                self.embedding_dim,
                covariates.shape[1],
                max(1, len(fit_rows) - 1),
            )
            self.phenotype_embedding = PCA(
                n_components=phenotype_components,
                whiten=False,
                random_state=self.random_state,
            )
            embedded_covariates = self.phenotype_embedding.fit_transform(
                normalized_covariates[fit_rows]
            )
            direct_adapter_features = np.concatenate(
                (embedded, normalized_covariates), axis=1
            )
            embedded = np.concatenate(
                (
                    embedded,
                    self.phenotype_embedding.transform(normalized_covariates),
                ),
                axis=1,
            )
            embedded_fit = np.concatenate(
                (embedded_fit, embedded_covariates), axis=1
            )
        self.mixture.fit(embedded_fit)
        responsibility = self.mixture.predict_proba(embedded)
        adapter_features = np.concatenate((embedded, responsibility), axis=1)
        if responsibility_permutation_seed is not None:
            unique_subjects = np.unique(subjects)
            permutation = np.random.default_rng(
                int(responsibility_permutation_seed)
            ).permutation(len(unique_subjects))
            def permuted_patient_means(matrix: np.ndarray) -> np.ndarray:
                subject_representation = np.stack(
                    [
                        matrix[subjects == subject].mean(axis=0)
                        for subject in unique_subjects
                    ]
                )
                placebo_by_subject = {
                    subject: subject_representation[permutation[index]]
                    for index, subject in enumerate(unique_subjects)
                }
                return np.stack(
                    [placebo_by_subject[subject] for subject in subjects]
                )

            adapter_features = permuted_patient_means(adapter_features)
            direct_adapter_features = permuted_patient_means(
                direct_adapter_features
            )
            responsibility = adapter_features[:, -self.subtype_count :]
        patient_weight = _patient_equal_weights(subjects, balance_group_ids)
        residual = target - base

        cells = len(cell_keys)
        means = np.zeros((self.subtype_count, cells), dtype=np.float64)
        support = np.zeros_like(means)
        supported = np.zeros_like(means, dtype=bool)
        for component in range(self.subtype_count):
            for cell in range(cells):
                valid = mask[:, cell] & np.isfinite(residual[:, cell])
                if not valid.any():
                    continue
                weights = responsibility[valid, component] * patient_weight[valid]
                total = float(weights.sum())
                squared = float(np.square(weights).sum())
                effective = total * total / squared if squared > 0.0 else 0.0
                support[component, cell] = effective
                if total <= 0.0:
                    continue
                raw_mean = float(np.dot(weights, residual[valid, cell]) / total)
                shrinkage = effective / (effective + self.subtype_prior_strength)
                means[component, cell] = shrinkage * raw_mean
                supported[component, cell] = effective >= self.min_effective_subjects

        self.cell_keys = tuple(str(cell) for cell in cell_keys)
        self.phenotype_covariate_width = (
            0 if phenotype_covariates is None else int(np.asarray(phenotype_covariates).shape[1])
        )
        self.residual_prior = means
        self.effective_subjects = support
        self.supported = supported
        target_models: list[Ridge | None] = []
        for cell in range(cells):
            valid = mask[:, cell] & np.isfinite(residual[:, cell])
            if (
                self.target_adapter not in {"ridge", "direct_ridge"}
                or len(np.unique(subjects[valid])) < self.min_effective_subjects
            ):
                target_models.append(None)
                continue
            weights = patient_weight[valid]
            weights = weights / max(float(weights.mean()), 1e-12)
            adapter = Ridge(alpha=self.target_adapter_alpha)
            model_features = (
                direct_adapter_features
                if self.target_adapter == "direct_ridge"
                else adapter_features
            )
            adapter.fit(
                model_features[valid],
                residual[valid, cell],
                sample_weight=weights,
            )
            target_models.append(adapter)
        self.target_models = tuple(target_models)
        self.fitted = True
        return self

    def subtype_corrections(
        self,
        feature_matrix: np.ndarray,
        *,
        cell: str,
        phenotype_covariates: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return vectorized candidate-only residual corrections for one cell."""

        if not self.fitted:
            raise ValueError("subtype model is not fitted")
        try:
            cell_index = self.cell_keys.index(str(cell))
        except ValueError as exc:
            raise ValueError(f"unknown subtype target cell: {cell}") from exc
        if self.target_adapter in {"ridge", "direct_ridge"}:
            adapter = self.target_models[cell_index]
            if adapter is None:
                matrix = np.asarray(feature_matrix)
                rows = 1 if matrix.ndim == 1 else len(matrix)
                return np.zeros(rows, dtype=np.float64)
            representation = (
                self._direct_representation(feature_matrix, phenotype_covariates)
                if self.target_adapter == "direct_ridge"
                else self._representation(feature_matrix, phenotype_covariates)[0]
            )
            return adapter.predict(representation).astype(np.float64)
        probability = self.responsibilities(feature_matrix, phenotype_covariates)
        valid = self.supported[:, cell_index]
        if not valid.any():
            return np.zeros(len(probability), dtype=np.float64)
        supported_probability = probability[:, valid].sum(axis=1)
        weighted = probability[:, valid] @ self.residual_prior[valid, cell_index]
        return np.divide(
            weighted,
            supported_probability,
            out=np.zeros_like(weighted, dtype=np.float64),
            where=supported_probability > 0.0,
        )

    def responsibilities(
        self,
        feature_matrix: np.ndarray,
        phenotype_covariates: np.ndarray | None = None,
    ) -> np.ndarray:
        _, probability = self._representation(
            feature_matrix, phenotype_covariates
        )
        return probability

    def _representation(
        self,
        feature_matrix: np.ndarray,
        phenotype_covariates: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("subtype model is not fitted")
        if self.embedding is None:
            raise ValueError("subtype embedding is missing")
        combined_features = subtype_features(feature_matrix, phenotype_covariates)
        features = combined_features[:, :SUBTYPE_BASE_WIDTH]
        covariates = combined_features[:, SUBTYPE_BASE_WIDTH:]
        expected = getattr(self, "phenotype_covariate_width", 0)
        actual = 0 if phenotype_covariates is None else int(
            np.asarray(phenotype_covariates).reshape(len(features), -1).shape[1]
        )
        if actual != expected:
            raise ValueError(
                f"phenotype covariate width mismatch: expected {expected}, got {actual}"
            )
        normalized = self.scaler.transform(features)
        embedded = self.embedding.transform(normalized)
        if expected:
            if self.phenotype_scaler is None or self.phenotype_embedding is None:
                raise ValueError("phenotype embedding is missing")
            normalized_covariates = self.phenotype_scaler.transform(covariates)
            embedded = np.concatenate(
                (
                    embedded,
                    self.phenotype_embedding.transform(normalized_covariates),
                ),
                axis=1,
            )
        probability = self.mixture.predict_proba(embedded)
        probability = probability / probability.sum(axis=1, keepdims=True)
        return np.concatenate((embedded, probability), axis=1), probability

    def _direct_representation(
        self,
        feature_matrix: np.ndarray,
        phenotype_covariates: np.ndarray | None,
    ) -> np.ndarray:
        """Return target-head features without unsupervised phenotype PCA."""

        if self.embedding is None:
            raise ValueError("subtype embedding is missing")
        combined = subtype_features(feature_matrix, phenotype_covariates)
        features = combined[:, :SUBTYPE_BASE_WIDTH]
        covariates = combined[:, SUBTYPE_BASE_WIDTH:]
        embedded = self.embedding.transform(self.scaler.transform(features))
        if not covariates.shape[1]:
            return embedded
        if self.phenotype_scaler is None:
            raise ValueError("phenotype scaler is missing")
        return np.concatenate(
            (embedded, self.phenotype_scaler.transform(covariates)), axis=1
        )

    def candidate_predict(
        self,
        feature_vector: np.ndarray,
        *,
        cell: str,
        base_point: float,
        resolved_patient_errors: Sequence[float] = (),
        phenotype_covariates: np.ndarray | None = None,
    ) -> SubtypeCandidatePrediction:
        """Return a research candidate without changing the promoted runtime."""

        if not self.fitted:
            raise ValueError("subtype model is not fitted")
        try:
            cell_index = self.cell_keys.index(str(cell))
        except ValueError as exc:
            raise ValueError(f"unknown subtype target cell: {cell}") from exc
        probability = self.responsibilities(
            np.asarray(feature_vector), phenotype_covariates
        )[0]
        valid = self.supported[:, cell_index]
        supported_probability = float(probability[valid].sum())
        if self.target_adapter in {"ridge", "direct_ridge"}:
            subtype_correction = float(
                self.subtype_corrections(
                    np.asarray(feature_vector),
                    cell=cell,
                    phenotype_covariates=phenotype_covariates,
                )[0]
            )
        elif supported_probability > 0.0:
            normalized_probability = probability[valid] / supported_probability
            subtype_correction = float(
                np.dot(normalized_probability, self.residual_prior[valid, cell_index])
            )
        else:
            subtype_correction = 0.0

        personal = np.asarray(resolved_patient_errors, dtype=np.float64)
        personal = personal[np.isfinite(personal)]
        count = int(len(personal))
        patient_weight = count / (count + self.patient_prior_strength)
        personal_mean = float(personal.mean()) if count else subtype_correction
        correction = (
            (1.0 - patient_weight) * subtype_correction
            + patient_weight * personal_mean
        )
        base = float(base_point)
        return SubtypeCandidatePrediction(
            cell=str(cell),
            base_point=base,
            subtype_point=base + subtype_correction,
            personalized_point=base + correction,
            subtype_correction=subtype_correction,
            patient_correction=correction,
            patient_weight=float(patient_weight),
            resolved_personal_errors=count,
            subtype_probabilities=tuple(float(value) for value in probability),
            supported_probability=supported_probability,
        )

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise ValueError("cannot save an unfitted subtype model")
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "SoftPhysiologySubtypeModel":
        model = joblib.load(Path(path))
        if not isinstance(model, cls) or model.schema != cls.schema:
            raise ValueError("unsupported physiological subtype artifact")
        return model


__all__ = [
    "SUBTYPE_FEATURE_BLOCKS",
    "SoftPhysiologySubtypeModel",
    "SubtypeCandidatePrediction",
    "causal_history_covariates",
    "causal_history_covariates_at_anchors",
    "causal_trajectory_features",
    "subtype_features",
]
