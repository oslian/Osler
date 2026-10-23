"""Teacher-anchored whole-body JEPA residual audit.

The validated factual router remains the forecast anchor.  A frozen joint JEPA
provides shared slow context, target-specific fast future latents, and organ
tokens.  Small zero-initialized residual adapters may improve the router, but
cannot silently replace it.

Bounded candidates are compared on identical patient splits, including a
strict adjacent hierarchy whose only representation edges are
system <-> organ <-> body.

* hierarchical: fast/slow JEPA latents, no explicit organ route;
* dynamic_route: patient/target/horizon-specific sparse cross-organ attention;
* dynamic_route_pcgrad: the same adapter with PCGrad across target losses.

This file is research-only.  It never writes a promotion registry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.utils.extmath import randomized_svd
from torch import nn

from osler_jepa.adjacent_hierarchy import (
    HIERARCHY_MODELS,
    AdjacentThreeLevelResidualAdapter,
    split_adjacent_training_epochs,
)
from osler_jepa.patient_split import selection_patients
from osler_jepa.organ_schema import (
    canonical_organ_adjacency,
    canonical_target_organ_membership,
)
from osler_jepa.interval_personalization import (
    online_adaptive_interval_widths,
    patient_weighted_quantile,
    selective_interval_widths,
)

from whole_body_joint_gate import (
    _finite_sample_conformal_quantile,
    _patient_paired_bootstrap,
    _split_by_column,
    _split_forward_time,
    _split_known_groups,
    _split_train_calibration,
    _validated_router_prediction,
)
from whole_body_joint_jepa import (
    fit_joint,
    load_joint_event_examples,
)


DEFAULT_TARGETS = (
    "glucose",
    "bicarbonate",
    "creatinine",
    "potassium",
    "map",
    "heart_rate",
)

DEFAULT_VARIANTS = ("adjacent_three_level",)


def _validate_resume_seed_extension(previous_seeds, requested_seeds, reports):
    """Allow adding seeds while refusing a changed or narrowed experiment."""

    previous = {int(value) for value in previous_seeds}
    requested = {int(value) for value in requested_seeds}
    if not previous.issubset(requested):
        raise ValueError("Resume may extend, but cannot remove, requested seeds")
    completed = {
        int(report["seed"]) for report in reports if "seed" in report
    }
    if not completed.issubset(requested):
        raise ValueError(
            "Completed progress contains a seed absent from this run"
        )


def _select_targets(requested: str, variables: Iterable[str]) -> tuple[str, ...]:
    """Resolve an explicit target list or the complete canonical schema."""

    available = tuple(str(value) for value in variables)
    values = tuple(
        value.strip() for value in requested.split(",") if value.strip()
    )
    if len(values) == 1 and values[0].lower() == "all":
        return available
    return tuple(value for value in values if value in available)


def _future_label_support(
    frame: pd.DataFrame,
    target_names: Iterable[str],
    horizons: Iterable[int],
    *,
    minimum_rows: int = 30,
) -> dict[str, dict]:
    """Audit which target/horizon cells can enter the formal evaluation.

    Every requested target remains in the model. Cells without enough measured
    future labels are recorded as support-limited instead of disappearing from
    the report or being treated as successful forecasts.
    """

    output: dict[str, dict] = {}
    horizon_values = pd.to_numeric(frame["horizon_hours"], errors="coerce")
    subjects = frame["subject_id"].astype(str)
    for target in target_names:
        future_column = f"future_{target}"
        if future_column not in frame:
            output[target] = {
                "total_rows": 0,
                "total_subjects": 0,
                "evaluable_horizons": [],
                "support_limited_horizons": [int(value) for value in horizons],
                "by_horizon": {},
            }
            continue
        observed = pd.to_numeric(frame[future_column], errors="coerce").notna()
        by_horizon = {}
        for horizon in horizons:
            selected = observed & (horizon_values == float(horizon))
            rows = int(selected.sum())
            subject_count = int(subjects[selected].nunique())
            by_horizon[f"{int(horizon)}h"] = {
                "rows": rows,
                "subjects": subject_count,
                "evaluable": bool(rows >= minimum_rows and subject_count >= 20),
            }
        output[target] = {
            "total_rows": int(observed.sum()),
            "total_subjects": int(subjects[observed].nunique()),
            "evaluable_horizons": [
                key for key, value in by_horizon.items() if value["evaluable"]
            ],
            "support_limited_horizons": [
                key for key, value in by_horizon.items() if not value["evaluable"]
            ],
            "by_horizon": by_horizon,
        }
    return output


def _split_rows_for_mode(
    frame: pd.DataFrame,
    mode: str,
    seed: int,
    *,
    train_database: str | None = None,
    test_database: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the requested leakage-safe external evaluation partition."""

    if mode == "subject":
        return _split_by_column(frame, "subject_id", seed)
    if mode == "hospital":
        return _split_known_groups(frame, "hospitalid", seed)
    if mode == "careunit":
        return _split_known_groups(frame, "careunit", seed)
    if mode == "time":
        return _split_forward_time(frame)
    if mode == "database":
        if not train_database or not test_database:
            raise ValueError(
                "database split requires train_database and test_database"
            )
        database = frame["database"].fillna("").astype(str).str.lower()
        train = np.flatnonzero(database.eq(str(train_database).lower()))
        test = np.flatnonzero(database.isin([value.strip().lower() for value in str(test_database).split(",")]))
        if not len(train) or not len(test):
            raise ValueError(
                "database split has no rows for the requested train/test pair"
            )
        if set(frame.iloc[train]["subject_id"].astype(str)) & set(
            frame.iloc[test]["subject_id"].astype(str)
        ):
            raise ValueError("database split contains overlapping patients")
        return train, test
    raise ValueError(f"Unknown split mode: {mode}")


def _load_expert_anchor_registry(path: Path | None) -> dict | None:
    if path is None:
        return None
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = registry.get("cells", [])
    if not cells:
        raise ValueError("expert anchor registry has no cells")
    for cell in cells:
        required = {"cell", "target", "horizon_hours", "anchor_column"}
        missing = required - set(cell)
        if missing:
            raise ValueError(
                f"expert anchor cell missing fields: {sorted(missing)}"
            )
    return registry


def _fixed_expert_anchor_prediction(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    arrays,
    prediction_rows: np.ndarray,
    registry: dict,
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    """Read frozen expert anchors without selecting on held-out labels."""

    scaler = arrays[0]
    horizon = arrays[8]
    rows = np.asarray(prediction_rows, dtype=np.int64)
    output = np.full((len(rows), len(variables)), np.nan, dtype=np.float32)
    metadata: dict[str, dict[str, object]] = {}
    for cell in registry["cells"]:
        if cell.get("task_type") == "binary_event_probability":
            continue
        target = str(cell["target"])
        if target not in variables:
            continue
        column = str(cell["anchor_column"])
        if column not in frame:
            raise ValueError(f"missing fixed expert anchor column: {column}")
        target_index = variables.index(target)
        horizon_value = float(cell["horizon_hours"])
        local = np.flatnonzero(horizon[rows] == horizon_value)
        raw = pd.to_numeric(
            frame.iloc[rows[local]][column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        output[local, target_index] = (
            (raw - scaler.medians[target_index]) / scaler.scales[target_index]
        ).astype(np.float32)
        metadata[str(cell["cell"])] = {
            "selected_source": str(cell.get("source_model", "fixed_expert")),
            "fixed_external_expert": True,
            "anchor_column": column,
            "available_rows": int(np.isfinite(raw).sum()),
            "heldout_labels_used_for_selection": False,
        }
    return output, metadata


def _time_features(horizon: torch.Tensor) -> torch.Tensor:
    horizon = horizon.clamp(min=0.05, max=168.0)
    log_horizon = torch.log1p(horizon)
    return torch.stack(
        (
            log_horizon / math.log1p(48.0),
            horizon / 6.0,
            torch.sin(log_horizon),
            torch.cos(log_horizon),
        ),
        dim=-1,
    )


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project logits onto the probability simplex with exact sparsity."""

    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits = torch.sort(shifted, dim=dim, descending=True).values
    cumulative = sorted_logits.cumsum(dim)
    width = logits.shape[dim]
    rank_shape = [1] * logits.ndim
    rank_shape[dim] = width
    ranks = torch.arange(
        1, width + 1, device=logits.device, dtype=logits.dtype
    ).reshape(rank_shape)
    support = 1.0 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (cumulative.gather(dim, support_size - 1) - 1.0) / (
        support_size.to(logits.dtype)
    )
    return torch.clamp(shifted - threshold, min=0.0)


class TeacherAnchoredResidualAdapter(nn.Module):
    """Learn only a bounded residual over a fixed factual-router teacher."""

    def __init__(
        self,
        *,
        target_indices: Iterable[int],
        target_module_membership: np.ndarray,
        latent_dim: int,
        organ_dim: int,
        module_count: int,
        adapter_dim: int = 48,
        dynamic_organ_route: bool = True,
        sparse_experts: bool = False,
        distributional_scale: bool = False,
        target_specific_projections: bool = False,
    ):
        super().__init__()
        target_indices = tuple(int(value) for value in target_indices)
        self.target_indices = target_indices
        self.target_count = len(target_indices)
        self.module_count = int(module_count)
        self.dynamic_organ_route = bool(dynamic_organ_route)
        self.sparse_experts = bool(sparse_experts)
        self.distributional_scale = bool(distributional_scale)
        self.target_specific_projections = bool(
            target_specific_projections
        )
        self.slow_projection = nn.Sequential(
            nn.Linear(latent_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.fast_projection = nn.Sequential(
            nn.Linear(latent_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.timescale_query = nn.Sequential(
            nn.Linear(4, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, self.target_count),
        )
        self.target_timescale_bias = nn.Parameter(
            torch.zeros(self.target_count)
        )
        self.target_embedding = nn.Parameter(
            torch.randn(self.target_count, adapter_dim) * 0.02
        )
        self.current_query = nn.Sequential(
            nn.Linear(3, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        if self.dynamic_organ_route:
            self.organ_key = nn.Sequential(
                nn.Linear(organ_dim, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.SiLU(),
            )
            self.organ_value = nn.Sequential(
                nn.Linear(organ_dim, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.SiLU(),
            )
            self.route_time_query = nn.Sequential(
                nn.Linear(4, adapter_dim),
                nn.SiLU(),
            )
            self.route_gates = nn.Parameter(
                torch.full((self.target_count,), -2.0)
            )
        membership = np.asarray(target_module_membership, dtype=np.float32)
        expected = (self.target_count, self.module_count)
        if membership.shape != expected:
            raise ValueError(
                f"target_module_membership must have shape {expected}"
            )
        own_membership = np.clip(membership, 0.0, 1.0)
        own_membership /= np.maximum(
            own_membership.sum(axis=1, keepdims=True), 1.0
        )
        self.register_buffer(
            "own_organ_membership",
            torch.as_tensor(own_membership),
            persistent=True,
        )
        if self.sparse_experts:
            self.own_organ_projection = nn.Sequential(
                nn.Linear(organ_dim, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.SiLU(),
            )
            self.expert_gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(7, adapter_dim),
                        nn.SiLU(),
                        nn.Linear(adapter_dim, 3),
                    )
                    for _ in target_indices
                ]
            )
        if self.target_specific_projections:
            def projection_adapter(input_dim):
                block = nn.Sequential(
                    nn.Linear(input_dim, adapter_dim),
                    nn.SiLU(),
                    nn.Linear(adapter_dim, adapter_dim),
                )
                nn.init.zeros_(block[-1].weight)
                nn.init.zeros_(block[-1].bias)
                return block

            self.target_slow_adapters = nn.ModuleList(
                [projection_adapter(latent_dim) for _ in target_indices]
            )
            self.target_fast_adapters = nn.ModuleList(
                [projection_adapter(latent_dim) for _ in target_indices]
            )
            self.target_organ_adapters = nn.ModuleList(
                [projection_adapter(organ_dim) for _ in target_indices]
            )
        # Routing is explicitly cross-organ.  A target's own owning organ is
        # removed because the teacher already contains the local history.
        self.register_buffer(
            "cross_organ_allowed",
            torch.as_tensor(1.0 - np.clip(membership, 0.0, 1.0)),
            persistent=True,
        )
        residual_width = adapter_dim * 2 + 1 + 4
        self.residual_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(residual_width, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.SiLU(),
                    nn.Linear(adapter_dim, 1),
                )
                for _ in target_indices
            ]
        )
        for head in self.residual_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.residual_gates = nn.Parameter(
            torch.full((self.target_count,), -2.0)
        )
        if self.distributional_scale:
            self.scale_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(residual_width, adapter_dim),
                        nn.LayerNorm(adapter_dim),
                        nn.SiLU(),
                        nn.Linear(adapter_dim, 1),
                    )
                    for _ in target_indices
                ]
            )
            initial_scale = 0.50
            initial_raw = math.log(math.expm1(initial_scale - 0.03))
            for head in self.scale_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.constant_(head[-1].bias, initial_raw)

    def forward(
        self,
        *,
        teacher_anchor: torch.Tensor,
        context_latent: torch.Tensor,
        target_latent: torch.Tensor,
        organ_latents: torch.Tensor,
        organ_presence: torch.Tensor,
        current: torch.Tensor,
        current_mask: torch.Tensor,
        ages: torch.Tensor,
        horizon: torch.Tensor,
        disable_organ_route: bool = False,
        disable_sparse_experts: bool = False,
        disable_target_specific_projections: bool = False,
    ) -> dict[str, torch.Tensor]:
        time = _time_features(horizon)
        slow = self.slow_projection(context_latent).unsqueeze(1).expand(
            -1, self.target_count, -1
        )
        fast = self.fast_projection(target_latent)
        if (
            self.target_specific_projections
            and not disable_target_specific_projections
        ):
            slow_delta = torch.stack(
                [
                    adapter(context_latent)
                    for adapter in self.target_slow_adapters
                ],
                dim=1,
            )
            fast_delta = torch.stack(
                [
                    adapter(target_latent[:, index])
                    for index, adapter in enumerate(
                        self.target_fast_adapters
                    )
                ],
                dim=1,
            )
            slow = slow + 0.5 * torch.tanh(slow_delta)
            fast = fast + 0.5 * torch.tanh(fast_delta)
        timescale = torch.sigmoid(
            self.timescale_query(time) + self.target_timescale_bias
        )
        hierarchical = (
            timescale.unsqueeze(-1) * fast
            + (1.0 - timescale.unsqueeze(-1)) * slow
        )

        expert_weights = hierarchical.new_zeros(
            (len(hierarchical), self.target_count, 3)
        )
        expert_weights[..., 0] = 1.0 - timescale
        expert_weights[..., 1] = timescale
        if self.sparse_experts and not disable_sparse_experts:
            own_weights = self.own_organ_membership.unsqueeze(0)
            own_presence = (
                organ_presence.unsqueeze(1) * own_weights
            ).sum(dim=-1)
            own_latent = torch.einsum(
                "tm,bmd->btd", self.own_organ_membership, organ_latents
            )
            own = self.own_organ_projection(own_latent)
            if (
                self.target_specific_projections
                and not disable_target_specific_projections
            ):
                own_delta = torch.stack(
                    [
                        adapter(own_latent[:, index])
                        for index, adapter in enumerate(
                            self.target_organ_adapters
                        )
                    ],
                    dim=1,
                )
                own = own + 0.5 * torch.tanh(own_delta)
            target_state = torch.stack((current, current_mask, ages), dim=-1)
            gate_inputs = torch.cat(
                (
                    target_state,
                    time.unsqueeze(1).expand(-1, self.target_count, -1),
                ),
                dim=-1,
            )
            gate_logits = torch.stack(
                [
                    gate(gate_inputs[:, index])
                    for index, gate in enumerate(self.expert_gates)
                ],
                dim=1,
            )
            gate_logits[..., 2] = gate_logits[..., 2].masked_fill(
                own_presence <= 0.0, -1.0e4
            )
            expert_weights = _sparsemax(gate_logits, dim=-1)
            expert_stack = torch.stack((slow, fast, own), dim=2)
            hierarchical = (
                expert_weights.unsqueeze(-1) * expert_stack
            ).sum(dim=2)

        route_weights = organ_latents.new_zeros(
            (len(organ_latents), self.target_count, self.module_count)
        )
        route_context = torch.zeros_like(hierarchical)
        if self.dynamic_organ_route and not disable_organ_route:
            keys = self.organ_key(organ_latents)
            values = self.organ_value(organ_latents)
            target_current = torch.stack(
                (current, current_mask, ages), dim=-1
            )
            query = (
                self.target_embedding.unsqueeze(0)
                + self.current_query(target_current)
                + self.route_time_query(time).unsqueeze(1)
            )
            logits = torch.einsum("btd,bmd->btm", query, keys)
            logits = logits / math.sqrt(float(keys.shape[-1]))
            allowed = (
                organ_presence.unsqueeze(1)
                * self.cross_organ_allowed.unsqueeze(0)
            )
            valid = allowed.sum(dim=-1, keepdim=True) > 0.0
            masked_logits = logits.masked_fill(allowed <= 0.0, -1.0e4)
            route_weights = torch.softmax(masked_logits, dim=-1)
            route_weights = torch.where(
                valid, route_weights * allowed, torch.zeros_like(route_weights)
            )
            route_weights = route_weights / route_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
            routed = torch.einsum("btm,bmd->btd", route_weights, values)
            route_context = (
                torch.sigmoid(self.route_gates).view(1, -1, 1) * routed
            )

        predictions, residuals, scales = [], [], []
        for local_index, head in enumerate(self.residual_heads):
            residual_input = torch.cat(
                (
                    hierarchical[:, local_index],
                    route_context[:, local_index],
                    current[:, local_index : local_index + 1],
                    time,
                ),
                dim=-1,
            )
            residual = (
                0.75
                * torch.sigmoid(self.residual_gates[local_index])
                * torch.tanh(head(residual_input))
            )
            residuals.append(residual)
            predictions.append(
                teacher_anchor[:, local_index : local_index + 1] + residual
            )
            if self.distributional_scale:
                raw_scale = self.scale_heads[local_index](
                    residual_input.detach()
                )
                scale = (0.03 + nn.functional.softplus(raw_scale)).clamp(
                    max=3.0
                )
            else:
                scale = torch.ones_like(residual)
            scales.append(scale)
        probability = route_weights.clamp_min(1.0e-8)
        route_entropy = -(
            probability * probability.log()
        ).sum(dim=-1)
        expert_probability = expert_weights.clamp_min(1.0e-8)
        expert_entropy = -(
            expert_probability * expert_probability.log()
        ).sum(dim=-1)
        return {
            "value": torch.cat(predictions, dim=-1),
            "residual": torch.cat(residuals, dim=-1),
            "timescale_fast_weight": timescale,
            "route_weights": route_weights,
            "route_entropy": route_entropy,
            "expert_weights": expert_weights,
            "expert_entropy": expert_entropy,
            "scale": torch.cat(scales, dim=-1),
        }


def _cache_base_features(
    model,
    arrays,
    target_indices,
    rows,
    *,
    batch_size=2048,
):
    (
        _,
        input_matrix,
        current,
        mask,
        ages,
        _,
        _,
        history,
        horizon,
        presence,
    ) = arrays[:10]
    target_indices = np.asarray(target_indices, dtype=np.int64)
    cached = {
        "context": [],
        "target": [],
        "organ": [],
        "presence": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = np.asarray(rows[start : start + batch_size], dtype=np.int64)
            output = model(
                torch.from_numpy(input_matrix[history[batch]]).float(),
                torch.from_numpy(current[batch]).float(),
                torch.from_numpy(mask[batch]).float(),
                torch.from_numpy(ages[batch]).float(),
                torch.from_numpy(presence[batch]).float(),
                torch.from_numpy(horizon[batch]).float(),
            )
            cached["context"].append(
                output["context_latent"].detach().cpu().to(torch.float16)
            )
            cached["target"].append(
                output["predicted_target_latent"][:, target_indices]
                .detach()
                .cpu()
                .to(torch.float16)
            )
            cached["organ"].append(
                output["organ_latents"].detach().cpu().to(torch.float16)
            )
            cached["presence"].append(
                output["organ_presence"].detach().cpu().to(torch.uint8)
            )
    return {key: torch.cat(value, dim=0) for key, value in cached.items()}


def _feature_batch(features, key, rows):
    """Materialize compressed CPU latent features for model computation."""

    return features[key][rows].to(dtype=torch.float32)


def _pairwise_gradient_cosines(gradients):
    output = []
    for left in range(len(gradients)):
        for right in range(left + 1, len(gradients)):
            a, b = gradients[left], gradients[right]
            denominator = a.norm() * b.norm()
            if float(denominator) <= 1e-12:
                continue
            output.append(float(torch.dot(a, b) / denominator))
    return output


def _flatten_gradient(parameters, gradients):
    parts = []
    for parameter, gradient in zip(parameters, gradients):
        parts.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
        )
    return torch.cat(parts)


def _assign_flat_gradient(parameters, gradient):
    offset = 0
    for parameter in parameters:
        width = parameter.numel()
        parameter.grad = gradient[offset : offset + width].reshape_as(
            parameter
        ).clone()
        offset += width


def _pcgrad(target_losses, parameters, rng):
    gradients = []
    for index, loss in enumerate(target_losses):
        raw = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index < len(target_losses) - 1,
            allow_unused=True,
        )
        gradients.append(_flatten_gradient(parameters, raw))
    original_cosines = _pairwise_gradient_cosines(gradients)
    projected = []
    for index, gradient in enumerate(gradients):
        adjusted = gradient.clone()
        order = rng.permutation(len(gradients))
        for other_index in order:
            if int(other_index) == index:
                continue
            other = gradients[int(other_index)]
            dot = torch.dot(adjusted, other)
            if float(dot) < 0.0:
                adjusted = adjusted - dot * other / other.dot(other).clamp_min(
                    1e-12
                )
        projected.append(adjusted)
    _assign_flat_gradient(parameters, torch.stack(projected).mean(dim=0))
    return original_cosines, _pairwise_gradient_cosines(projected)


def _adapter_batches(rows, batch_size, rng, balance_labels=None):
    rows = np.asarray(rows, dtype=np.int64)
    if balance_labels is None:
        order = rng.permutation(rows)
    else:
        labels = np.asarray(balance_labels, dtype=object)
        groups = {
            label: rows[labels[rows] == label]
            for label in np.unique(labels[rows])
        }
        group_names = np.asarray(sorted(groups), dtype=object)
        sampled_groups = rng.choice(group_names, size=len(rows), replace=True)
        order = np.asarray(
            [
                rng.choice(groups[label])
                for label in sampled_groups
            ],
            dtype=np.int64,
        )
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def _train_adapter(
    adapter,
    features,
    arrays,
    teacher,
    train_rows,
    target_indices,
    *,
    epochs,
    batch_size,
    seed,
    pcgrad,
    route_entropy_weight,
    residual_l1_weight,
    expert_entropy_weight=0.0,
    distributional_weight=0.0,
    horizon_balanced_loss=False,
    balance_labels=None,
):
    (
        _,
        _,
        current,
        mask,
        ages,
        future,
        future_mask,
        _,
        horizon,
    ) = arrays[:9]
    target_indices = np.asarray(target_indices, dtype=np.int64)
    trainable_parameters = [
        parameter
        for parameter in adapter.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=5e-4, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed)
    losses = []
    conflict_before, conflict_after = [], []
    for _ in range(int(epochs)):
        adapter.train()
        epoch_losses = []
        for batch in _adapter_batches(
            train_rows, batch_size, rng, balance_labels=balance_labels
        ):
            current_batch = torch.from_numpy(
                current[batch][:, target_indices]
            ).float()
            output = adapter(
                teacher_anchor=torch.from_numpy(
                    teacher[batch][:, target_indices]
                ).float(),
                context_latent=_feature_batch(features, "context", batch),
                target_latent=_feature_batch(features, "target", batch),
                organ_latents=_feature_batch(features, "organ", batch),
                organ_presence=_feature_batch(features, "presence", batch),
                current=current_batch,
                current_mask=torch.from_numpy(
                    mask[batch][:, target_indices]
                ).float(),
                ages=torch.from_numpy(
                    ages[batch][:, target_indices]
                ).float(),
                horizon=torch.from_numpy(horizon[batch]).float(),
            )
            truth = torch.from_numpy(
                future[batch][:, target_indices]
            ).float()
            observed = torch.from_numpy(
                future_mask[batch][:, target_indices]
            ).float()
            batch_horizon = torch.from_numpy(horizon[batch]).float()
            target_losses = []
            for index in range(len(target_indices)):
                valid = observed[:, index] > 0.0
                if int(valid.sum()) < 2:
                    continue
                groups = [valid]
                if horizon_balanced_loss:
                    groups = [
                        valid & (batch_horizon == value)
                        for value in torch.unique(batch_horizon[valid])
                    ]
                    groups = [group for group in groups if int(group.sum()) >= 2]
                horizon_losses = []
                for group in groups:
                    value_loss = nn.functional.smooth_l1_loss(
                        output["value"][group, index],
                        truth[group, index],
                        beta=0.5,
                    )
                    shrinkage = residual_l1_weight * output["residual"][
                        group, index
                    ].abs().mean()
                    route_penalty = (
                        route_entropy_weight
                        * output["route_entropy"][group, index].mean()
                    )
                    expert_penalty = (
                        expert_entropy_weight
                        * output["expert_entropy"][group, index].mean()
                    )
                    distributional_loss = output["value"].new_tensor(0.0)
                    if (
                        adapter.distributional_scale
                        and distributional_weight > 0.0
                    ):
                        error = (
                            truth[group, index]
                            - output["value"][group, index]
                        )
                        scale = output["scale"][group, index].clamp_min(0.03)
                        degrees_of_freedom = 4.0
                        distributional_loss = distributional_weight * (
                            scale.log()
                            + 0.5
                            * (degrees_of_freedom + 1.0)
                            * torch.log1p(
                                error.square()
                                / (degrees_of_freedom * scale.square())
                            )
                        ).mean()
                    horizon_losses.append(
                        value_loss
                        + shrinkage
                        + route_penalty
                        + expert_penalty
                        + distributional_loss
                    )
                if horizon_losses:
                    target_losses.append(torch.stack(horizon_losses).mean())
            if not target_losses:
                continue
            optimizer.zero_grad(set_to_none=True)
            if pcgrad and len(target_losses) > 1:
                before, after = _pcgrad(
                    target_losses, trainable_parameters, rng
                )
                conflict_before.extend(before)
                conflict_after.extend(after)
            else:
                torch.stack(target_losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            optimizer.step()
            epoch_losses.append(
                float(torch.stack([loss.detach() for loss in target_losses]).mean())
            )
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else None)
    return {
        "loss_history": losses,
        "gradient_cosine_before": conflict_before,
        "gradient_cosine_after": conflict_after,
    }


def _train_distributional_scale_heads(
    adapter,
    features,
    arrays,
    teacher,
    train_rows,
    target_indices,
    *,
    epochs,
    batch_size,
    seed,
    horizon_balanced_loss=False,
):
    """Fit heteroscedastic scale after freezing the point predictor."""

    if not adapter.distributional_scale or int(epochs) <= 0:
        return {"loss_history": [], "point_predictor_frozen": True}
    _, _, current, mask, ages, future, future_mask, _, horizon = arrays[:9]
    target_indices = np.asarray(target_indices, dtype=np.int64)
    parameters = list(adapter.scale_heads.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=1.0e-3, weight_decay=1.0e-4
    )
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(int(epochs)):
        adapter.eval()
        adapter.scale_heads.train()
        epoch_losses = []
        for batch in _adapter_batches(train_rows, batch_size, rng):
            output = adapter(
                teacher_anchor=torch.from_numpy(
                    teacher[batch][:, target_indices]
                ).float(),
                context_latent=_feature_batch(features, "context", batch),
                target_latent=_feature_batch(features, "target", batch),
                organ_latents=_feature_batch(features, "organ", batch),
                organ_presence=_feature_batch(features, "presence", batch),
                current=torch.from_numpy(
                    current[batch][:, target_indices]
                ).float(),
                current_mask=torch.from_numpy(
                    mask[batch][:, target_indices]
                ).float(),
                ages=torch.from_numpy(
                    ages[batch][:, target_indices]
                ).float(),
                horizon=torch.from_numpy(horizon[batch]).float(),
            )
            truth = torch.from_numpy(
                future[batch][:, target_indices]
            ).float()
            observed = torch.from_numpy(
                future_mask[batch][:, target_indices]
            ).float()
            batch_horizon = torch.from_numpy(horizon[batch]).float()
            target_losses = []
            for index in range(len(target_indices)):
                valid = observed[:, index] > 0.0
                if int(valid.sum()) < 2:
                    continue
                groups = [valid]
                if horizon_balanced_loss:
                    groups = [
                        valid & (batch_horizon == value)
                        for value in torch.unique(batch_horizon[valid])
                    ]
                    groups = [group for group in groups if int(group.sum()) >= 2]
                horizon_losses = []
                for group in groups:
                    error = (
                        truth[group, index]
                        - output["value"][group, index].detach()
                    )
                    scale = output["scale"][group, index].clamp_min(0.03)
                    degrees_of_freedom = 4.0
                    horizon_losses.append(
                        (
                            scale.log()
                            + 0.5
                            * (degrees_of_freedom + 1.0)
                            * torch.log1p(
                                error.square()
                                / (degrees_of_freedom * scale.square())
                            )
                        ).mean()
                    )
                if horizon_losses:
                    target_losses.append(torch.stack(horizon_losses).mean())
            if not target_losses:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(target_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else None)
    return {
        "loss_history": losses,
        "point_predictor_frozen": True,
        "optimized_parameters": "scale_heads_only",
    }


def _predict_adapter(
    adapter,
    features,
    arrays,
    teacher,
    rows,
    target_indices,
    *,
    batch_size=4096,
    disable_organ_route=False,
    disable_organ_coupling=False,
    disable_body_feedback=False,
    disable_sparse_experts=False,
    disable_target_specific_projections=False,
):
    _, _, current, mask, ages, _, _, _, horizon = arrays[:9]
    target_indices = np.asarray(target_indices, dtype=np.int64)
    values, scales, route_weights, timescales, expert_weights = [], [], [], [], []
    organ_coupling_weights, organ_presence = [], []
    target_organ_coupling = []
    target_organ_coupling_weights = []
    adapter.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = np.asarray(rows[start : start + batch_size], dtype=np.int64)
            output = adapter(
                teacher_anchor=torch.from_numpy(
                    teacher[batch][:, target_indices]
                ).float(),
                context_latent=_feature_batch(features, "context", batch),
                target_latent=_feature_batch(features, "target", batch),
                organ_latents=_feature_batch(features, "organ", batch),
                organ_presence=_feature_batch(features, "presence", batch),
                current=torch.from_numpy(
                    current[batch][:, target_indices]
                ).float(),
                current_mask=torch.from_numpy(
                    mask[batch][:, target_indices]
                ).float(),
                ages=torch.from_numpy(
                    ages[batch][:, target_indices]
                ).float(),
                horizon=torch.from_numpy(horizon[batch]).float(),
                disable_organ_route=disable_organ_route,
                disable_organ_coupling=disable_organ_coupling,
                disable_body_feedback=disable_body_feedback,
                disable_sparse_experts=disable_sparse_experts,
                disable_target_specific_projections=(
                    disable_target_specific_projections
                ),
            )
            values.append(output["value"].cpu().numpy())
            scales.append(output["scale"].cpu().numpy())
            route_weights.append(output["route_weights"].cpu().numpy())
            timescales.append(
                output["timescale_fast_weight"].cpu().numpy()
            )
            expert_weights.append(output["expert_weights"].cpu().numpy())
            organ_presence.append(output["organ_presence"].cpu().numpy())
            if "organ_coupling_weights" in output:
                organ_coupling_weights.append(
                    output["organ_coupling_weights"].cpu().numpy()
                )
                target_organ_coupling.append(
                    output["target_organ_coupling"].cpu().numpy()
                )
                target_organ_coupling_weights.append(
                    output["target_organ_coupling_weights"].cpu().numpy()
                )
    return {
        "value": np.concatenate(values),
        "scale": np.concatenate(scales),
        "route_weights": np.concatenate(route_weights),
        "timescale_fast_weight": np.concatenate(timescales),
        "expert_weights": np.concatenate(expert_weights),
        "organ_presence": np.concatenate(organ_presence),
        "organ_coupling_weights": (
            np.concatenate(organ_coupling_weights)
            if organ_coupling_weights
            else None
        ),
        "target_organ_coupling": (
            np.concatenate(target_organ_coupling)
            if target_organ_coupling
            else None
        ),
        "target_organ_coupling_weights": (
            np.concatenate(target_organ_coupling_weights)
            if target_organ_coupling_weights
            else None
        ),
    }


def _evaluate(
    frame,
    arrays,
    rows,
    target_names,
    target_indices,
    candidate,
    teacher,
    *,
    seed,
):
    scaler, _, current, _, _, future, future_mask, _, horizon = arrays[:9]
    current_raw = current * scaler.scales + scaler.medians
    future_raw = future * scaler.scales + scaler.medians
    report = {}
    rows = np.asarray(rows, dtype=np.int64)
    for local_index, (target, target_index) in enumerate(
        zip(target_names, target_indices)
    ):
        for horizon_value in sorted(set(horizon[rows])):
            selected_local = np.flatnonzero(
                (horizon[rows] == float(horizon_value))
                & (future_mask[rows, target_index] > 0.0)
            )
            if len(selected_local) < 30:
                continue
            selected_rows = rows[selected_local]
            truth = future_raw[selected_rows, target_index]
            candidate_raw = (
                candidate[selected_local, local_index]
                * scaler.scales[target_index]
                + scaler.medians[target_index]
            )
            teacher_raw = (
                teacher[selected_rows, target_index]
                * scaler.scales[target_index]
                + scaler.medians[target_index]
            )
            persistence_raw = current_raw[selected_rows, target_index]
            candidate_error = np.abs(candidate_raw - truth)
            teacher_error = np.abs(teacher_raw - truth)
            persistence_error = np.abs(persistence_raw - truth)
            observed_delta = truth - persistence_raw
            predicted_delta = candidate_raw - persistence_raw
            changed = np.abs(observed_delta) > 1.0e-8
            direction_accuracy = (
                float(
                    np.mean(
                        np.sign(predicted_delta[changed])
                        == np.sign(observed_delta[changed])
                    )
                )
                if bool(changed.any())
                else None
            )
            delta_correlation = (
                float(np.corrcoef(predicted_delta, observed_delta)[0, 1])
                if np.std(predicted_delta) > 1.0e-9
                and np.std(observed_delta) > 1.0e-9
                else None
            )
            versus_teacher = _patient_paired_bootstrap(
                frame,
                selected_rows,
                candidate_error,
                teacher_error,
                seed=seed + target_index * 101 + int(horizon_value),
            )
            versus_persistence = _patient_paired_bootstrap(
                frame,
                selected_rows,
                candidate_error,
                persistence_error,
                seed=seed + 10_000 + target_index * 101 + int(horizon_value),
            )
            report[
                f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            ] = {
                "rows": int(len(selected_rows)),
                "subjects": int(
                    frame.iloc[selected_rows]["subject_id"].nunique()
                ),
                "candidate_mae": float(candidate_error.mean()),
                "teacher_router_mae": float(teacher_error.mean()),
                "persistence_mae": float(persistence_error.mean()),
                "delta_vs_teacher": float(
                    candidate_error.mean() - teacher_error.mean()
                ),
                "delta_vs_persistence": float(
                    candidate_error.mean() - persistence_error.mean()
                ),
                "direction_accuracy": direction_accuracy,
                "delta_correlation": delta_correlation,
                "teacher_bootstrap": versus_teacher,
                "persistence_bootstrap": versus_persistence,
            }
    return report


def _evaluate_route_ablation(
    frame,
    arrays,
    rows,
    target_names,
    target_indices,
    candidate,
    no_route,
    *,
    seed,
    baseline_label="no_route",
):
    scaler, _, _, _, _, future, future_mask, _, horizon = arrays[:9]
    future_raw = future * scaler.scales + scaler.medians
    report = {}
    rows = np.asarray(rows, dtype=np.int64)
    for local_index, (target, target_index) in enumerate(
        zip(target_names, target_indices)
    ):
        for horizon_value in sorted(set(horizon[rows])):
            selected_local = np.flatnonzero(
                (horizon[rows] == float(horizon_value))
                & (future_mask[rows, target_index] > 0.0)
            )
            if len(selected_local) < 30:
                continue
            selected_rows = rows[selected_local]
            truth = future_raw[selected_rows, target_index]
            candidate_raw = (
                candidate[selected_local, local_index]
                * scaler.scales[target_index]
                + scaler.medians[target_index]
            )
            ablation_raw = (
                no_route[selected_local, local_index]
                * scaler.scales[target_index]
                + scaler.medians[target_index]
            )
            candidate_error = np.abs(candidate_raw - truth)
            ablation_error = np.abs(ablation_raw - truth)
            comparison = _patient_paired_bootstrap(
                frame,
                selected_rows,
                candidate_error,
                ablation_error,
                seed=seed + 20_000 + target_index * 101 + int(horizon_value),
            )
            report[
                f"{target}@{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            ] = {
                "rows": int(len(selected_rows)),
                "subjects": int(
                    frame.iloc[selected_rows]["subject_id"].nunique()
                ),
                "candidate_mae": float(candidate_error.mean()),
                f"{baseline_label}_mae": float(ablation_error.mean()),
                f"delta_vs_{baseline_label}": float(
                    candidate_error.mean() - ablation_error.mean()
                ),
                f"{baseline_label}_bootstrap": comparison,
            }
    return report


def _select_coupling_paths(
    frame,
    arrays,
    calibration_rows,
    target_names,
    target_indices,
    path_predictions,
    *,
    seed,
):
    """Select one hierarchy path per cell using the calibration set's selection patients only.

    ``path_predictions`` are aligned with ``calibration_rows``; conformal
    patients (``osler_jepa.patient_split``) never influence the choice.
    """

    choosing = selection_patients(frame.iloc[np.asarray(calibration_rows, dtype=np.int64)]["subject_id"].to_numpy())
    calibration_rows = np.asarray(calibration_rows, dtype=np.int64)[choosing]
    path_predictions = {path: np.asarray(values)[choosing] for path, values in path_predictions.items()}
    system_only = path_predictions["system_only"]
    comparisons = {}
    for path in ("direct_organ", "body_mediated", "full"):
        comparisons[path] = _evaluate_route_ablation(
            frame,
            arrays,
            calibration_rows,
            target_names,
            target_indices,
            path_predictions[path],
            system_only,
            seed=seed + {"direct_organ": 1, "body_mediated": 2, "full": 3}[path],
            baseline_label="system_only",
        )
    cells = sorted(
        {
            cell
            for report in comparisons.values()
            for cell in report
        }
    )
    selection = {}
    for cell in cells:
        eligible = []
        evidence = {}
        for path, report in comparisons.items():
            value = report.get(cell)
            if value is None:
                continue
            bootstrap = value["system_only_bootstrap"]
            delta = float(value["delta_vs_system_only"])
            evidence[path] = {
                "delta_vs_system_only": delta,
                "patient_equal_delta": bootstrap["patient_equal_delta"],
                "bootstrap_low": bootstrap["bootstrap_low"],
                "bootstrap_high": bootstrap["bootstrap_high"],
                "pass": bool(bootstrap["pass"]),
            }
            if bootstrap["pass"] and delta < 0.0:
                eligible.append((delta, path))
        selected_path = min(eligible)[1] if eligible else "system_only"
        selection[cell] = {
            "selected_path": selected_path,
            "selection_rows_are_calibration_selection_patients_only": True,
            "test_labels_used_for_selection": False,
            "evidence": evidence,
        }
    return selection


def _export_predictions(frame, arrays, rows, target_names, target_indices, teacher, models, split: str = "test") -> pd.DataFrame:
    """Labelled-row predictions of the frozen expert anchor and every matched hierarchy model.

    ``models`` maps each ``HIERARCHY_MODELS`` name to its predictions on ``rows``
    (normalized space).  Path choice and matched-ablation tests happen downstream
    (``jepa_gate.py``) on selection patients and per test database.
    ``hours_since_onset`` is the loader's (complete event stream: every anchor of
    the stay, labelled or not); ``label_available_hr`` = horizon + |measurement
    time - target time| is when the outcome could first be used for online adaptation.
    """

    scaler, _, current, _, _, future, future_mask, _, horizon = arrays[:9]
    rows = np.asarray(rows, dtype=np.int64)
    meta = frame.iloc[rows].reset_index(drop=True)
    onset = frame["hours_since_onset"].to_numpy(dtype=float)
    pieces = []
    for local_index, (target, target_index) in enumerate(zip(target_names, target_indices)):
        labelled = np.flatnonzero(future_mask[rows, target_index] > 0.0)
        if not len(labelled):
            continue
        selected = rows[labelled]
        scale, median = scaler.scales[target_index], scaler.medians[target_index]
        hours = horizon[selected]
        pieces.append(pd.DataFrame({
            "subject_id": meta.loc[labelled, "subject_id"].astype(str).to_numpy(),
            "stay_id": meta.loc[labelled, "stay_id"].astype(str).to_numpy(),
            "hospitalid": meta.loc[labelled, "hospitalid"].astype(str).to_numpy() if "hospitalid" in meta else "unknown",
            "careunit": meta.loc[labelled, "careunit"].astype(str).to_numpy() if "careunit" in meta else "unknown",
            "database": meta.loc[labelled, "database"].astype(str).to_numpy() if "database" in meta else "unknown",
            "anchor_time": pd.to_datetime(meta.loc[labelled, "t"]).to_numpy(),
            "horizon_hours": hours,
            "hours_since_onset": onset[selected],
            "label_available_hr": hours + np.abs(
                pd.to_numeric(frame[f"future_delta_t_hr_{target}"], errors="coerce").to_numpy()[selected] - hours),
            "target": target,
            "cell": [f"{target}@{int(h) if float(h).is_integer() else h}h" for h in hours],
            "current": current[selected, target_index] * scale + median,
            "future": future[selected, target_index] * scale + median,
            "anchor": teacher[selected, target_index] * scale + median,
            **{name: np.asarray(values)[labelled, local_index] * scale + median for name, values in models.items()},
            "split": split,
        }))
    return pd.concat(pieces, ignore_index=True)


def _compose_coupling_path_prediction(
    rows,
    arrays,
    target_names,
    path_predictions,
    selection,
):
    """Apply calibration-selected paths without consulting test labels."""

    horizon = arrays[8]
    rows = np.asarray(rows, dtype=np.int64)
    output = np.asarray(path_predictions["system_only"]).copy()
    for local_index, target in enumerate(target_names):
        for horizon_value in sorted(set(horizon[rows])):
            label = int(horizon_value) if float(horizon_value).is_integer() else horizon_value
            cell = f"{target}@{label}h"
            path = selection.get(cell, {}).get("selected_path", "system_only")
            selected_local = np.flatnonzero(
                horizon[rows] == float(horizon_value)
            )
            output[selected_local, local_index] = path_predictions[path][
                selected_local, local_index
            ]
    return output


def _adaptive_conformal_quantile(
    frame,
    calibration_rows,
    residuals,
    *,
    seed,
    target_coverage=0.90,
    candidate_coverages=None,
    final_fraction=0.40,
    cv_folds=3,
):
    """Tune interval width on calibration patients without seeing test labels.

    A patient-disjoint subset is reserved for the final conformal quantile.
    The remaining calibration patients choose a nominal coverage by grouped
    cross-validation.  This corrects systematic over-coverage while keeping
    the evaluation set completely untouched.
    """

    rows = np.asarray(calibration_rows, dtype=np.int64)
    scores = np.asarray(residuals, dtype=np.float64)
    finite = np.isfinite(scores)
    rows = rows[finite]
    scores = scores[finite]
    if candidate_coverages is None:
        candidate_coverages = np.round(
            np.arange(0.75, 0.901, 0.01), 2
        )
    candidate_coverages = tuple(
        float(value) for value in candidate_coverages
    )
    fallback_q, fallback_rank = _finite_sample_conformal_quantile(
        scores, coverage=target_coverage
    )
    fallback = {
        "method": "fixed_split_conformal_fallback",
        "selected_nominal_coverage": float(target_coverage),
        "target_empirical_coverage": float(target_coverage),
        "tuning_rows": 0,
        "final_calibration_rows": int(len(scores)),
        "cv_folds": 0,
        "cv_coverage": None,
        "test_labels_used": False,
        "patient_disjoint_final_calibration": False,
    }
    if len(scores) < 90:
        return fallback_q, fallback_rank, fallback

    subjects = (
        frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    )
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < max(12, cv_folds * 3):
        return fallback_q, fallback_rank, fallback

    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    final_count = max(
        4,
        min(
            len(shuffled) - cv_folds * 2,
            int(round(len(shuffled) * float(final_fraction))),
        ),
    )
    if final_count <= 0 or len(shuffled) - final_count < cv_folds * 2:
        return fallback_q, fallback_rank, fallback
    final_subjects = set(shuffled[:final_count].tolist())
    final_mask = np.asarray(
        [subject in final_subjects for subject in subjects], dtype=bool
    )
    tuning_scores = scores[~final_mask]
    tuning_subjects = subjects[~final_mask]
    final_scores = scores[final_mask]
    if len(tuning_scores) < 45 or len(final_scores) < 30:
        return fallback_q, fallback_rank, fallback

    fold_subjects = np.array_split(
        np.unique(tuning_subjects), int(cv_folds)
    )
    coverage_by_candidate = {}
    for nominal in candidate_coverages:
        covered_parts = []
        for heldout_subjects in fold_subjects:
            heldout_set = set(heldout_subjects.tolist())
            check_mask = np.asarray(
                [
                    subject in heldout_set
                    for subject in tuning_subjects
                ],
                dtype=bool,
            )
            fit_mask = ~check_mask
            if int(fit_mask.sum()) < 30 or int(check_mask.sum()) < 10:
                continue
            quantile, _ = _finite_sample_conformal_quantile(
                tuning_scores[fit_mask], coverage=nominal
            )
            covered_parts.append(
                tuning_scores[check_mask] <= float(quantile)
            )
        if len(covered_parts) != int(cv_folds):
            continue
        coverage_by_candidate[nominal] = float(
            np.concatenate(covered_parts).mean()
        )
    if not coverage_by_candidate:
        return fallback_q, fallback_rank, fallback

    selected = min(
        coverage_by_candidate,
        key=lambda nominal: (
            abs(coverage_by_candidate[nominal] - target_coverage),
            abs(nominal - target_coverage),
        ),
    )
    q90, rank = _finite_sample_conformal_quantile(
        final_scores, coverage=selected
    )
    metadata = {
        "method": "patient_grouped_cv_adaptive_split_conformal",
        "selected_nominal_coverage": float(selected),
        "target_empirical_coverage": float(target_coverage),
        "tuning_rows": int(len(tuning_scores)),
        "final_calibration_rows": int(len(final_scores)),
        "tuning_subjects": int(len(np.unique(tuning_subjects))),
        "final_calibration_subjects": int(
            len(np.unique(subjects[final_mask]))
        ),
        "cv_folds": int(cv_folds),
        "cv_coverage": float(coverage_by_candidate[selected]),
        "candidate_cv_coverage": {
            f"{nominal:.2f}": coverage
            for nominal, coverage in coverage_by_candidate.items()
        },
        "test_labels_used": False,
        "patient_disjoint_final_calibration": True,
    }
    return q90, rank, metadata


def _patient_cluster_conformal_quantile(
    frame,
    calibration_rows,
    residuals,
    *,
    coverage=0.90,
):
    """Calibrate a target/horizon interval with equal patient influence.

    Each patient contributes total weight one regardless of how many
    measurements they have.  This estimates coverage for a randomly selected
    patient rather than letting long, densely measured admissions dominate.
    """

    rows = np.asarray(calibration_rows, dtype=np.int64)
    scores = np.asarray(residuals, dtype=np.float64)
    finite = np.isfinite(scores)
    rows = rows[finite]
    scores = scores[finite]
    if not len(scores):
        raise ValueError("Patient-cluster conformal requires finite residuals")
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique, inverse, counts = np.unique(
        subjects, return_inverse=True, return_counts=True
    )
    row_weights = 1.0 / counts[inverse].astype(np.float64)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_weights = row_weights[order]
    patient_corrected_coverage = min(
        1.0,
        float(coverage) * (len(unique) + 1.0) / max(len(unique), 1),
    )
    threshold = patient_corrected_coverage * float(sorted_weights.sum())
    rank_index = int(
        np.searchsorted(
            np.cumsum(sorted_weights), threshold, side="left"
        )
    )
    rank_index = min(rank_index, len(sorted_scores) - 1)
    quantile = float(sorted_scores[rank_index])
    return quantile, rank_index + 1, {
        "method": "target_horizon_patient_cluster_conformal",
        "selected_nominal_coverage": float(coverage),
        "patient_finite_sample_coverage": float(
            patient_corrected_coverage
        ),
        "target_empirical_coverage": 0.90,
        "calibration_rows": int(len(scores)),
        "calibration_subjects": int(len(unique)),
        "q90_rank": int(rank_index + 1),
        "equal_total_weight_per_patient": True,
        "test_labels_used": False,
    }


def _adaptive_patient_cluster_conformal_quantile(
    frame,
    calibration_rows,
    residuals,
    *,
    seed,
    target_coverage=0.90,
    candidate_coverages=None,
    final_fraction=0.40,
    cv_folds=3,
):
    """Tune patient-cluster conformal width without using test labels."""

    rows = np.asarray(calibration_rows, dtype=np.int64)
    scores = np.asarray(residuals, dtype=np.float64)
    finite = np.isfinite(scores)
    rows = rows[finite]
    scores = scores[finite]
    fallback_q, fallback_rank, fallback = (
        _patient_cluster_conformal_quantile(
            frame, rows, scores, coverage=target_coverage
        )
    )
    fallback = {
        **fallback,
        "method": "adaptive_patient_cluster_fallback",
        "selected_nominal_coverage": float(target_coverage),
    }
    if candidate_coverages is None:
        candidate_coverages = np.round(np.arange(0.75, 0.901, 0.01), 2)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    if len(scores) < 90 or len(unique_subjects) < max(15, cv_folds * 4):
        return fallback_q, fallback_rank, fallback

    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    final_count = max(
        5,
        min(
            len(shuffled) - cv_folds * 3,
            int(round(len(shuffled) * float(final_fraction))),
        ),
    )
    if final_count <= 0:
        return fallback_q, fallback_rank, fallback
    final_subjects = set(shuffled[:final_count].tolist())
    final_mask = np.asarray(
        [subject in final_subjects for subject in subjects], dtype=bool
    )
    tuning_rows = rows[~final_mask]
    tuning_scores = scores[~final_mask]
    tuning_subjects = subjects[~final_mask]
    final_rows = rows[final_mask]
    final_scores = scores[final_mask]
    if len(tuning_scores) < 45 or len(final_scores) < 30:
        return fallback_q, fallback_rank, fallback

    fold_subjects = np.array_split(np.unique(tuning_subjects), int(cv_folds))
    coverage_by_candidate = {}
    for nominal in (float(value) for value in candidate_coverages):
        patient_coverages = []
        for heldout_subjects in fold_subjects:
            heldout_set = set(heldout_subjects.tolist())
            check_mask = np.asarray(
                [subject in heldout_set for subject in tuning_subjects],
                dtype=bool,
            )
            fit_mask = ~check_mask
            if int(fit_mask.sum()) < 30 or int(check_mask.sum()) < 10:
                continue
            quantile, _, _ = _patient_cluster_conformal_quantile(
                frame,
                tuning_rows[fit_mask],
                tuning_scores[fit_mask],
                coverage=nominal,
            )
            fold = pd.DataFrame(
                {
                    "subject_id": tuning_subjects[check_mask],
                    "covered": tuning_scores[check_mask] <= quantile,
                }
            )
            patient_coverages.extend(
                fold.groupby("subject_id", sort=False)["covered"]
                .mean()
                .to_list()
            )
        if patient_coverages:
            coverage_by_candidate[nominal] = float(
                np.mean(patient_coverages)
            )
    if not coverage_by_candidate:
        return fallback_q, fallback_rank, fallback
    selected = min(
        coverage_by_candidate,
        key=lambda nominal: (
            abs(coverage_by_candidate[nominal] - target_coverage),
            abs(nominal - target_coverage),
        ),
    )
    quantile, rank, final_metadata = _patient_cluster_conformal_quantile(
        frame, final_rows, final_scores, coverage=selected
    )
    return quantile, rank, {
        **final_metadata,
        "method": "patient_grouped_cv_adaptive_patient_cluster_conformal",
        "selected_nominal_coverage": float(selected),
        "target_empirical_coverage": float(target_coverage),
        "tuning_rows": int(len(tuning_rows)),
        "tuning_subjects": int(len(np.unique(tuning_subjects))),
        "final_calibration_rows": int(len(final_rows)),
        "final_calibration_subjects": int(len(np.unique(subjects[final_mask]))),
        "cv_folds": int(cv_folds),
        "cv_coverage": float(coverage_by_candidate[selected]),
        "candidate_cv_coverage": {
            f"{nominal:.2f}": coverage
            for nominal, coverage in coverage_by_candidate.items()
        },
        "patient_disjoint_final_calibration": True,
        "test_labels_used": False,
    }


def _crossfit_adaptive_patient_cluster_conformal_quantile(
    frame,
    calibration_rows,
    residuals,
    *,
    seed,
    target_coverage=0.90,
    candidate_coverages=None,
    cv_folds=5,
):
    """Select nominal coverage by patient CV, then use all calibration rows.

    Cross-validation chooses the nominal level without looking at forecast
    test labels.  The final patient-cluster quantile uses the complete
    calibration pool, avoiding the variance introduced by throwing most
    calibration patients away after selection.
    """

    rows = np.asarray(calibration_rows, dtype=np.int64)
    scores = np.asarray(residuals, dtype=np.float64)
    finite = np.isfinite(scores)
    rows = rows[finite]
    scores = scores[finite]
    fallback_q, fallback_rank, fallback = (
        _patient_cluster_conformal_quantile(
            frame, rows, scores, coverage=target_coverage
        )
    )
    fallback = {
        **fallback,
        "method": "crossfit_adaptive_patient_cluster_fallback",
        "selected_nominal_coverage": float(target_coverage),
    }
    if candidate_coverages is None:
        candidate_coverages = np.round(np.arange(0.80, 0.911, 0.01), 2)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    if len(scores) < 120 or len(unique_subjects) < max(25, cv_folds * 5):
        return fallback_q, fallback_rank, fallback

    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, int(cv_folds))
    coverage_by_candidate = {}
    for nominal in (float(value) for value in candidate_coverages):
        patient_coverages = []
        for heldout_subjects in folds:
            heldout = set(heldout_subjects.tolist())
            check = np.asarray(
                [subject in heldout for subject in subjects], dtype=bool
            )
            fit = ~check
            if int(fit.sum()) < 60 or int(check.sum()) < 20:
                continue
            quantile, _, _ = _patient_cluster_conformal_quantile(
                frame, rows[fit], scores[fit], coverage=nominal
            )
            fold = pd.DataFrame(
                {
                    "subject_id": subjects[check],
                    "covered": scores[check] <= quantile,
                }
            )
            patient_coverages.extend(
                fold.groupby("subject_id", sort=False)["covered"]
                .mean()
                .to_list()
            )
        if patient_coverages:
            coverage_by_candidate[nominal] = float(
                np.mean(patient_coverages)
            )
    if not coverage_by_candidate:
        return fallback_q, fallback_rank, fallback
    selected = min(
        coverage_by_candidate,
        key=lambda nominal: (
            abs(coverage_by_candidate[nominal] - target_coverage),
            abs(nominal - target_coverage),
        ),
    )
    quantile, rank, metadata = _patient_cluster_conformal_quantile(
        frame, rows, scores, coverage=selected
    )
    return quantile, rank, {
        **metadata,
        "method": "patient_grouped_crossfit_adaptive_conformal",
        "selected_nominal_coverage": float(selected),
        "target_empirical_coverage": float(target_coverage),
        "calibration_rows": int(len(rows)),
        "calibration_subjects": int(len(unique_subjects)),
        "cv_folds": int(cv_folds),
        "cv_coverage": float(coverage_by_candidate[selected]),
        "candidate_cv_coverage": {
            f"{nominal:.2f}": coverage
            for nominal, coverage in coverage_by_candidate.items()
        },
        "all_calibration_rows_used_after_crossfit_selection": True,
        "test_labels_used": False,
    }


def _patient_coverage_diagnostics(
    frame,
    rows,
    covered,
    *,
    seed,
    bootstrap_samples=1000,
):
    """Return row and patient-equalized coverage with a patient bootstrap."""

    rows = np.asarray(rows, dtype=np.int64)
    covered = np.asarray(covered, dtype=np.float64)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    patient_coverage = (
        pd.DataFrame({"subject_id": subjects, "covered": covered})
        .groupby("subject_id", sort=False)["covered"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    rng = np.random.default_rng(seed)
    if len(patient_coverage):
        indices = rng.integers(
            0,
            len(patient_coverage),
            size=(int(bootstrap_samples), len(patient_coverage)),
        )
        draws = patient_coverage[indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
    else:
        lower, upper = np.nan, np.nan
    return {
        "row_coverage": float(covered.mean()),
        "patient_equalized_coverage": float(patient_coverage.mean()),
        "patient_coverage_ci95": [float(lower), float(upper)],
        "test_subjects": int(len(patient_coverage)),
    }


def _interval_personalization_diagnostics(
    frame,
    calibration_rows,
    test_rows,
    target,
    calibration_normalized_residual,
    calibration_scale,
    test_scale,
    test_point,
    test_observed,
    *,
    seed,
    coverage=0.90,
):
    """Audit selective and causal-order online interval personalization."""

    calibration_rows = np.asarray(calibration_rows, dtype=np.int64)
    test_rows = np.asarray(test_rows, dtype=np.int64)
    calibration_subjects = (
        frame.iloc[calibration_rows]["subject_id"].astype(str).to_numpy()
    )
    test_subjects = frame.iloc[test_rows]["subject_id"].astype(str).to_numpy()
    global_quantile = patient_weighted_quantile(
        calibration_normalized_residual,
        calibration_subjects,
        coverage=coverage,
    )
    global_width = np.maximum(test_scale, 1.0e-8) * global_quantile

    def summarize(rows, widths, *, reference_widths, extra=None):
        rows = np.asarray(rows, dtype=np.int64)
        widths = np.asarray(widths, dtype=np.float64)
        reference_widths = np.asarray(reference_widths, dtype=np.float64)
        if not len(rows):
            return {
                "test_n": 0,
                "test_subjects": 0,
                "pass": False,
                **(extra or {}),
            }
        row_to_local = {
            int(row): index for index, row in enumerate(test_rows.tolist())
        }
        local_indices = np.asarray(
            [row_to_local[int(row)] for row in rows], dtype=np.int64
        )
        covered = (
            np.abs(test_observed[local_indices] - test_point[local_indices])
            <= widths
        )
        diagnostics = _patient_coverage_diagnostics(
            frame,
            rows,
            covered,
            seed=seed,
        )
        median_width = float(np.median(widths))
        reference_median = float(np.median(reference_widths))
        return {
            "test_n": int(len(rows)),
            "test_subjects": diagnostics["test_subjects"],
            "median_half_width": median_width,
            "reference_global_median_half_width": reference_median,
            "median_width_change_fraction": float(
                median_width / max(reference_median, 1.0e-8) - 1.0
            ),
            "row_coverage": diagnostics["row_coverage"],
            "patient_equalized_coverage": diagnostics[
                "patient_equalized_coverage"
            ],
            "patient_coverage_ci95": diagnostics["patient_coverage_ci95"],
            "pass": bool(
                diagnostics["test_subjects"] >= 20
                and 0.87
                <= diagnostics["patient_equalized_coverage"]
                <= 0.93
            ),
            **(extra or {}),
        }

    try:
        selective = selective_interval_widths(
            calibration_normalized_residual,
            calibration_scale,
            calibration_subjects,
            test_scale,
            coverage=coverage,
        )
    except ValueError as error:
        selective = None
        selected_indices = np.asarray([], dtype=np.int64)
        selected_rows = np.asarray([], dtype=np.int64)
        selective_summary = {
            "available": False,
            "reason": str(error),
            "test_n": 0,
            "test_subjects": 0,
            "pass": False,
        }
    else:
        selected_indices = np.flatnonzero(selective.selected)
        selected_rows = test_rows[selected_indices]
        selective_summary = summarize(
            selected_rows,
            selective.widths[selected_indices],
            reference_widths=global_width[selected_indices],
            extra={
                "available": True,
                "selection_rate": float(selective.selected.mean()),
                "abstained_n": int((~selective.selected).sum()),
                "scale_threshold": selective.scale_threshold,
                "normalized_quantile": selective.normalized_quantile,
                "high_risk_policy": "abstain_no_interval",
                "selection_uses_test_labels": False,
            },
        )

    future_time_column = f"future_event_time_{target}"
    if future_time_column not in frame:
        online_summary = {
            "available": False,
            "reason": f"missing_{future_time_column}",
            "pass": False,
        }
        selective_online_summary = {
            "available": False,
            "reason": f"missing_{future_time_column}",
            "pass": False,
        }
    else:
        online = online_adaptive_interval_widths(
            calibration_normalized_residual,
            calibration_subjects,
            test_subjects,
            frame.iloc[test_rows]["anchor_time"],
            frame.iloc[test_rows][future_time_column],
            test_scale,
            test_point,
            test_observed,
            coverage=coverage,
        )
        online_summary = summarize(
            test_rows,
            online.widths,
            reference_widths=global_width,
            extra={
                "available": True,
                "personalized_n": int(online.personalized.sum()),
                "personalized_rate": float(online.personalized.mean()),
                "median_prior_resolved_count": float(
                    np.median(online.prior_resolved_count)
                ),
                "median_effective_alpha": float(
                    np.median(online.effective_alpha)
                ),
                "current_or_future_test_labels_used": False,
                "historically_resolved_test_outcomes_used": True,
                "first_forecast_uses_population_interval": True,
                "duplicate_future_measurements_deduplicated": True,
            },
        )
        if selective is None:
            selective_online_summary = {
                "available": False,
                "reason": selective_summary["reason"],
                "pass": False,
            }
        else:
            selective_online_summary = summarize(
                selected_rows,
                online.widths[selected_indices],
                reference_widths=global_width[selected_indices],
                extra={
                    "available": True,
                    "selection_rate": float(selective.selected.mean()),
                    "abstained_n": int((~selective.selected).sum()),
                    "personalized_selected_n": int(
                        online.personalized[selected_indices].sum()
                    ),
                    "high_risk_policy": "abstain_no_interval",
                    "current_or_future_test_labels_used": False,
                    "historically_resolved_test_outcomes_used": True,
                },
            )

    return {
        "schema": "target_horizon_interval_personalization_audit.v1",
        "global_distributional_reference": {
            "normalized_quantile": float(global_quantile),
            "median_half_width": float(np.median(global_width)),
        },
        "selective_low_risk": selective_summary,
        "online_patient_adaptive": online_summary,
        "selective_online_patient_adaptive": selective_online_summary,
        "physiology_latent_detached": True,
        "causal_claim_allowed": False,
        "runtime_promotion_allowed": False,
    }


def _conformal_scale_features(
    arrays,
    rows,
    target_index,
    prediction,
):
    _, _, current, mask, ages = arrays[:5]
    rows = np.asarray(rows, dtype=np.int64)
    values = np.nan_to_num(
        np.asarray(current[rows], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    observed = np.asarray(mask[rows], dtype=np.float64)
    age = np.asarray(ages[rows], dtype=np.float64)
    age = np.log1p(np.clip(np.nan_to_num(age, nan=168.0), 0.0, 168.0))
    age /= np.log1p(168.0)
    point = np.asarray(prediction, dtype=np.float64).reshape(-1, 1)
    target_current = values[:, [target_index]]
    predicted_change = np.abs(point - target_current)
    return np.concatenate(
        [
            values * observed,
            observed,
            age * observed,
            point,
            target_current,
            predicted_change,
        ],
        axis=1,
    )


def _fit_residual_scale_model(features, residuals, ridge=10.0):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(residuals, dtype=np.float64)
    center = np.nanmedian(x, axis=0)
    x = np.nan_to_num(x, nan=center)
    scale = np.nanstd(x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
    standardized = np.clip((x - center) / scale, -8.0, 8.0)
    design = np.concatenate(
        [np.ones((len(standardized), 1)), standardized], axis=1
    )
    positive = y[np.isfinite(y) & (y > 0.0)]
    floor = (
        max(float(np.quantile(positive, 0.02)), 1.0e-6)
        if len(positive)
        else 1.0e-6
    )
    log_y = np.log(np.maximum(np.nan_to_num(y, nan=floor), floor))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ log_y
    lower = float(np.quantile(np.maximum(y, floor), 0.05))
    upper = float(np.quantile(np.maximum(y, floor), 0.95))
    return {
        "center": center,
        "scale": scale,
        "beta": beta,
        "lower": max(lower, floor),
        "upper": max(upper, lower, floor),
    }


def _predict_residual_scale(model, features):
    x = np.asarray(features, dtype=np.float64)
    x = np.nan_to_num(x, nan=model["center"])
    standardized = np.clip(
        (x - model["center"]) / model["scale"], -8.0, 8.0
    )
    design = np.concatenate(
        [np.ones((len(standardized), 1)), standardized], axis=1
    )
    prediction = np.exp(
        np.clip(design @ model["beta"], -20.0, 20.0)
    )
    return np.clip(prediction, model["lower"], model["upper"])


def _distributional_mondrian_widths(
    frame,
    calibration_rows,
    test_rows,
    normalized_residual,
    calibration_scale,
    test_scale,
    calibration_current,
    test_current,
    calibration_observed,
    test_observed,
    *,
    coverage=0.90,
):
    """Calibrate scale-normalized errors within current-state strata."""

    calibration_rows = np.asarray(calibration_rows, dtype=np.int64)
    test_rows = np.asarray(test_rows, dtype=np.int64)
    normalized_residual = np.asarray(normalized_residual, dtype=np.float64)
    calibration_scale = np.asarray(calibration_scale, dtype=np.float64)
    test_scale = np.asarray(test_scale, dtype=np.float64)
    calibration_current = np.asarray(calibration_current, dtype=np.float64)
    test_current = np.asarray(test_current, dtype=np.float64)
    calibration_observed = np.asarray(calibration_observed, dtype=bool)
    test_observed = np.asarray(test_observed, dtype=bool)

    global_quantile, global_rank, global_metadata = (
        _patient_cluster_conformal_quantile(
            frame,
            calibration_rows,
            normalized_residual,
            coverage=coverage,
        )
    )
    finite = calibration_observed & np.isfinite(calibration_current)
    if int(finite.sum()) >= 90:
        cutpoints = np.unique(
            np.quantile(calibration_current[finite], [1.0 / 3.0, 2.0 / 3.0])
        )
    else:
        cutpoints = np.asarray([], dtype=np.float64)
    calibration_regime = np.digitize(calibration_current, cutpoints)
    test_regime = np.digitize(test_current, cutpoints)
    missing_regime = int(len(cutpoints) + 1)
    calibration_regime[~finite] = missing_regime
    test_regime[
        ~(test_observed & np.isfinite(test_current))
    ] = missing_regime

    widths = np.empty(len(test_rows), dtype=np.float64)
    strata = {}
    for regime in np.unique(test_regime):
        calibration_mask = calibration_regime == regime
        test_mask = test_regime == regime
        stratum_rows = calibration_rows[calibration_mask]
        stratum_subjects = frame.iloc[stratum_rows][
            "subject_id"
        ].astype(str).nunique()
        if int(calibration_mask.sum()) >= 30 and int(stratum_subjects) >= 20:
            quantile, rank, metadata = _patient_cluster_conformal_quantile(
                frame,
                stratum_rows,
                normalized_residual[calibration_mask],
                coverage=coverage,
            )
            fallback = False
        else:
            quantile, rank, metadata = (
                global_quantile,
                global_rank,
                global_metadata,
            )
            fallback = True
        widths[test_mask] = float(quantile) * test_scale[test_mask]
        strata[str(int(regime))] = {
            "calibration_rows": int(calibration_mask.sum()),
            "calibration_subjects": int(stratum_subjects),
            "test_rows": int(test_mask.sum()),
            "normalized_quantile": float(quantile),
            "q90_rank": int(rank),
            "global_fallback": fallback,
            "equal_total_weight_per_patient": bool(
                metadata.get("equal_total_weight_per_patient", False)
            ),
        }
    return widths, {
        "method": (
            "target_distributional_scale_current_state_mondrian_"
            "patient_cluster_conformal"
        ),
        "cutpoints_normalized": [float(value) for value in cutpoints],
        "global_normalized_q90": float(global_quantile),
        "q90_rank": int(global_rank),
        "strata": strata,
        "median_calibration_predicted_scale": float(
            np.median(calibration_scale)
        ),
        "median_test_predicted_scale": float(np.median(test_scale)),
        "test_labels_used": False,
        "cutpoints_fit_on_calibration_only": True,
    }


def _distributional_adaptive_widths(
    frame,
    calibration_rows,
    normalized_residual,
    calibration_scale,
    test_scale,
    *,
    seed,
    coverage=0.90,
):
    """Select a distributional conformal level using calibration patients.

    Predicted residual scales preserve target-specific heteroscedasticity. A
    patient-grouped calibration-only CV then selects the nominal quantile, and
    a disjoint final calibration subset estimates the deployed quantile. Test
    labels are never used to choose either value.
    """

    calibration_rows = np.asarray(calibration_rows, dtype=np.int64)
    normalized_residual = np.asarray(normalized_residual, dtype=np.float64)
    calibration_scale = np.asarray(calibration_scale, dtype=np.float64)
    test_scale = np.asarray(test_scale, dtype=np.float64)
    quantile, rank, metadata = _adaptive_patient_cluster_conformal_quantile(
        frame,
        calibration_rows,
        normalized_residual,
        seed=seed,
        target_coverage=coverage,
    )
    return float(quantile) * test_scale, {
        **metadata,
        "method": (
            "target_distributional_scale_adaptive_patient_cluster_conformal"
        ),
        "normalized_quantile": float(quantile),
        "q90_rank": int(rank),
        "median_calibration_predicted_scale": float(
            np.median(calibration_scale)
        ),
        "median_test_predicted_scale": float(np.median(test_scale)),
        "scale_head_test_labels_used": False,
        "test_labels_used": False,
    }


def _distributional_asymmetric_widths(
    frame,
    calibration_rows,
    lower_scores,
    upper_scores,
    test_scale,
    *,
    coverage=0.90,
):
    """Return patient-clustered, scale-normalized asymmetric intervals.

    The two tails each receive half of the total error budget.  Signed lower
    and upper nonconformity scores are calibrated independently, which keeps
    skewed residuals from being forced into a symmetric interval.  Only the
    calibration patients are used to estimate either bound.
    """

    calibration_rows = np.asarray(calibration_rows, dtype=np.int64)
    lower_scores = np.asarray(lower_scores, dtype=np.float64)
    upper_scores = np.asarray(upper_scores, dtype=np.float64)
    test_scale = np.asarray(test_scale, dtype=np.float64)
    tail_coverage = 1.0 - (1.0 - float(coverage)) / 2.0
    lower_q, lower_rank, lower_metadata = (
        _patient_cluster_conformal_quantile(
            frame,
            calibration_rows,
            lower_scores,
            coverage=tail_coverage,
        )
    )
    upper_q, upper_rank, upper_metadata = (
        _patient_cluster_conformal_quantile(
            frame,
            calibration_rows,
            upper_scores,
            coverage=tail_coverage,
        )
    )
    return (
        float(lower_q) * test_scale,
        float(upper_q) * test_scale,
        {
            "method": (
                "target_distributional_scale_asymmetric_patient_cluster_"
                "conformal"
            ),
            "tail_coverage": float(tail_coverage),
            "lower_normalized_quantile": float(lower_q),
            "upper_normalized_quantile": float(upper_q),
            "lower_q90_rank": int(lower_rank),
            "upper_q90_rank": int(upper_rank),
            "q90_rank": int(max(lower_rank, upper_rank)),
            "equal_total_weight_per_patient": bool(
                lower_metadata.get("equal_total_weight_per_patient", False)
                and upper_metadata.get("equal_total_weight_per_patient", False)
            ),
            "test_labels_used": False,
        },
    )


def _distributional_asymmetric_adaptive_widths(
    frame,
    calibration_rows,
    signed_normalized_error,
    test_scale,
    *,
    seed,
    coverage=0.90,
    candidate_coverages=None,
    cv_folds=5,
):
    """Select asymmetric tail coverage by calibration-patient CV only."""

    rows = np.asarray(calibration_rows, dtype=np.int64)
    error = np.asarray(signed_normalized_error, dtype=np.float64)
    test_scale = np.asarray(test_scale, dtype=np.float64)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    if candidate_coverages is None:
        candidate_coverages = np.round(np.arange(0.80, 0.911, 0.01), 2)
    fallback = _distributional_asymmetric_widths(
        frame,
        rows,
        -error,
        error,
        test_scale,
        coverage=coverage,
    )
    if len(rows) < 120 or len(unique_subjects) < max(25, cv_folds * 5):
        lower, upper, metadata = fallback
        return lower, upper, {
            **metadata,
            "method": "distributional_asymmetric_adaptive_fallback",
            "selected_nominal_coverage": float(coverage),
        }

    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, int(cv_folds))
    coverage_by_candidate = {}
    for nominal in (float(value) for value in candidate_coverages):
        patient_coverages = []
        for heldout_subjects in folds:
            heldout = set(heldout_subjects.tolist())
            check = np.asarray(
                [subject in heldout for subject in subjects], dtype=bool
            )
            fit = ~check
            if int(fit.sum()) < 60 or int(check.sum()) < 20:
                continue
            lower, upper, _ = _distributional_asymmetric_widths(
                frame,
                rows[fit],
                -error[fit],
                error[fit],
                np.ones(int(check.sum()), dtype=np.float64),
                coverage=nominal,
            )
            covered = (error[check] >= -lower) & (error[check] <= upper)
            fold = pd.DataFrame(
                {
                    "subject_id": subjects[check],
                    "covered": covered,
                }
            )
            patient_coverages.extend(
                fold.groupby("subject_id", sort=False)["covered"]
                .mean()
                .to_list()
            )
        if patient_coverages:
            coverage_by_candidate[nominal] = float(
                np.mean(patient_coverages)
            )
    if not coverage_by_candidate:
        lower, upper, metadata = fallback
        return lower, upper, {
            **metadata,
            "method": "distributional_asymmetric_adaptive_fallback",
            "selected_nominal_coverage": float(coverage),
        }
    selected = min(
        coverage_by_candidate,
        key=lambda nominal: (
            abs(coverage_by_candidate[nominal] - float(coverage)),
            abs(nominal - float(coverage)),
        ),
    )
    lower, upper, metadata = _distributional_asymmetric_widths(
        frame,
        rows,
        -error,
        error,
        test_scale,
        coverage=selected,
    )
    return lower, upper, {
        **metadata,
        "method": "distributional_asymmetric_patient_crossfit_adaptive",
        "selected_nominal_coverage": float(selected),
        "target_empirical_coverage": float(coverage),
        "cv_folds": int(cv_folds),
        "cv_coverage": float(coverage_by_candidate[selected]),
        "candidate_cv_coverage": {
            f"{nominal:.2f}": value
            for nominal, value in coverage_by_candidate.items()
        },
        "all_calibration_rows_used_after_crossfit_selection": True,
        "test_labels_used": False,
    }


def _distributional_shape_adaptive_widths(
    frame,
    calibration_rows,
    signed_normalized_error,
    calibration_scale,
    test_scale,
    *,
    seed,
    coverage=0.90,
):
    """Choose symmetric or asymmetric shape without touching test labels.

    Calibration patients are split into a shape-selection set and a disjoint
    final-calibration set.  The selection set uses grouped cross-validation to
    choose the interval shape whose patient-equalized coverage is closest to
    the requested level.  The chosen shape is then calibrated from scratch on
    untouched final-calibration patients.
    """

    rows = np.asarray(calibration_rows, dtype=np.int64)
    error = np.asarray(signed_normalized_error, dtype=np.float64)
    calibration_scale = np.asarray(calibration_scale, dtype=np.float64)
    test_scale = np.asarray(test_scale, dtype=np.float64)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    fallback_scores = np.abs(error)
    fallback_q, fallback_rank, fallback_metadata = (
        _patient_cluster_conformal_quantile(
            frame, rows, fallback_scores, coverage=coverage
        )
    )
    fallback_width = float(fallback_q) * test_scale
    fallback = (
        fallback_width,
        fallback_width,
        {
            **fallback_metadata,
            "method": "distributional_shape_adaptive_symmetric_fallback",
            "selected_shape": "symmetric",
            "q90_rank": int(fallback_rank),
            "test_labels_used": False,
        },
    )
    if len(unique) < 50 or len(rows) < 120:
        return fallback

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    selection_subjects = set(
        shuffled[: max(24, int(round(len(shuffled) * 0.40)))].tolist()
    )
    selection_mask = np.asarray(
        [subject in selection_subjects for subject in subjects], dtype=bool
    )
    final_mask = ~selection_mask
    if int(selection_mask.sum()) < 60 or int(final_mask.sum()) < 45:
        return fallback

    selection_rows = rows[selection_mask]
    selection_error = error[selection_mask]
    selection_subject_array = subjects[selection_mask]
    folds = np.array_split(np.unique(selection_subject_array), 3)
    candidate_coverage = {"symmetric": [], "asymmetric": []}
    for heldout_subjects in folds:
        heldout = set(heldout_subjects.tolist())
        check = np.asarray(
            [subject in heldout for subject in selection_subject_array],
            dtype=bool,
        )
        fit = ~check
        if int(fit.sum()) < 30 or int(check.sum()) < 10:
            continue
        fit_rows = selection_rows[fit]
        symmetric_q, _, _ = _patient_cluster_conformal_quantile(
            frame,
            fit_rows,
            np.abs(selection_error[fit]),
            coverage=coverage,
        )
        asymmetric_lower, asymmetric_upper, _ = (
            _distributional_asymmetric_widths(
                frame,
                fit_rows,
                -selection_error[fit],
                selection_error[fit],
                np.ones(int(check.sum()), dtype=np.float64),
                coverage=coverage,
            )
        )
        check_error = selection_error[check]
        fold_subjects = selection_subject_array[check]
        symmetric_covered = np.abs(check_error) <= float(symmetric_q)
        asymmetric_covered = (
            (check_error >= -asymmetric_lower)
            & (check_error <= asymmetric_upper)
        )
        for name, covered in (
            ("symmetric", symmetric_covered),
            ("asymmetric", asymmetric_covered),
        ):
            values = (
                pd.DataFrame(
                    {"subject_id": fold_subjects, "covered": covered}
                )
                .groupby("subject_id", sort=False)["covered"]
                .mean()
                .to_numpy(dtype=np.float64)
            )
            candidate_coverage[name].extend(values.tolist())
    usable = {
        name: float(np.mean(values))
        for name, values in candidate_coverage.items()
        if values
    }
    if len(usable) != 2:
        return fallback
    selected = min(
        ("symmetric", "asymmetric"),
        key=lambda name: (
            abs(usable[name] - float(coverage)),
            0 if name == "symmetric" else 1,
        ),
    )
    final_rows = rows[final_mask]
    final_error = error[final_mask]
    if selected == "symmetric":
        quantile, rank, metadata = _patient_cluster_conformal_quantile(
            frame,
            final_rows,
            np.abs(final_error),
            coverage=coverage,
        )
        lower_widths = float(quantile) * test_scale
        upper_widths = lower_widths.copy()
        metadata["q90_rank"] = int(rank)
    else:
        lower_widths, upper_widths, metadata = (
            _distributional_asymmetric_widths(
                frame,
                final_rows,
                -final_error,
                final_error,
                test_scale,
                coverage=coverage,
            )
        )
    return lower_widths, upper_widths, {
        **metadata,
        "method": "distributional_shape_adaptive_patient_split_conformal",
        "selected_shape": selected,
        "selection_cv_patient_coverage": usable,
        "selection_rows": int(selection_mask.sum()),
        "selection_subjects": int(len(selection_subjects)),
        "final_calibration_rows": int(final_mask.sum()),
        "final_calibration_subjects": int(
            len(np.unique(subjects[final_mask]))
        ),
        "selection_final_patient_disjoint": True,
        "test_labels_used": False,
    }


def _crossfit_normalized_conformal(
    frame,
    calibration_rows,
    residuals,
    calibration_features,
    test_features,
    *,
    seed,
    coverage=0.90,
    folds=5,
):
    """Return heteroscedastic widths from patient-grouped cross-conformal."""

    rows = np.asarray(calibration_rows, dtype=np.int64)
    residuals = np.asarray(residuals, dtype=np.float64)
    calibration_features = np.asarray(
        calibration_features, dtype=np.float64
    )
    test_features = np.asarray(test_features, dtype=np.float64)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    if len(residuals) < 90 or len(unique_subjects) < folds * 3:
        quantile, rank = _finite_sample_conformal_quantile(
            residuals, coverage=coverage
        )
        widths = np.full(len(test_features), float(quantile))
        return widths, {
            "method": "fixed_split_conformal_fallback",
            "normalized_q90": None,
            "q90_rank": int(rank),
            "calibration_rows": int(len(residuals)),
            "calibration_subjects": int(len(unique_subjects)),
            "test_labels_used": False,
        }

    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    subject_folds = np.array_split(shuffled, int(folds))
    out_of_fold_scale = np.zeros(len(residuals), dtype=np.float64)
    fold_counts = []
    for heldout_subjects in subject_folds:
        heldout = set(heldout_subjects.tolist())
        check = np.asarray(
            [subject in heldout for subject in subjects], dtype=bool
        )
        fit = ~check
        if int(fit.sum()) < 45 or int(check.sum()) < 10:
            quantile, rank = _finite_sample_conformal_quantile(
                residuals, coverage=coverage
            )
            widths = np.full(len(test_features), float(quantile))
            return widths, {
                "method": "fixed_split_conformal_fallback",
                "normalized_q90": None,
                "q90_rank": int(rank),
                "calibration_rows": int(len(residuals)),
                "calibration_subjects": int(len(unique_subjects)),
                "test_labels_used": False,
            }
        scale_model = _fit_residual_scale_model(
            calibration_features[fit], residuals[fit]
        )
        out_of_fold_scale[check] = _predict_residual_scale(
            scale_model, calibration_features[check]
        )
        fold_counts.append(
            {
                "fit_rows": int(fit.sum()),
                "check_rows": int(check.sum()),
            }
        )
    normalized_scores = residuals / np.maximum(
        out_of_fold_scale, 1.0e-8
    )
    normalized_q90, rank = _finite_sample_conformal_quantile(
        normalized_scores, coverage=coverage
    )
    final_model = _fit_residual_scale_model(
        calibration_features, residuals
    )
    test_scale = _predict_residual_scale(final_model, test_features)
    widths = float(normalized_q90) * test_scale
    return widths, {
        "method": "patient_grouped_crossfit_normalized_conformal",
        "normalized_q90": float(normalized_q90),
        "q90_rank": int(rank),
        "calibration_rows": int(len(residuals)),
        "calibration_subjects": int(len(unique_subjects)),
        "folds": int(folds),
        "fold_counts": fold_counts,
        "median_calibration_scale": float(
            np.median(out_of_fold_scale)
        ),
        "median_test_scale": float(np.median(test_scale)),
        "median_interval_half_width": float(np.median(widths)),
        "test_labels_used": False,
        "patient_grouped_scale_crossfit": True,
    }


def _evaluate_conformal(
    frame,
    arrays,
    calibration_rows,
    test_rows,
    target_names,
    target_indices,
    calibration_prediction,
    test_prediction,
    *,
    seed,
    method="fixed",
    nominal_coverage=0.90,
    calibration_scale=None,
    test_scale=None,
):
    scaler, _, _, _, _, future, future_mask, _, horizon = arrays[:9]
    future_raw = future * scaler.scales + scaler.medians
    output = {}
    for local_index, (target, target_index) in enumerate(
        zip(target_names, target_indices)
    ):
        common_horizons = sorted(
            set(horizon[calibration_rows]).intersection(horizon[test_rows])
        )
        for horizon_value in common_horizons:
            calibration_mask = (
                (horizon[calibration_rows] == float(horizon_value))
                & np.asarray(
                    future_mask[calibration_rows, target_index], dtype=bool
                )
            )
            test_mask = (
                (horizon[test_rows] == float(horizon_value))
                & np.asarray(future_mask[test_rows, target_index], dtype=bool)
            )
            if int(calibration_mask.sum()) < 30 or int(test_mask.sum()) < 30:
                continue
            cal_rows = calibration_rows[calibration_mask]
            heldout_rows = test_rows[test_mask]
            cal_raw = (
                calibration_prediction[calibration_mask, local_index]
                * scaler.scales[target_index]
                + scaler.medians[target_index]
            )
            test_raw = (
                test_prediction[test_mask, local_index]
                * scaler.scales[target_index]
                + scaler.medians[target_index]
            )
            residual = np.abs(
                cal_raw - future_raw[cal_rows, target_index]
            )
            calibration_seed = (
                seed
                + target_index * 10_007
                + int(round(float(horizon_value) * 101))
            )
            if method == "normalized":
                calibration_features = _conformal_scale_features(
                    arrays,
                    cal_rows,
                    target_index,
                    calibration_prediction[
                        calibration_mask, local_index
                    ],
                )
                test_features = _conformal_scale_features(
                    arrays,
                    heldout_rows,
                    target_index,
                    test_prediction[test_mask, local_index],
                )
                widths, calibration = _crossfit_normalized_conformal(
                    frame,
                    cal_rows,
                    residual,
                    calibration_features,
                    test_features,
                    seed=calibration_seed,
                    coverage=nominal_coverage,
                )
            elif method == "adaptive":
                quantile, rank, calibration = (
                    _adaptive_conformal_quantile(
                        frame,
                        cal_rows,
                        residual,
                        seed=calibration_seed,
                        target_coverage=nominal_coverage,
                    )
                )
                widths = np.full(len(heldout_rows), float(quantile))
                calibration["q90_rank"] = int(rank)
            elif method == "patient_cluster":
                quantile, rank, calibration = (
                    _patient_cluster_conformal_quantile(
                        frame,
                        cal_rows,
                        residual,
                        coverage=nominal_coverage,
                    )
                )
                widths = np.full(len(heldout_rows), float(quantile))
                calibration["q90_rank"] = int(rank)
            elif method == "patient_cluster_adaptive":
                quantile, rank, calibration = (
                    _adaptive_patient_cluster_conformal_quantile(
                        frame,
                        cal_rows,
                        residual,
                        seed=calibration_seed,
                        target_coverage=nominal_coverage,
                    )
                )
                widths = np.full(len(heldout_rows), float(quantile))
                calibration["q90_rank"] = int(rank)
            elif method == "patient_cluster_cv_adaptive":
                quantile, rank, calibration = (
                    _crossfit_adaptive_patient_cluster_conformal_quantile(
                        frame,
                        cal_rows,
                        residual,
                        seed=calibration_seed,
                        target_coverage=nominal_coverage,
                    )
                )
                widths = np.full(len(heldout_rows), float(quantile))
                calibration["q90_rank"] = int(rank)
            elif method == "distributional":
                if calibration_scale is None or test_scale is None:
                    raise ValueError(
                        "distributional conformal requires predicted scales"
                    )
                cal_scale_raw = np.maximum(
                    calibration_scale[calibration_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                test_scale_raw = np.maximum(
                    test_scale[test_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                normalized_residual = residual / cal_scale_raw
                quantile, rank, calibration = (
                    _patient_cluster_conformal_quantile(
                        frame,
                        cal_rows,
                        normalized_residual,
                        coverage=nominal_coverage,
                    )
                )
                widths = float(quantile) * test_scale_raw
                calibration.update(
                    {
                        "method": (
                            "target_distributional_scale_patient_cluster_"
                            "conformal"
                        ),
                        "normalized_q90": float(quantile),
                        "q90_rank": int(rank),
                        "median_calibration_predicted_scale": float(
                            np.median(cal_scale_raw)
                        ),
                        "median_test_predicted_scale": float(
                            np.median(test_scale_raw)
                        ),
                        "scale_head_test_labels_used": False,
                    }
                )
            elif method == "distributional_mondrian":
                if calibration_scale is None or test_scale is None:
                    raise ValueError(
                        "distributional_mondrian conformal requires "
                        "predicted scales"
                    )
                cal_scale_raw = np.maximum(
                    calibration_scale[calibration_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                test_scale_raw = np.maximum(
                    test_scale[test_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                normalized_residual = residual / cal_scale_raw
                current = np.asarray(arrays[2], dtype=np.float64)
                current_mask = np.asarray(arrays[3], dtype=bool)
                widths, calibration = _distributional_mondrian_widths(
                    frame,
                    cal_rows,
                    heldout_rows,
                    normalized_residual,
                    cal_scale_raw,
                    test_scale_raw,
                    current[cal_rows, target_index],
                    current[heldout_rows, target_index],
                    current_mask[cal_rows, target_index],
                    current_mask[heldout_rows, target_index],
                    coverage=nominal_coverage,
                )
            elif method == "distributional_adaptive":
                if calibration_scale is None or test_scale is None:
                    raise ValueError(
                        "distributional_adaptive conformal requires "
                        "predicted scales"
                    )
                cal_scale_raw = np.maximum(
                    calibration_scale[calibration_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                test_scale_raw = np.maximum(
                    test_scale[test_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                normalized_residual = residual / cal_scale_raw
                widths, calibration = _distributional_adaptive_widths(
                    frame,
                    cal_rows,
                    normalized_residual,
                    cal_scale_raw,
                    test_scale_raw,
                    seed=calibration_seed,
                    coverage=nominal_coverage,
                )
            elif method == "distributional_asymmetric":
                if calibration_scale is None or test_scale is None:
                    raise ValueError(
                        "distributional_asymmetric conformal requires "
                        "predicted scales"
                    )
                cal_scale_raw = np.maximum(
                    calibration_scale[calibration_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                test_scale_raw = np.maximum(
                    test_scale[test_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                observed_calibration = future_raw[cal_rows, target_index]
                lower_widths, upper_widths, calibration = (
                    _distributional_asymmetric_widths(
                        frame,
                        cal_rows,
                        (cal_raw - observed_calibration) / cal_scale_raw,
                        (observed_calibration - cal_raw) / cal_scale_raw,
                        test_scale_raw,
                        coverage=nominal_coverage,
                    )
                )
                calibration.update(
                    {
                        "median_calibration_predicted_scale": float(
                            np.median(cal_scale_raw)
                        ),
                        "median_test_predicted_scale": float(
                            np.median(test_scale_raw)
                        ),
                        "scale_head_test_labels_used": False,
                    }
                )
            elif method == "distributional_asymmetric_adaptive":
                if calibration_scale is None or test_scale is None:
                    raise ValueError(
                        "distributional_asymmetric_adaptive conformal "
                        "requires predicted scales"
                    )
                cal_scale_raw = np.maximum(
                    calibration_scale[calibration_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                test_scale_raw = np.maximum(
                    test_scale[test_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                signed_error = (
                    future_raw[cal_rows, target_index] - cal_raw
                ) / cal_scale_raw
                lower_widths, upper_widths, calibration = (
                    _distributional_asymmetric_adaptive_widths(
                        frame,
                        cal_rows,
                        signed_error,
                        test_scale_raw,
                        seed=calibration_seed,
                        coverage=nominal_coverage,
                    )
                )
                calibration.update(
                    {
                        "median_calibration_predicted_scale": float(
                            np.median(cal_scale_raw)
                        ),
                        "median_test_predicted_scale": float(
                            np.median(test_scale_raw)
                        ),
                        "scale_head_test_labels_used": False,
                    }
                )
            elif method == "distributional_shape_adaptive":
                if calibration_scale is None or test_scale is None:
                    raise ValueError(
                        "distributional_shape_adaptive conformal requires "
                        "predicted scales"
                    )
                cal_scale_raw = np.maximum(
                    calibration_scale[calibration_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                test_scale_raw = np.maximum(
                    test_scale[test_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                signed_error = (
                    future_raw[cal_rows, target_index] - cal_raw
                ) / cal_scale_raw
                lower_widths, upper_widths, calibration = (
                    _distributional_shape_adaptive_widths(
                        frame,
                        cal_rows,
                        signed_error,
                        cal_scale_raw,
                        test_scale_raw,
                        seed=calibration_seed,
                        coverage=nominal_coverage,
                    )
                )
            elif method == "fixed":
                quantile, rank = _finite_sample_conformal_quantile(
                    residual, coverage=nominal_coverage
                )
                widths = np.full(len(heldout_rows), float(quantile))
                calibration = {
                    "method": "fixed_split_conformal",
                    "selected_nominal_coverage": float(
                        nominal_coverage
                    ),
                    "target_empirical_coverage": 0.90,
                    "calibration_rows": int(len(residual)),
                    "q90_rank": int(rank),
                    "test_labels_used": False,
                }
            else:
                raise ValueError(f"Unknown conformal method: {method}")
            if method not in (
                "distributional_asymmetric",
                "distributional_asymmetric_adaptive",
                "distributional_shape_adaptive",
            ):
                lower_widths = widths
                upper_widths = widths
            observed = future_raw[heldout_rows, target_index]
            covered = (
                (observed >= test_raw - lower_widths)
                & (observed <= test_raw + upper_widths)
            )
            interval_personalization = None
            if (
                method.startswith("distributional")
                and calibration_scale is not None
                and test_scale is not None
            ):
                personalization_calibration_scale = np.maximum(
                    calibration_scale[calibration_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                personalization_test_scale = np.maximum(
                    test_scale[test_mask, local_index]
                    * scaler.scales[target_index],
                    1.0e-6,
                )
                interval_personalization = (
                    _interval_personalization_diagnostics(
                        frame,
                        cal_rows,
                        heldout_rows,
                        target,
                        residual / personalization_calibration_scale,
                        personalization_calibration_scale,
                        personalization_test_scale,
                        test_raw,
                        observed,
                        seed=calibration_seed + 701,
                        coverage=nominal_coverage,
                    )
                )
            coverage = _patient_coverage_diagnostics(
                frame,
                heldout_rows,
                covered,
                seed=calibration_seed + 503,
            )
            key = (
                f"{target}@"
                f"{int(horizon_value) if float(horizon_value).is_integer() else horizon_value}h"
            )
            output[key] = {
                "calibration_n": int(calibration_mask.sum()),
                "test_n": int(test_mask.sum()),
                "q90": float(
                    np.median(
                        np.maximum(lower_widths, upper_widths)
                    )
                ),
                "median_lower_width": float(np.median(lower_widths)),
                "median_upper_width": float(np.median(upper_widths)),
                "q90_rank": int(calibration["q90_rank"]),
                "coverage": coverage["patient_equalized_coverage"],
                "row_coverage": coverage["row_coverage"],
                "patient_equalized_coverage": coverage[
                    "patient_equalized_coverage"
                ],
                "patient_coverage_ci95": coverage[
                    "patient_coverage_ci95"
                ],
                "test_subjects": coverage["test_subjects"],
                "pass": bool(
                    coverage["test_subjects"] >= 20
                    and 0.87
                    <= coverage["patient_equalized_coverage"]
                    <= 0.93
                ),
                "finite_sample_corrected": True,
                "adaptive_calibration": calibration,
                "interval_personalization": interval_personalization,
                "calibration_rows_untouched_during_training": True,
                "test_labels_used_for_calibration": False,
            }
    return output


def _summarize_routes(
    route_weights,
    timescales,
    target_names,
    module_names,
):
    output = {}
    for index, target in enumerate(target_names):
        mean_route = route_weights[:, index].mean(axis=0)
        ranked = np.argsort(-mean_route)
        output[target] = {
            "mean_fast_weight": float(timescales[:, index].mean()),
            "top_cross_organ_routes": [
                {
                    "module": str(module_names[module_index]),
                    "weight": float(mean_route[module_index]),
                }
                for module_index in ranked[:3]
                if mean_route[module_index] > 0.0
            ],
        }
    return output


def _summarize_organ_coupling(
    coupling_weights,
    organ_presence,
    target_coupling,
    target_coupling_weights,
    target_names,
    organ_names,
    coupling_gates=None,
):
    """Report whether canonical organ edges are observable and actually used."""

    if coupling_weights is None or target_coupling is None:
        return {
            "available": False,
            "reason": "variant_has_no_canonical_organ_graph",
        }
    coupling_weights = np.asarray(coupling_weights, dtype=np.float64)
    organ_presence = np.asarray(organ_presence, dtype=np.float64)
    target_coupling = np.asarray(target_coupling, dtype=np.float64)
    target_coupling_weights = np.asarray(
        target_coupling_weights, dtype=np.float64
    )
    adjacency = np.asarray(
        canonical_organ_adjacency(organ_names), dtype=np.float64
    )
    gates = (
        np.asarray(coupling_gates, dtype=np.float64)
        if coupling_gates is not None
        else np.full(len(organ_names), np.nan)
    )
    edges = []
    for left in range(len(organ_names)):
        for right in range(left + 1, len(organ_names)):
            if adjacency[left, right] <= 0.0:
                continue
            supported = (
                (organ_presence[:, left] > 0.0)
                & (organ_presence[:, right] > 0.0)
            )
            support_rows = int(supported.sum())
            edges.append(
                {
                    "left": str(organ_names[left]),
                    "right": str(organ_names[right]),
                    "coobserved_rows": support_rows,
                    "coobserved_fraction": float(supported.mean()),
                    "left_receives_from_right": (
                        float(coupling_weights[supported, left, right].mean())
                        if support_rows
                        else None
                    ),
                    "right_receives_from_left": (
                        float(coupling_weights[supported, right, left].mean())
                        if support_rows
                        else None
                    ),
                    "left_gate": (
                        float(gates[left]) if np.isfinite(gates[left]) else None
                    ),
                    "right_gate": (
                        float(gates[right]) if np.isfinite(gates[right]) else None
                    ),
                }
            )
    target_signal = {}
    for index, target in enumerate(target_names):
        magnitude = np.linalg.norm(target_coupling[:, index], axis=-1)
        target_signal[str(target)] = {
            "mean_l2": float(magnitude.mean()),
            "median_l2": float(np.median(magnitude)),
            "nonzero_fraction": float(np.mean(magnitude > 1.0e-8)),
            "top_neighbor_organs": [
                {
                    "organ": str(organ_names[organ_index]),
                    "mean_weight": float(
                        target_coupling_weights[:, index, organ_index].mean()
                    ),
                }
                for organ_index in np.argsort(
                    -target_coupling_weights[:, index].mean(axis=0)
                )[:3]
                if target_coupling_weights[:, index, organ_index].mean() > 0.0
            ],
        }
    return {
        "available": True,
        "rows": int(len(organ_presence)),
        "edges": edges,
        "target_signal": target_signal,
    }


def _summarize_experts(expert_weights, target_names):
    expert_names = ("shared_slow", "target_fast", "own_organ")
    output = {}
    for index, target in enumerate(target_names):
        weights = expert_weights[:, index]
        output[target] = {
            "mean_weights": {
                name: float(weights[:, expert_index].mean())
                for expert_index, name in enumerate(expert_names)
            },
            "zero_weight_fraction": {
                name: float(np.mean(weights[:, expert_index] <= 1.0e-8))
                for expert_index, name in enumerate(expert_names)
            },
        }
    return output


def _summarize_scales(scales, target_names):
    output = {}
    for index, target in enumerate(target_names):
        values = np.asarray(scales[:, index], dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0.0)]
        if not len(values):
            output[target] = {"rows": 0}
            continue
        mean = float(values.mean())
        output[target] = {
            "rows": int(len(values)),
            "mean_normalized_scale": mean,
            "std_normalized_scale": float(values.std()),
            "coefficient_of_variation": float(
                values.std() / max(mean, 1.0e-8)
            ),
            "q10": float(np.quantile(values, 0.10)),
            "q50": float(np.quantile(values, 0.50)),
            "q90": float(np.quantile(values, 0.90)),
        }
    return output


def _latent_svd_stability(
    frame,
    features,
    fit_rows,
    test_rows,
    *,
    max_rows=4096,
    seed=0,
):
    def patient_equalized_sample(rows, sample_seed):
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) <= int(max_rows):
            return rows
        subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
        unique_subjects = np.unique(subjects)
        rng = np.random.default_rng(sample_seed)
        rng.shuffle(unique_subjects)
        if len(unique_subjects) > int(max_rows):
            unique_subjects = unique_subjects[: int(max_rows)]
        per_subject = max(1, int(max_rows) // len(unique_subjects))
        selected = []
        leftovers = []
        for subject in unique_subjects:
            candidates = rows[subjects == subject].copy()
            rng.shuffle(candidates)
            selected.extend(candidates[:per_subject].tolist())
            leftovers.extend(candidates[per_subject:].tolist())
        remaining = int(max_rows) - len(selected)
        if remaining > 0 and leftovers:
            leftovers = np.asarray(leftovers, dtype=np.int64)
            rng.shuffle(leftovers)
            selected.extend(leftovers[:remaining].tolist())
        return np.asarray(selected[: int(max_rows)], dtype=np.int64)

    def matrix(rows):
        rows = np.asarray(rows, dtype=np.int64)
        context = features["context"][rows].cpu().numpy()
        target = features["target"][rows].cpu().numpy().reshape(len(rows), -1)
        return np.concatenate((context, target), axis=1).astype(np.float32)

    sampled_fit_rows = patient_equalized_sample(fit_rows, seed + 1)
    sampled_test_rows = patient_equalized_sample(test_rows, seed + 2)
    fit = matrix(sampled_fit_rows)
    test = matrix(sampled_test_rows)
    center = fit.mean(axis=0, keepdims=True)
    scale = fit.std(axis=0, keepdims=True)
    scale[scale < 1.0e-6] = 1.0
    fit = (fit - center) / scale
    test = (test - center) / scale

    def decompose(values):
        values = values - values.mean(axis=0, keepdims=True)
        components = min(64, values.shape[0] - 1, values.shape[1])
        _, singular, right = randomized_svd(
            values,
            n_components=components,
            n_iter=5,
            random_state=seed,
        )
        power = np.square(singular)
        probability = power / max(float(power.sum()), 1.0e-12)
        effective_rank = float(
            np.exp(
                -np.sum(
                    probability
                    * np.log(np.clip(probability, 1.0e-12, None))
                )
            )
        )
        return effective_rank, right

    fit_rank, fit_right = decompose(fit)
    test_rank, test_right = decompose(test)
    width = min(5, len(fit_right), len(test_right))
    overlap = np.linalg.svd(
        fit_right[:width] @ test_right[:width].T,
        compute_uv=False,
    )
    mean_overlap = float(np.mean(overlap)) if len(overlap) else None
    ratio = test_rank / max(fit_rank, 1.0e-8)
    return {
        "fit_effective_rank": fit_rank,
        "test_effective_rank": test_rank,
        "rank_ratio": float(ratio),
        "mean_top_subspace_cosine": mean_overlap,
        "fit_sample_rows": int(len(sampled_fit_rows)),
        "test_sample_rows": int(len(sampled_test_rows)),
        "patient_equalized_sampling": True,
        "svd_components": int(min(64, fit.shape[0] - 1, fit.shape[1])),
        "truncated_randomized_svd": True,
        "pass": bool(
            fit_rank >= 2.0
            and test_rank >= 2.0
            and 0.50 <= ratio <= 2.0
            and mean_overlap is not None
            and mean_overlap >= 0.50
        ),
    }


def run(args):
    modules = tuple(
        value.strip() for value in args.modules.split(",") if value.strip()
    )
    horizons = tuple(
        int(value) for value in args.horizons.split(",") if value.strip()
    )
    requested_event_time_targets = tuple(
        value.strip()
        for value in args.targets.split(",")
        if value.strip() and value.strip().lower() != "all"
    )
    frame, variables, module_names = load_joint_event_examples(
        args.event_examples,
        modules,
        horizons,
        args.max_stays,
        min_modules=args.min_modules,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
        history_lags_hours=tuple(
            float(value)
            for value in args.history_lags_hours.split(",")
            if value.strip()
        ),
        event_time_targets=(
            requested_event_time_targets or None
        ),
    )
    target_names = _select_targets(args.targets, variables)
    if not target_names:
        raise ValueError("No requested target is available")
    target_indices = tuple(variables.index(value) for value in target_names)
    target_support = _future_label_support(frame, target_names, horizons)
    expert_anchor_registry = _load_expert_anchor_registry(
        getattr(args, "expert_anchor_registry", None)
    )
    seeds = tuple(
        int(value) for value in args.seeds.split(",") if value.strip()
    )
    variant_configuration = {
        "adjacent_three_level": (False, False, True, True, True, False),
        "hierarchical": (False, False, False, False, False, False),
        "sparse_experts": (False, False, True, False, False, False),
        "sparse_distributional": (False, False, True, True, False, False),
        "sparse_horizon_balanced_distributional": (
            False, False, True, True, False, True
        ),
        "sparse_target_distributional": (
            False, False, True, True, True, False
        ),
        "dynamic_route": (True, False, False, False, False, False),
        "dynamic_route_pcgrad": (True, True, False, False, False, False),
    }
    variant_names = tuple(
        value.strip()
        for value in args.variants.split(",")
        if value.strip()
    )
    unknown_variants = set(variant_names) - set(variant_configuration)
    if unknown_variants:
        raise ValueError(f"Unknown variants: {sorted(unknown_variants)}")
    progress_path = Path(f"{args.output}.progress.json")
    progress_config = {
        "event_examples": str(Path(args.event_examples).resolve()),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "modules": list(module_names),
        "targets": list(target_names),
        "target_mode": (
            "all_available" if args.targets.strip().lower() == "all" else "explicit"
        ),
        "target_future_label_support": target_support,
        "horizons": list(horizons),
        "variants": list(variant_names),
        "base_epochs": int(args.base_epochs),
        "adapter_epochs": int(args.adapter_epochs),
        "scale_epochs": int(args.scale_epochs),
        "expert_entropy_weight": float(args.expert_entropy_weight),
        "distributional_weight": float(args.distributional_weight),
        "conformal_method": str(args.conformal_method),
        "conformal_nominal_coverage": float(
            args.conformal_nominal_coverage
        ),
        "split_mode": str(args.split_mode),
        "train_database": getattr(args, "train_database", None),
        "test_database": getattr(args, "test_database", None),
        "expert_anchor_registry": (
            str(Path(args.expert_anchor_registry).resolve())
            if getattr(args, "expert_anchor_registry", None)
            else None
        ),
        "candidate_coupling_path": str(args.candidate_coupling_path),
        "adapter_balance_columns": [
            value.strip()
            for value in args.adapter_balance_columns.split(",")
            if value.strip()
        ],
    }
    reports = []
    if getattr(args, "resume", False) and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("config") != progress_config:
            raise ValueError(
                "Existing progress report does not match this audit config"
            )
        reports = list(progress.get("reports", []))
        _validate_resume_seed_extension(
            progress.get("requested_seeds", []), seeds, reports
        )
    completed_seeds = {
        int(report["seed"]) for report in reports if "seed" in report
    }
    adapter_balance_columns = tuple(
        value.strip()
        for value in args.adapter_balance_columns.split(",")
        if value.strip()
    )
    missing_balance_columns = sorted(
        set(adapter_balance_columns) - set(frame.columns)
    )
    if missing_balance_columns:
        raise ValueError(
            "Adapter balance columns missing from frame: "
            f"{missing_balance_columns}"
        )
    adapter_balance_labels = None
    if adapter_balance_columns:
        label_parts = [
            column
            + "="
            + frame[column].fillna("__missing__").astype(str)
            for column in adapter_balance_columns
        ]
        labels = label_parts[0]
        for part in label_parts[1:]:
            labels = labels + "|" + part
        adapter_balance_labels = labels.to_numpy(dtype=object)
    for seed in seeds:
        if seed in completed_seeds:
            print(
                json.dumps({"seed": seed, "status": "resumed_skip"}),
                flush=True,
            )
            continue
        train_rows, test_rows = _split_rows_for_mode(
            frame,
            args.split_mode,
            seed,
            train_database=getattr(args, "train_database", None),
            test_database=getattr(args, "test_database", None),
        )
        fit_rows, calibration_rows = _split_train_calibration(
            frame, train_rows, seed + 1001
        )
        model, arrays, losses = fit_joint(
            frame,
            variables,
            module_names,
            fit_rows,
            seed=seed,
            epochs=args.base_epochs,
            batch_size=args.batch_size,
            measurement_process_mode="observed",
            cross_forecast_weight=1.0,
            masked_forecast_probability=0.15,
            module_mask_probability=0.15,
            world_model=True,
            world_model_weight=1.0,
            target_momentum=0.99,
            future_target_mode="window",
            target_min_observations=2,
            regime_balanced_loss=True,
            regime_count=3,
            state_normalization="robust",
            hospital_invariance_weight=args.domain_invariance_weight,
            domain_invariance_columns=tuple(
                value.strip()
                for value in args.domain_invariance_columns.split(",")
                if value.strip()
            ),
            target_encoder_mode="fast",
            target_horizon_regime_adapter=True,
            patient_residual_adapter=True,
            future_task_balanced_sampling=True,
            uncertainty_gate=True,
            validated_edge_adapters=True,
            validated_edge_weight=0.1,
            target_head_stage_epochs=0,
            measurement_time_head=True,
            measurement_time_weight=0.1,
            uncertainty_calibration_weight=0.05,
            include_treatment_context=True,
        )
        teacher_fit, teacher_selection = _split_train_calibration(
            frame, fit_rows, seed + 9001
        )
        all_rows = np.arange(len(frame), dtype=np.int64)
        if expert_anchor_registry is None:
            teacher, teacher_metadata = _validated_router_prediction(
                frame,
                variables,
                module_names,
                arrays,
                teacher_fit,
                teacher_selection,
                all_rows,
                include_treatment_context=True,
                seed=seed + 30_000,
            )
        else:
            teacher, teacher_metadata = _fixed_expert_anchor_prediction(
                frame,
                variables,
                arrays,
                all_rows,
                expert_anchor_registry,
            )
        features = _cache_base_features(
            model,
            arrays,
            target_indices,
            all_rows,
            batch_size=args.batch_size,
        )
        membership = canonical_target_organ_membership(
            target_names,
            module_names,
        )
        variants = {}
        for name in variant_names:
            (
                dynamic_route,
                pcgrad,
                sparse_experts,
                distributional_scale,
                target_specific_projections,
                horizon_balanced_loss,
            ) = variant_configuration[name]
            torch.manual_seed(
                seed
                + {
                    "adjacent_three_level": 97,
                    "hierarchical": 101,
                    "sparse_experts": 151,
                    "sparse_distributional": 151,
                    "sparse_horizon_balanced_distributional": 191,
                    "sparse_target_distributional": 181,
                    "dynamic_route": 202,
                    "dynamic_route_pcgrad": 303,
                }[name]
            )
            hierarchy, training_stages = {}, {}
            if name == "adjacent_three_level":
                # Every matched hierarchy model starts from identical weights and
                # gets the same per-stage epoch budget; a structurally disabled
                # layer's stage is skipped (it has nothing to train).
                system_epochs, organ_epochs, body_epochs = (
                    split_adjacent_training_epochs(args.adapter_epochs)
                )
                stage_settings = {
                    "system": (system_epochs, 40_000, 0.0, args.expert_entropy_weight),
                    "organ": (organ_epochs, 41_000, args.route_entropy_weight, 0.0),
                    "body": (body_epochs, 42_000, args.route_entropy_weight, 0.0),
                }
                for model_name, disabled_paths in HIERARCHY_MODELS.items():
                    torch.manual_seed(seed + 97)
                    model_adapter = AdjacentThreeLevelResidualAdapter(
                        target_indices=target_indices,
                        target_module_membership=membership,
                        latent_dim=model.latent,
                        organ_dim=model.token_dim,
                        module_count=len(module_names),
                        organ_names=module_names,
                        distributional_scale=distributional_scale,
                        disabled_paths=disabled_paths,
                    )
                    for stage in model_adapter.trainable_stages():
                        epochs, stage_seed, route_weight, expert_weight = stage_settings[stage]
                        model_adapter.set_training_stage(stage)
                        training_stages[f"{model_name}:{stage}"] = _train_adapter(
                            model_adapter,
                            features,
                            arrays,
                            teacher,
                            fit_rows,
                            target_indices,
                            epochs=epochs,
                            batch_size=args.batch_size,
                            seed=seed + stage_seed,
                            pcgrad=False,
                            route_entropy_weight=route_weight,
                            expert_entropy_weight=expert_weight,
                            residual_l1_weight=args.residual_l1_weight,
                            distributional_weight=0.0,
                            horizon_balanced_loss=horizon_balanced_loss,
                            balance_labels=adapter_balance_labels,
                        )
                    model_adapter.set_training_stage("all")
                    hierarchy[model_name] = model_adapter
                adapter = hierarchy["full"]
                training = {
                    "loss_history": [
                        loss
                        for stage in adapter.trainable_stages()
                        for loss in training_stages[f"full:{stage}"]["loss_history"]
                    ],
                    "gradient_cosine_before": [],
                    "gradient_cosine_after": [],
                    "matched_models_trained": sorted(hierarchy),
                }
            else:
                adapter = TeacherAnchoredResidualAdapter(
                    target_indices=target_indices,
                    target_module_membership=membership,
                    latent_dim=model.latent,
                    organ_dim=model.token_dim,
                    module_count=len(module_names),
                    dynamic_organ_route=dynamic_route,
                    sparse_experts=sparse_experts,
                    distributional_scale=distributional_scale,
                    target_specific_projections=target_specific_projections,
                )
            if not hierarchy:
                training = _train_adapter(
                    adapter,
                    features,
                    arrays,
                    teacher,
                    fit_rows,
                    target_indices,
                    epochs=args.adapter_epochs,
                    batch_size=args.batch_size,
                    seed=seed + 40_000,
                    pcgrad=pcgrad,
                    route_entropy_weight=args.route_entropy_weight,
                    expert_entropy_weight=args.expert_entropy_weight,
                    residual_l1_weight=args.residual_l1_weight,
                    distributional_weight=(
                        0.0
                        if distributional_scale
                        else args.distributional_weight
                    ),
                    horizon_balanced_loss=horizon_balanced_loss,
                    balance_labels=adapter_balance_labels,
                )
            scale_training = _train_distributional_scale_heads(
                adapter,
                features,
                arrays,
                teacher,
                fit_rows,
                target_indices,
                epochs=args.scale_epochs if distributional_scale else 0,
                batch_size=args.batch_size,
                seed=seed + 45_000,
                horizon_balanced_loss=horizon_balanced_loss,
            )
            prediction = _predict_adapter(
                adapter,
                features,
                arrays,
                teacher,
                test_rows,
                target_indices,
                batch_size=args.batch_size,
            )
            calibration_prediction = _predict_adapter(
                adapter,
                features,
                arrays,
                teacher,
                calibration_rows,
                target_indices,
                batch_size=args.batch_size,
            )
            calibration_no_route = _predict_adapter(
                hierarchy["system_only"] if hierarchy else adapter,
                features,
                arrays,
                teacher,
                calibration_rows,
                target_indices,
                batch_size=args.batch_size,
                **({} if hierarchy else {"disable_organ_route": True}),
            )
            calibration_no_organ_coupling = _predict_adapter(
                hierarchy["body_mediated"] if hierarchy else adapter,
                features,
                arrays,
                teacher,
                calibration_rows,
                target_indices,
                batch_size=args.batch_size,
                **({} if hierarchy else {"disable_organ_coupling": True}),
            )
            calibration_no_body_feedback = _predict_adapter(
                hierarchy["direct_organ"] if hierarchy else adapter,
                features,
                arrays,
                teacher,
                calibration_rows,
                target_indices,
                batch_size=args.batch_size,
                **({} if hierarchy else {"disable_body_feedback": True}),
            )
            no_route = _predict_adapter(
                hierarchy["system_only"] if hierarchy else adapter,
                features,
                arrays,
                teacher,
                test_rows,
                target_indices,
                batch_size=args.batch_size,
                **({} if hierarchy else {"disable_organ_route": True}),
            )
            no_organ_coupling = _predict_adapter(
                hierarchy["body_mediated"] if hierarchy else adapter,
                features,
                arrays,
                teacher,
                test_rows,
                target_indices,
                batch_size=args.batch_size,
                **({} if hierarchy else {"disable_organ_coupling": True}),
            )
            no_body_feedback = _predict_adapter(
                hierarchy["direct_organ"] if hierarchy else adapter,
                features,
                arrays,
                teacher,
                test_rows,
                target_indices,
                batch_size=args.batch_size,
                **({} if hierarchy else {"disable_body_feedback": True}),
            )
            no_expert = _predict_adapter(
                adapter,
                features,
                arrays,
                teacher,
                test_rows,
                target_indices,
                batch_size=args.batch_size,
                disable_sparse_experts=True,
            )
            no_target_projection = _predict_adapter(
                adapter,
                features,
                arrays,
                teacher,
                test_rows,
                target_indices,
                batch_size=args.batch_size,
                disable_target_specific_projections=True,
            )
            candidate_path = (
                args.candidate_coupling_path
                if name == "adjacent_three_level"
                else "full"
            )
            test_path_outputs = {
                "full": prediction,
                "direct_organ": no_body_feedback,
                "body_mediated": no_organ_coupling,
            }
            calibration_path_outputs = {
                "full": calibration_prediction,
                "direct_organ": calibration_no_body_feedback,
                "body_mediated": calibration_no_organ_coupling,
            }
            candidate_prediction = test_path_outputs[candidate_path]
            candidate_calibration = calibration_path_outputs[candidate_path]
            coupling_path_selection = _select_coupling_paths(
                frame,
                arrays,
                calibration_rows,
                target_names,
                target_indices,
                {
                    "system_only": calibration_no_route["value"],
                    "direct_organ": calibration_no_body_feedback["value"],
                    "body_mediated": calibration_no_organ_coupling["value"],
                    "full": calibration_prediction["value"],
                },
                seed=seed + 680_000,
            )
            selected_coupling_prediction = _compose_coupling_path_prediction(
                test_rows,
                arrays,
                target_names,
                {
                    "system_only": no_route["value"],
                    "direct_organ": no_body_feedback["value"],
                    "body_mediated": no_organ_coupling["value"],
                    "full": prediction["value"],
                },
                coupling_path_selection,
            )
            if hierarchy and getattr(args, "export_predictions", None):
                def hierarchy_predictions(rows, known):
                    return {
                        model_name: known[model_name] if model_name in known else _predict_adapter(
                            hierarchy[model_name], features, arrays, teacher, rows, target_indices,
                            batch_size=args.batch_size)["value"]
                        for model_name in HIERARCHY_MODELS
                    }

                pd.concat([
                    _export_predictions(frame, arrays, test_rows, target_names, target_indices, teacher, hierarchy_predictions(
                        test_rows, {"full": prediction["value"], "direct_organ": no_body_feedback["value"],
                                    "body_mediated": no_organ_coupling["value"], "system_only": no_route["value"]}),
                        split="test"),
                    _export_predictions(frame, arrays, calibration_rows, target_names, target_indices, teacher, hierarchy_predictions(
                        calibration_rows, {"full": calibration_prediction["value"],
                                           "direct_organ": calibration_no_body_feedback["value"],
                                           "body_mediated": calibration_no_organ_coupling["value"],
                                           "system_only": calibration_no_route["value"]}),
                        split="calibration"),
                ], ignore_index=True).to_parquet(Path(args.export_predictions).with_suffix(f".seed{seed}.parquet"), index=False)
            before = training["gradient_cosine_before"]
            after = training["gradient_cosine_after"]
            variants[name] = {
                "cells": _evaluate(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    candidate_prediction["value"],
                    teacher,
                    seed=seed,
                ),
                "selected_coupling_cells": _evaluate(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    selected_coupling_prediction,
                    teacher,
                    seed=seed + 681_000,
                ),
                "coupling_path_selection": coupling_path_selection,
                "selected_coupling_ablation": _evaluate_route_ablation(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    selected_coupling_prediction,
                    no_route["value"],
                    seed=seed + 682_000,
                    baseline_label="system_only",
                ),
                "no_route_cells": _evaluate(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    no_route["value"],
                    teacher,
                    seed=seed + 500_000,
                ),
                "route_ablation": _evaluate_route_ablation(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    candidate_prediction["value"],
                    no_route["value"],
                    seed=seed,
                ),
                "organ_coupling_ablation": _evaluate_route_ablation(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    candidate_prediction["value"],
                    no_organ_coupling["value"],
                    seed=seed + 600_000,
                    baseline_label="no_organ_coupling",
                ),
                "body_feedback_ablation": _evaluate_route_ablation(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    prediction["value"],
                    no_body_feedback["value"],
                    seed=seed + 650_000,
                    baseline_label="no_body_feedback",
                ),
                "body_mediated_coupling_ablation": _evaluate_route_ablation(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    no_organ_coupling["value"],
                    no_route["value"],
                    seed=seed + 660_000,
                    baseline_label="system_only",
                ),
                "direct_organ_coupling_ablation": _evaluate_route_ablation(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    no_body_feedback["value"],
                    no_route["value"],
                    seed=seed + 670_000,
                    baseline_label="system_only",
                ),
                "expert_ablation": _evaluate_route_ablation(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    prediction["value"],
                    no_expert["value"],
                    seed=seed + 700_000,
                    baseline_label="no_expert",
                ),
                "target_projection_ablation": _evaluate_route_ablation(
                    frame,
                    arrays,
                    test_rows,
                    target_names,
                    target_indices,
                    prediction["value"],
                    no_target_projection["value"],
                    seed=seed + 710_000,
                    baseline_label="no_target_projection",
                ),
                "conformal": _evaluate_conformal(
                    frame,
                    arrays,
                    calibration_rows,
                    test_rows,
                    target_names,
                    target_indices,
                    candidate_calibration["value"],
                    candidate_prediction["value"],
                    method=args.conformal_method,
                    nominal_coverage=args.conformal_nominal_coverage,
                    seed=seed,
                    calibration_scale=candidate_calibration["scale"],
                    test_scale=candidate_prediction["scale"],
                ),
                "conformal_global_baseline": (
                    _evaluate_conformal(
                        frame,
                        arrays,
                        calibration_rows,
                        test_rows,
                        target_names,
                        target_indices,
                        candidate_calibration["value"],
                        candidate_prediction["value"],
                        method="patient_cluster_cv_adaptive",
                        nominal_coverage=args.conformal_nominal_coverage,
                        seed=seed,
                        calibration_scale=None,
                        test_scale=None,
                    )
                    if str(args.conformal_method).startswith("distributional")
                    else {}
                ),
                "training": {
                    "candidate_coupling_path": candidate_path,
                    "loss_history": training["loss_history"],
                    "stages": {
                        stage: result["loss_history"]
                        for stage, result in training_stages.items()
                    },
                    "gradient_pairs": len(before),
                    "negative_gradient_fraction_before": (
                        float(np.mean(np.asarray(before) < 0.0))
                        if before
                        else None
                    ),
                    "negative_gradient_fraction_after": (
                        float(np.mean(np.asarray(after) < 0.0))
                        if after
                        else None
                    ),
                    "median_gradient_cosine_before": (
                        float(np.median(before)) if before else None
                    ),
                    "median_gradient_cosine_after": (
                        float(np.median(after)) if after else None
                    ),
                    "scale_loss_history": scale_training["loss_history"],
                    "scale_point_predictor_frozen": scale_training[
                        "point_predictor_frozen"
                    ],
                },
                "routing": _summarize_routes(
                    prediction["route_weights"],
                    prediction["timescale_fast_weight"],
                    target_names,
                    module_names,
                ),
                "organ_coupling": _summarize_organ_coupling(
                    prediction["organ_coupling_weights"],
                    prediction["organ_presence"],
                    prediction["target_organ_coupling"],
                    prediction["target_organ_coupling_weights"],
                    target_names,
                    module_names,
                    coupling_gates=(
                        torch.sigmoid(adapter.organ_coupling_gates)
                        .detach()
                        .cpu()
                        .numpy()
                        if hasattr(adapter, "organ_coupling_gates")
                        else None
                    ),
                ),
                "expert_usage": _summarize_experts(
                    prediction["expert_weights"], target_names
                ),
                "predicted_scale": _summarize_scales(
                    prediction["scale"], target_names
                ),
            }
        reports.append(
            {
                "seed": seed,
                "fit_rows": int(len(fit_rows)),
                "calibration_rows_untouched": int(len(calibration_rows)),
                "test_rows": int(len(test_rows)),
                "base_loss": float(losses[-1]),
                "latent_svd": _latent_svd_stability(
                    frame,
                    features,
                    fit_rows,
                    test_rows,
                    seed=seed,
                ),
                "teacher_selection": {
                    key: value
                    for key, value in teacher_metadata.items()
                    if key.split("@", 1)[0] in target_names
                },
                "variants": variants,
            }
        )
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "schema": "teacher_anchored_joint_jepa_progress.v1",
                    "status": "in_progress",
                    "config": progress_config,
                    "requested_seeds": list(seeds),
                    "completed_seeds": [
                        int(report["seed"]) for report in reports
                    ],
                    "reports": reports,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "rows": len(frame),
                    "targets": list(target_names),
                    "variants": list(variants),
                }
            ),
            flush=True,
        )
    aggregate = {}
    for variant in variant_names:
        cells = sorted(
            {
                cell
                for report in reports
                for cell in report["variants"][variant]["cells"]
            }
        )
        aggregate[variant] = {}
        for cell in cells:
            values = [
                report["variants"][variant]["cells"][cell]
                for report in reports
                if cell in report["variants"][variant]["cells"]
            ]
            selected_values = [
                report["variants"][variant]["selected_coupling_cells"][cell]
                for report in reports
                if cell
                in report["variants"][variant].get(
                    "selected_coupling_cells", {}
                )
            ]
            aggregate[variant][cell] = {
                "evaluated_splits": len(values),
                "beats_teacher_splits": int(
                    sum(
                        value["teacher_bootstrap"]["pass"]
                        for value in values
                    )
                ),
                "beats_persistence_splits": int(
                    sum(
                        value["persistence_bootstrap"]["pass"]
                        for value in values
                    )
                ),
                "selected_path_beats_teacher_splits": int(
                    sum(
                        value["teacher_bootstrap"]["pass"]
                        for value in selected_values
                    )
                ),
                "selected_path_beats_persistence_splits": int(
                    sum(
                        value["persistence_bootstrap"]["pass"]
                        for value in selected_values
                    )
                ),
                "selected_path_beats_system_only_splits": int(
                    sum(
                        report["variants"][variant]
                        .get("selected_coupling_ablation", {})
                        .get(cell, {})
                        .get("system_only_bootstrap", {})
                        .get("pass", False)
                        for report in reports
                    )
                ),
                "selected_paths": [
                    report["variants"][variant]
                    .get("coupling_path_selection", {})
                    .get(cell, {})
                    .get("selected_path", "system_only")
                    for report in reports
                ],
                "median_delta_vs_teacher": float(
                    np.median(
                        [value["delta_vs_teacher"] for value in values]
                    )
                ),
                "median_delta_vs_persistence": float(
                    np.median(
                        [value["delta_vs_persistence"] for value in values]
                    )
                ),
                "selected_path_median_delta_vs_teacher": float(
                    np.median(
                        [value["delta_vs_teacher"] for value in selected_values]
                    )
                ),
                "selected_path_median_delta_vs_persistence": float(
                    np.median(
                        [
                            value["delta_vs_persistence"]
                            for value in selected_values
                        ]
                    )
                ),
                "selected_path_median_delta_vs_system_only": float(
                    np.median(
                        [
                            report["variants"][variant][
                                "selected_coupling_ablation"
                            ][cell]["delta_vs_system_only"]
                            for report in reports
                            if cell
                            in report["variants"][variant].get(
                                "selected_coupling_ablation", {}
                            )
                        ]
                    )
                ),
                "beats_no_route_splits": int(
                    sum(
                        report["variants"][variant]["route_ablation"]
                        .get(cell, {})
                        .get("no_route_bootstrap", {})
                        .get("pass", False)
                        for report in reports
                    )
                ),
                "beats_no_organ_coupling_splits": int(
                    sum(
                        report["variants"][variant]
                        .get("organ_coupling_ablation", {})
                        .get(cell, {})
                        .get("no_organ_coupling_bootstrap", {})
                        .get("pass", False)
                        for report in reports
                    )
                ),
                "beats_no_body_feedback_splits": int(
                    sum(
                        report["variants"][variant]
                        .get("body_feedback_ablation", {})
                        .get(cell, {})
                        .get("no_body_feedback_bootstrap", {})
                        .get("pass", False)
                        for report in reports
                    )
                ),
                "body_mediated_beats_system_only_splits": int(
                    sum(
                        report["variants"][variant]
                        .get("body_mediated_coupling_ablation", {})
                        .get(cell, {})
                        .get("system_only_bootstrap", {})
                        .get("pass", False)
                        for report in reports
                    )
                ),
                "direct_organ_beats_system_only_splits": int(
                    sum(
                        report["variants"][variant]
                        .get("direct_organ_coupling_ablation", {})
                        .get(cell, {})
                        .get("system_only_bootstrap", {})
                        .get("pass", False)
                        for report in reports
                    )
                ),
                "median_delta_vs_no_organ_coupling": float(
                    np.median(
                        [
                            report["variants"][variant][
                                "organ_coupling_ablation"
                            ][cell]["delta_vs_no_organ_coupling"]
                            for report in reports
                            if cell
                            in report["variants"][variant].get(
                                "organ_coupling_ablation", {}
                            )
                        ]
                    )
                ),
                "median_delta_vs_no_body_feedback": float(
                    np.median(
                        [
                            report["variants"][variant][
                                "body_feedback_ablation"
                            ][cell]["delta_vs_no_body_feedback"]
                            for report in reports
                            if cell
                            in report["variants"][variant].get(
                                "body_feedback_ablation", {}
                            )
                        ]
                    )
                ),
                "median_body_mediated_delta_vs_system_only": float(
                    np.median(
                        [
                            report["variants"][variant][
                                "body_mediated_coupling_ablation"
                            ][cell]["delta_vs_system_only"]
                            for report in reports
                            if cell
                            in report["variants"][variant].get(
                                "body_mediated_coupling_ablation", {}
                            )
                        ]
                    )
                ),
                "median_direct_organ_delta_vs_system_only": float(
                    np.median(
                        [
                            report["variants"][variant][
                                "direct_organ_coupling_ablation"
                            ][cell]["delta_vs_system_only"]
                            for report in reports
                            if cell
                            in report["variants"][variant].get(
                                "direct_organ_coupling_ablation", {}
                            )
                        ]
                    )
                ),
                "median_delta_vs_no_route": float(
                    np.median(
                        [
                            report["variants"][variant]["route_ablation"][cell][
                                "delta_vs_no_route"
                            ]
                            for report in reports
                            if cell
                            in report["variants"][variant]["route_ablation"]
                        ]
                    )
                ),
                "beats_no_expert_splits": int(
                    sum(
                        report["variants"][variant]["expert_ablation"]
                        .get(cell, {})
                        .get("no_expert_bootstrap", {})
                        .get("pass", False)
                        for report in reports
                    )
                ),
                "median_delta_vs_no_expert": float(
                    np.median(
                        [
                            report["variants"][variant]["expert_ablation"][
                                cell
                            ]["delta_vs_no_expert"]
                            for report in reports
                            if cell
                            in report["variants"][variant]["expert_ablation"]
                        ]
                    )
                ),
                "conformal_pass_splits": int(
                    sum(
                        report["variants"][variant]
                        .get("conformal", {})
                        .get(cell, {})
                        .get("pass", False)
                        for report in reports
                    )
                ),
                "beats_no_target_projection_splits": int(
                    sum(
                        report["variants"][variant]
                        .get("target_projection_ablation", {})
                        .get(cell, {})
                        .get("no_target_projection_bootstrap", {})
                        .get("pass", False)
                        for report in reports
                    )
                ),
                "median_delta_vs_no_target_projection": float(
                    np.median(
                        [
                            report["variants"][variant][
                                "target_projection_ablation"
                            ][cell]["delta_vs_no_target_projection"]
                            for report in reports
                            if cell
                            in report["variants"][variant].get(
                                "target_projection_ablation", {}
                            )
                        ]
                    )
                ),
            }
    return {
        "schema": "teacher_anchored_joint_jepa_audit.v5",
        "status": "bounded_candidate_only",
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "hospitals": int(frame["hospitalid"].nunique()),
        "modules": list(module_names),
        "targets": list(target_names),
        "target_mode": (
            "all_available"
            if args.targets.strip().lower() == "all"
            else "explicit"
        ),
        "target_future_label_support": target_support,
        "horizons": list(horizons),
        "seeds": list(seeds),
        "split_mode": str(args.split_mode),
        "architecture": {
            "strict_adjacent_three_level_candidate": True,
            "three_level_path": "system <-> organ <-> body",
            "organ_layer_graph": "canonical_sparse_organ_to_organ",
            "direct_system_body_path": False,
            "validated_expert_projection_layer": "system",
            "adjacent_training_schedule": (
                "train_system_then_organ_then_body"
            ),
            "matched_system_only_ablation_required": True,
            "path_specific_matched_ablation_required": True,
            "candidate_coupling_path": str(args.candidate_coupling_path),
            "direct_organ_must_beat_system_only": True,
            "body_mediated_must_beat_system_only": True,
            "validated_router_teacher_anchor": True,
            "fixed_external_expert_anchor_registry": bool(
                expert_anchor_registry is not None
            ),
            "zero_initialized_bounded_residual": True,
            "three_level_latent_hierarchy": True,
            "level_1_validated_system_experts": True,
            "level_2_canonical_organ_latent_tokens": True,
            "level_3_shared_whole_body_latent": True,
            "target_heads_anchored_at_system_layer": True,
            "shared_slow_context_latent": True,
            "target_specific_fast_future_latent": True,
            "target_horizon_sparse_experts": True,
            "optional_target_specific_projection_residuals": True,
            "target_specific_distributional_scale_heads": bool(
                variant_configuration["adjacent_three_level"][3]
            ),
            "scale_loss_detached_from_physiology_context": True,
            "two_stage_point_then_scale_training": True,
            "matched_no_expert_ablation_required": True,
            "target_specific_sparse_canonical_organ_routing": True,
            "target_own_organ_excluded_from_route": True,
            "patient_bootstrap_route_ablation_required": True,
            "pcgrad_variant": False,
            "calibration_rows_used_for_training": False,
            "conformal_method": args.conformal_method,
            "conformal_nominal_coverage": float(
                args.conformal_nominal_coverage
            ),
            "domain_invariance_weight": float(
                args.domain_invariance_weight
            ),
            "domain_invariance_columns": [
                value.strip()
                for value in args.domain_invariance_columns.split(",")
                if value.strip()
            ],
            "adapter_balance_columns": list(adapter_balance_columns),
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "reports": reports,
        "aggregate": aggregate,
        "promotion_status": "candidate_only",
        "next_gate": (
            "Run all seven patient seeds and hospital/care-unit/time/conformal "
            "only if a bounded variant repeatedly beats the teacher router."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-examples", type=Path, required=True)
    parser.add_argument("--modules", required=True)
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help=(
            "Comma-separated targets, or 'all' to train target-specific "
            "experts for every variable in the canonical event schema."
        ),
    )
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--history-lags-hours", default="48,24,12,8,6,4,3,2,1,0")
    parser.add_argument("--max-stays", type=int, default=200)
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument("--seeds", default="7,19,31")
    parser.add_argument(
        "--split-mode",
        choices=("subject", "hospital", "careunit", "time", "database"),
        default="subject",
    )
    parser.add_argument("--train-database", default=None)
    parser.add_argument("--test-database", default=None)
    parser.add_argument(
        "--expert-anchor-registry",
        type=Path,
        default=None,
        help="Frozen target-horizon expert columns used as system anchors.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=(
            "Canonical training defaults to adjacent_three_level. Other "
            "variants must be named explicitly and are ablation-only."
        ),
    )
    parser.add_argument("--base-epochs", type=int, default=1)
    parser.add_argument("--adapter-epochs", type=int, default=3)
    parser.add_argument("--scale-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--route-entropy-weight", type=float, default=0.001)
    parser.add_argument("--expert-entropy-weight", type=float, default=0.0005)
    parser.add_argument("--distributional-weight", type=float, default=0.25)
    parser.add_argument("--residual-l1-weight", type=float, default=0.01)
    parser.add_argument(
        "--candidate-coupling-path",
        choices=("full", "direct_organ", "body_mediated"),
        default="full",
        help=(
            "Evaluate the deployable adjacent path as the primary candidate. "
            "All path-specific matched ablations remain required."
        ),
    )
    parser.add_argument(
        "--domain-invariance-weight",
        type=float,
        default=0.05,
        help="Training-only adversarial provenance-domain loss weight.",
    )
    parser.add_argument(
        "--domain-invariance-columns",
        default="hospitalid",
        help=(
            "Comma-separated provenance columns removed adversarially from "
            "the physiology latent, for example hospitalid,careunit."
        ),
    )
    parser.add_argument(
        "--adapter-balance-columns",
        default="",
        help=(
            "Comma-separated provenance columns sampled with equal group "
            "probability during residual-adapter training."
        ),
    )
    parser.add_argument(
        "--conformal-method",
        choices=(
            "fixed",
            "adaptive",
            "normalized",
            "patient_cluster",
            "patient_cluster_adaptive",
            "patient_cluster_cv_adaptive",
            "distributional",
            "distributional_mondrian",
            "distributional_adaptive",
            "distributional_asymmetric",
            "distributional_asymmetric_adaptive",
            "distributional_shape_adaptive",
        ),
        default="distributional_adaptive",
    )
    parser.add_argument(
        "--conformal-nominal-coverage", type=float, default=0.90
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed seeds from OUTPUT.progress.json.",
    )
    parser.add_argument(
        "--export-predictions",
        type=Path,
        default=None,
        help="Write labelled test-row anchor and selected-path predictions (one parquet per seed).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": result["status"],
                "rows": result["rows"],
                "seeds": len(result["seeds"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
