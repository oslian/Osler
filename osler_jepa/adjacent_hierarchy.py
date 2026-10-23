"""Strict adjacent-layer residual adapter for whole-body factual forecasts.

The adapter preserves validated target experts as the system-layer anchor and
permits information flow only through the following graph:

    system <-> organ <-> body

There is deliberately no system-to-body or body-to-system projection.  Body
context can alter a target forecast only after passing through the target's
owning organ token.  This keeps the hierarchy auditable while allowing the
existing factual router to remain the safe prediction anchor.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from osler_jepa.organ_schema import canonical_organ_adjacency
from osler_jepa.patient_state import PatientStateAdapter


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


def split_adjacent_training_epochs(total_epochs: int) -> tuple[int, int, int]:
    """Split one fixed budget across system, organ, and body stages."""

    total_epochs = int(total_epochs)
    if total_epochs < 3:
        raise ValueError(
            "adjacent three-level training requires at least three epochs"
        )
    system_epochs = max(1, total_epochs // 3)
    organ_epochs = max(1, total_epochs // 3)
    return system_epochs, organ_epochs, total_epochs - system_epochs - organ_epochs


DISABLEABLE_PATHS = ("organ_route", "organ_coupling", "body_feedback")
# Matched, independently trained hierarchy models: name -> disabled paths.
# Each enabled component must beat the model that differs only by removing it:
# organ coupling (full vs body_mediated, direct_organ vs organ_route), body
# feedback (full vs direct_organ, body_mediated vs organ_route) and the organ
# route itself (organ_route vs system_only).
HIERARCHY_MODELS = {
    "full": (),
    "direct_organ": ("body_feedback",),
    "body_mediated": ("organ_coupling",),
    "organ_route": ("organ_coupling", "body_feedback"),
    "system_only": ("organ_route", "organ_coupling", "body_feedback"),
}
MATCHED_ABLATIONS = {
    "full": ("body_mediated", "direct_organ"),
    "direct_organ": ("organ_route",),
    "body_mediated": ("organ_route",),
    "organ_route": ("system_only",),
    "system_only": (),
}


class AdjacentThreeLevelResidualAdapter(nn.Module):
    """Refine expert cells through system-organ-body adjacent message passing."""

    architecture_contract = {
        "layers": ("system", "organ", "body"),
        "allowed_directed_edges": (
            "system_to_organ",
            "organ_to_organ",
            "organ_to_system",
            "organ_to_body",
            "body_to_organ",
        ),
        "forbidden_directed_edges": (
            "system_to_body",
            "body_to_system",
        ),
        "expert_projection_layer": "system",
        "prediction_anchor": "validated_router_teacher",
    }

    def __init__(
        self,
        *,
        target_indices: Iterable[int],
        target_module_membership: np.ndarray,
        latent_dim: int,
        organ_dim: int,
        module_count: int,
        organ_names: Iterable[str],
        adapter_dim: int = 48,
        distributional_scale: bool = False,
        patient_history_dim: int = 0,
        patient_belief_dim: int = 0,
        disabled_paths: Iterable[str] = (),
    ) -> None:
        super().__init__()
        self.disabled_paths = frozenset(str(path) for path in disabled_paths)
        unknown = self.disabled_paths - set(DISABLEABLE_PATHS)
        if unknown:
            raise ValueError(f"unknown disabled paths: {sorted(unknown)}")
        self.target_indices = tuple(int(value) for value in target_indices)
        self.target_count = len(self.target_indices)
        self.module_count = int(module_count)
        self.organ_names = tuple(str(value) for value in organ_names)
        if len(self.organ_names) != self.module_count:
            raise ValueError("organ_names must match module_count")
        self.distributional_scale = bool(distributional_scale)
        self.patient_state_adapter = (
            PatientStateAdapter(
                organ_dim=organ_dim,
                history_input_dim=patient_history_dim,
                belief_dim=patient_belief_dim,
                hidden_dim=adapter_dim,
            )
            if patient_history_dim > 0
            else None
        )
        self.dynamic_organ_route = True
        self.sparse_experts = True
        self.target_specific_projections = True

        membership = np.asarray(target_module_membership, dtype=np.float32)
        expected = (self.target_count, self.module_count)
        if membership.shape != expected:
            raise ValueError(
                f"target_module_membership must have shape {expected}"
            )
        membership = np.clip(membership, 0.0, 1.0)
        membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1.0)
        self.register_buffer(
            "system_organ_membership",
            torch.as_tensor(membership),
            persistent=True,
        )

        # System layer: current target state and the validated expert forecast
        # enter here.  Target JEPA features are also system-local evidence.
        self.system_state_projection = nn.Sequential(
            nn.Linear(9, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.system_jepa_projection = nn.Sequential(
            nn.Linear(latent_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.system_embedding = nn.Parameter(
            torch.randn(self.target_count, adapter_dim) * 0.02
        )

        # Adjacent upward path: system -> organ -> body.
        self.system_to_organ = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.observed_organ_projection = nn.Sequential(
            nn.Linear(organ_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.register_buffer(
            "organ_coupling_adjacency",
            torch.as_tensor(
                canonical_organ_adjacency(self.organ_names),
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.organ_query = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.organ_key = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.organ_value = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.organ_message = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.organ_coupling_gates = nn.Parameter(
            torch.zeros(self.module_count)
        )
        # A target may need a different neighboring-organ signal than another
        # target owned by the same organ (for example MAP versus heart rate).
        # Queries remain restricted to the owning organ's canonical neighbors.
        self.target_organ_query = nn.Linear(
            adapter_dim, adapter_dim, bias=False
        )
        self.target_organ_key = nn.Linear(
            adapter_dim, adapter_dim, bias=False
        )
        self.target_organ_value = nn.Linear(
            adapter_dim, adapter_dim, bias=False
        )
        self.target_organ_message = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.target_organ_coupling_gates = nn.Parameter(
            torch.zeros(self.target_count)
        )
        self.organ_to_body = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.observed_body_projection = nn.Sequential(
            nn.Linear(latent_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
        )
        self.body_normalization = nn.LayerNorm(adapter_dim)

        # Adjacent downward path: body -> organ -> system.  There is no direct
        # body-to-system module by construction.
        self.body_to_organ = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * adapter_dim, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.SiLU(),
                    nn.Linear(adapter_dim, adapter_dim),
                )
                for _ in range(self.module_count)
            ]
        )
        self.organ_to_system = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * adapter_dim, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.SiLU(),
                    nn.Linear(adapter_dim, adapter_dim),
                )
                for _ in range(self.target_count)
            ]
        )
        self.organ_feedback_gates = nn.Parameter(
            torch.zeros(self.module_count)
        )
        self.system_feedback_gates = nn.Parameter(
            torch.zeros(self.target_count)
        )
        self.body_system_feedback_gates = nn.Parameter(
            torch.zeros(self.target_count)
        )

        residual_width = adapter_dim + 1 + 4
        self.system_residual_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(residual_width, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.SiLU(),
                    nn.Linear(adapter_dim, 1),
                )
                for _ in self.target_indices
            ]
        )
        for head in self.system_residual_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.system_residual_gates = nn.Parameter(
            torch.full((self.target_count,), -2.0)
        )
        self.organ_residual_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(residual_width, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.SiLU(),
                    nn.Linear(adapter_dim, 1),
                )
                for _ in self.target_indices
            ]
        )
        for head in self.organ_residual_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.organ_residual_gates = nn.Parameter(
            torch.zeros(self.target_count)
        )
        self.body_residual_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(residual_width, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.SiLU(),
                    nn.Linear(adapter_dim, 1),
                )
                for _ in self.target_indices
            ]
        )
        for head in self.body_residual_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.body_residual_gates = nn.Parameter(
            torch.zeros(self.target_count)
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
                    for _ in self.target_indices
                ]
            )
            initial_scale = 0.50
            initial_raw = math.log(math.expm1(initial_scale - 0.03))
            for head in self.scale_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.constant_(head[-1].bias, initial_raw)
        self.training_stage = "all"

    def trainable_stages(self) -> tuple[str, ...]:
        """Stages with an enabled path: a structurally disabled layer has nothing to train."""

        stages = ["system"]
        if "organ_route" not in self.disabled_paths:
            stages.append("organ")
            if "body_feedback" not in self.disabled_paths:
                stages.append("body")
        return tuple(stages)

    def set_training_stage(self, stage: str) -> None:
        """Select a non-competing system, organ, or body stage."""

        if stage not in {"system", "organ", "body", "all"}:
            raise ValueError("stage must be system, organ, body, or all")
        self.training_stage = stage
        for parameter in self.parameters():
            parameter.requires_grad_(stage == "all")
        if stage == "all":
            return
        if stage == "system":
            prefixes = (
                "system_state_projection",
                "system_jepa_projection",
                "system_embedding",
                "system_residual_heads",
                "system_residual_gates",
            )
        elif stage == "organ":
            prefixes = (
                "system_to_organ",
                "observed_organ_projection",
                "organ_query",
                "organ_key",
                "organ_value",
                "organ_message",
                "organ_coupling_gates",
                "target_organ_query",
                "target_organ_key",
                "target_organ_value",
                "target_organ_message",
                "target_organ_coupling_gates",
                "organ_to_system",
                "system_feedback_gates",
                "organ_residual_heads",
                "organ_residual_gates",
                "patient_state_adapter",
            )
        else:
            prefixes = (
                "organ_to_body",
                "observed_body_projection",
                "body_normalization",
                "body_to_organ",
                "organ_feedback_gates",
                "body_system_feedback_gates",
                "body_residual_heads",
                "body_residual_gates",
            )
        for name, parameter in self.named_parameters():
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                parameter.requires_grad_(True)

    def _couple_organs(
        self,
        organ_tokens: torch.Tensor,
        organ_presence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange messages only along canonical same-layer organ edges."""

        query = self.organ_query(organ_tokens)
        key = self.organ_key(organ_tokens)
        value = self.organ_value(organ_tokens)
        score = torch.einsum("bmd,bnd->bmn", query, key) / math.sqrt(
            query.shape[-1]
        )
        present = organ_presence > 0.0
        allowed = self.organ_coupling_adjacency.bool().unsqueeze(0)
        allowed = allowed & present.unsqueeze(1) & present.unsqueeze(2)
        score = score.masked_fill(~allowed, -1.0e4)
        weight = torch.softmax(score, dim=-1) * allowed.to(score.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        message = torch.einsum("bmn,bnd->bmd", weight, value)
        message = self.organ_message(message)
        message = message * present.unsqueeze(-1).to(message.dtype)
        gate = torch.sigmoid(self.organ_coupling_gates).view(1, -1, 1)
        return organ_tokens + gate * torch.tanh(message), weight

    def _target_organ_messages(
        self,
        system_tokens: torch.Tensor,
        organ_tokens: torch.Tensor,
        organ_presence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select target-specific messages from canonical neighboring organs."""

        query = self.target_organ_query(system_tokens)
        key = self.target_organ_key(organ_tokens)
        value = self.target_organ_value(organ_tokens)
        score = torch.einsum("btd,bmd->btm", query, key) / math.sqrt(
            query.shape[-1]
        )
        target_neighbors = torch.einsum(
            "tm,mn->tn",
            self.system_organ_membership,
            self.organ_coupling_adjacency,
        ) > 0.0
        present = organ_presence > 0.0
        owner_present = torch.einsum(
            "tm,bm->bt", self.system_organ_membership, present.to(query.dtype)
        ) > 0.0
        allowed = (
            target_neighbors.unsqueeze(0)
            & present.unsqueeze(1)
            & owner_present.unsqueeze(-1)
        )
        score = score.masked_fill(~allowed, -1.0e4)
        weight = torch.softmax(score, dim=-1) * allowed.to(score.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        message = torch.einsum("btm,bmd->btd", weight, value)
        message = self.target_organ_message(message)
        has_neighbor = allowed.any(dim=-1, keepdim=True).to(message.dtype)
        gate = torch.sigmoid(self.target_organ_coupling_gates).view(
            1, -1, 1
        )
        return gate * torch.tanh(message) * has_neighbor, weight

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
        patient_history: torch.Tensor | None = None,
        patient_history_mask: torch.Tensor | None = None,
        patient_belief: torch.Tensor | None = None,
        disable_organ_route: bool = False,
        disable_organ_coupling: bool = False,
        disable_body_feedback: bool = False,
        disable_sparse_experts: bool = False,
        disable_target_specific_projections: bool = False,
    ) -> dict[str, torch.Tensor]:
        # Structural ablations (``disabled_paths``) hold during training and
        # prediction alike, so a matched ablation is an independently trained
        # model rather than a switch on the full model.
        disable_organ_route = disable_organ_route or "organ_route" in self.disabled_paths
        disable_organ_coupling = disable_organ_coupling or "organ_coupling" in self.disabled_paths
        disable_body_feedback = disable_body_feedback or "body_feedback" in self.disabled_paths
        time = _time_features(horizon)
        repeated_time = time.unsqueeze(1).expand(-1, self.target_count, -1)
        expert_available = torch.isfinite(teacher_anchor).to(current.dtype)
        # Zero is the training-normalized population median, not a physical
        # zero.  Availability remains explicit so a missing expert forecast
        # can never masquerade as a measured value.
        safe_teacher = torch.where(
            expert_available > 0.0,
            teacher_anchor,
            torch.zeros_like(teacher_anchor),
        )
        expert_feature = (
            torch.zeros_like(safe_teacher)
            if disable_sparse_experts
            else safe_teacher
        )
        system_state = torch.stack(
            (
                expert_feature,
                expert_available,
                current,
                current_mask,
                ages,
            ),
            dim=-1,
        )
        system_tokens = self.system_state_projection(
            torch.cat((system_state, repeated_time), dim=-1)
        )
        system_tokens = system_tokens + self.system_embedding.unsqueeze(0)
        if not disable_target_specific_projections:
            system_tokens = system_tokens + self.system_jepa_projection(
                target_latent
            )

        membership = self.system_organ_membership
        observed_system = current_mask.unsqueeze(-1)
        weighted_membership = (
            membership.unsqueeze(0) * current_mask.unsqueeze(-1)
        )
        organ_denominator = weighted_membership.sum(dim=1).unsqueeze(-1).clamp_min(1.0)
        organ_from_system = torch.einsum(
            "btd,btm->bmd", system_tokens * observed_system, weighted_membership
        ) / organ_denominator
        patient_state_output = None
        if self.patient_state_adapter is not None:
            if patient_history is None or patient_history_mask is None:
                raise ValueError(
                    "configured patient-state adapter requires causal patient history"
                )
            patient_state_output = self.patient_state_adapter(
                organ_latents=organ_latents,
                organ_presence=organ_presence,
                patient_history=patient_history,
                patient_history_mask=patient_history_mask,
                patient_belief=patient_belief,
            )
            organ_latents = patient_state_output["organ_latents"]
        organ_tokens = self.observed_organ_projection(organ_latents)
        organ_tokens = organ_tokens + self.system_to_organ(organ_from_system)
        system_supported_organs = (
            torch.einsum(
                "tm,bt->bm",
                membership,
                current_mask,
            )
            > 0.0
        ).to(organ_presence.dtype)
        effective_organ_presence = torch.maximum(
            organ_presence,
            system_supported_organs,
        )
        uncoupled_organ_tokens = organ_tokens
        if disable_organ_coupling:
            organ_coupling_weights = organ_tokens.new_zeros(
                (
                    len(organ_tokens),
                    self.module_count,
                    self.module_count,
                )
            )
            target_organ_coupling = organ_tokens.new_zeros(
                (len(organ_tokens), self.target_count, organ_tokens.shape[-1])
            )
            target_organ_coupling_weights = organ_tokens.new_zeros(
                (len(organ_tokens), self.target_count, self.module_count)
            )
        else:
            (
                target_organ_coupling,
                target_organ_coupling_weights,
            ) = self._target_organ_messages(
                system_tokens,
                uncoupled_organ_tokens,
                effective_organ_presence,
            )
            organ_tokens, organ_coupling_weights = self._couple_organs(
                organ_tokens, effective_organ_presence
            )

        organ_weights = effective_organ_presence.unsqueeze(-1)
        organ_pooled = (organ_tokens * organ_weights).sum(dim=1) / organ_weights.sum(
            dim=1
        ).clamp_min(1.0)
        body_token = self.body_normalization(
            self.organ_to_body(organ_pooled)
            + self.observed_body_projection(context_latent)
        )

        if self.training_stage == "system":
            disable_organ_route = True
        elif self.training_stage == "organ":
            disable_body_feedback = True
        route_weights = (
            membership.unsqueeze(0)
            * effective_organ_presence.unsqueeze(1)
        )
        route_weights = route_weights / route_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        if disable_organ_route:
            route_weights = torch.zeros_like(route_weights)
            organ_system = system_tokens
            final_system = system_tokens
        else:
            # Everything the organ layer tells a target (its routed organs and
            # the target-specific organ coupling) reaches the system layer only
            # through the organ -> system interface, so that interface is
            # trained by the organ-stage point loss.
            target_organ_context = (
                torch.einsum("btm,bmd->btd", route_weights, organ_tokens)
                + target_organ_coupling
            )
            organ_system = torch.stack(
                [
                    system_tokens[:, index]
                    + torch.sigmoid(self.system_feedback_gates[index])
                    * torch.tanh(
                        adapter(
                            torch.cat(
                                (
                                    system_tokens[:, index],
                                    target_organ_context[:, index],
                                ),
                                dim=-1,
                            )
                        )
                    )
                    for index, adapter in enumerate(self.organ_to_system)
                ],
                dim=1,
            )
            if disable_body_feedback:
                final_system = organ_system
            else:
                organ_feedback = torch.stack(
                    [
                        organ_tokens[:, index]
                        + torch.sigmoid(self.organ_feedback_gates[index])
                        * torch.tanh(
                            adapter(
                                torch.cat(
                                    (organ_tokens[:, index], body_token), dim=-1
                                )
                            )
                        )
                        for index, adapter in enumerate(self.body_to_organ)
                    ],
                    dim=1,
                )
                target_body_organ = torch.einsum(
                    "btm,bmd->btd", route_weights, organ_feedback
                )
                final_system = torch.stack(
                    [
                        organ_system[:, index]
                        + torch.sigmoid(
                            self.body_system_feedback_gates[index]
                        )
                        * torch.tanh(
                            adapter(
                                torch.cat(
                                    (
                                        organ_system[:, index],
                                        target_body_organ[:, index],
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                        for index, adapter in enumerate(self.organ_to_system)
                    ],
                    dim=1,
                )

        predictions, residuals, scales = [], [], []
        system_residuals, organ_residuals, body_residuals = [], [], []
        for index, system_head in enumerate(self.system_residual_heads):
            system_input = torch.cat(
                (
                    system_tokens[:, index],
                    current[:, index : index + 1],
                    time,
                ),
                dim=-1,
            )
            system_residual = (
                0.75
                * torch.sigmoid(self.system_residual_gates[index])
                * torch.tanh(system_head(system_input))
            )
            organ_residual = torch.zeros_like(system_residual)
            body_residual = torch.zeros_like(system_residual)
            if not disable_organ_route:
                organ_input = torch.cat(
                    (
                        organ_system[:, index] - system_tokens[:, index],
                        current[:, index : index + 1],
                        time,
                    ),
                    dim=-1,
                )
                organ_residual = (
                    0.75
                    * torch.sigmoid(self.organ_residual_gates[index])
                    * torch.tanh(
                        self.organ_residual_heads[index](organ_input)
                    )
                )
                if not disable_body_feedback:
                    body_input = torch.cat(
                        (
                            final_system[:, index] - organ_system[:, index],
                            current[:, index : index + 1],
                            time,
                        ),
                        dim=-1,
                    )
                    body_residual = (
                        0.75
                        * torch.sigmoid(self.body_residual_gates[index])
                        * torch.tanh(self.body_residual_heads[index](body_input))
                    )
            residual = system_residual + organ_residual + body_residual
            residuals.append(residual)
            system_residuals.append(system_residual)
            organ_residuals.append(organ_residual)
            body_residuals.append(body_residual)
            predictions.append(
                safe_teacher[:, index : index + 1] + residual
            )
            if self.distributional_scale:
                scale_input = torch.cat(
                    (
                        final_system[:, index],
                        current[:, index : index + 1],
                        time,
                    ),
                    dim=-1,
                )
                scales.append(
                    nn.functional.softplus(
                        self.scale_heads[index](scale_input.detach())
                    )
                    + 0.03
                )

        probability = route_weights.clamp_min(1.0e-8)
        route_entropy = -(probability * probability.log()).sum(dim=-1)
        expert_weights = safe_teacher.new_zeros(
            (len(safe_teacher), self.target_count, 3)
        )
        expert_weights[..., 2 if disable_sparse_experts else 0] = 1.0
        output = {
            "value": torch.cat(predictions, dim=-1),
            "residual": torch.cat(residuals, dim=-1),
            "system_residual": torch.cat(system_residuals, dim=-1),
            "organ_residual": torch.cat(organ_residuals, dim=-1),
            "body_residual": torch.cat(body_residuals, dim=-1),
            "timescale_fast_weight": torch.ones_like(teacher_anchor),
            "route_weights": route_weights,
            "organ_coupling_weights": organ_coupling_weights,
            "target_organ_coupling": target_organ_coupling,
            "target_organ_coupling_weights": target_organ_coupling_weights,
            "route_entropy": route_entropy,
            "expert_weights": expert_weights,
            "expert_entropy": torch.zeros_like(teacher_anchor),
            "scale": (
                torch.cat(scales, dim=-1)
                if self.distributional_scale
                else torch.ones_like(teacher_anchor)
            ),
            "system_tokens": final_system,
            "organ_tokens": organ_tokens,
            "organ_presence": effective_organ_presence,
            "body_token": body_token,
        }
        if patient_state_output is not None:
            output["patient_state"] = patient_state_output["patient_state"]
            output["patient_state_uncertainty"] = patient_state_output["uncertainty"]
            output["patient_organ_residual"] = patient_state_output["organ_residual"]
        return output


def assert_adjacent_hierarchy(module: nn.Module) -> None:
    """Fail if a direct system/body projection is added later."""

    contract = getattr(module, "architecture_contract", {})
    allowed = set(contract.get("allowed_directed_edges", ()))
    forbidden = set(contract.get("forbidden_directed_edges", ()))
    expected = {
        "system_to_organ",
        "organ_to_organ",
        "organ_to_system",
        "organ_to_body",
        "body_to_organ",
    }
    if allowed != expected:
        raise AssertionError(f"unexpected adjacent edge contract: {allowed}")
    parameter_names = set(dict(module.named_parameters()))
    for edge in forbidden:
        if any(name == edge or name.startswith(f"{edge}.") for name in parameter_names):
            raise AssertionError(f"forbidden direct hierarchy edge: {edge}")
