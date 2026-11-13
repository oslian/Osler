"""Persistent, auditable physiology state for whole-body JEPA candidates.

The layer keeps one body state and one state per anatomical/functional organ
group.  Observations update that state through an innovation gate; forecasts
advance the same state recurrently through a sparse organ graph.  It is a
factual state-space model, not a treatment-effect or clinical-decision model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class PersistentPhysiologyState:
    """One patient's continuously updated latent physiology."""

    body: torch.Tensor
    organs: torch.Tensor
    uncertainty: torch.Tensor
    elapsed_hours: torch.Tensor


def body_system_adjacency(module_names: Sequence[str]) -> torch.Tensor:
    """Return a sparse, unsigned physiology topology in checkpoint order.

    Edges only state which systems may exchange information.  They do not
    encode a treatment effect, direction, or clinical claim.
    """

    names = tuple(str(name) for name in module_names)
    aliases = {
        "cardiovascular_perfusion": "cardio",
        "cardiovascular_instability": "cardio",
        "respiratory": "respiratory",
        "renal_urinary": "renal",
        "aki": "renal",
        "acid_base_electrolyte": "electrolyte",
        "electrolyte_acid_base": "electrolyte",
        "endocrine_metabolic": "endocrine",
        "endocrine_stress": "endocrine",
        "dka": "endocrine",
        "hepatic_synthetic": "hepatic",
        "hepatic_failure": "hepatic",
        "gastrointestinal_pancreatic": "gi",
        "gi_pancreatic_nutrition": "gi",
        "hematologic_coagulation": "heme",
        "coagulopathy_heme": "heme",
        "immune_inflammatory": "immune",
        "sepsis": "immune",
        "cardiac_injury": "cardiac_injury",
        "musculoskeletal_injury": "musculoskeletal",
        "musculoskeletal_rhabdo": "musculoskeletal",
    }
    groups = tuple(aliases.get(name, name) for name in names)
    allowed = {
        frozenset(("cardio", "respiratory")),
        frozenset(("cardio", "renal")),
        frozenset(("cardio", "heme")),
        frozenset(("cardio", "cardiac_injury")),
        frozenset(("cardio", "immune")),
        frozenset(("respiratory", "electrolyte")),
        frozenset(("respiratory", "immune")),
        frozenset(("renal", "electrolyte")),
        frozenset(("renal", "endocrine")),
        frozenset(("renal", "hepatic")),
        frozenset(("renal", "musculoskeletal")),
        frozenset(("electrolyte", "endocrine")),
        frozenset(("electrolyte", "gi")),
        frozenset(("hepatic", "gi")),
        frozenset(("hepatic", "heme")),
        frozenset(("hepatic", "immune")),
        frozenset(("gi", "endocrine")),
        frozenset(("heme", "immune")),
        frozenset(("immune", "musculoskeletal")),
        frozenset(("heart", "blood_vessels")),
        frozenset(("heart", "lungs")),
        frozenset(("heart", "kidneys")),
        frozenset(("heart", "thyroid")),
        frozenset(("heart", "adrenal_glands")),
        frozenset(("blood_vessels", "kidneys")),
        frozenset(("blood_vessels", "liver")),
        frozenset(("lungs", "kidneys")),
        frozenset(("kidneys", "liver")),
        frozenset(("kidneys", "skeletal_muscle")),
        frozenset(("liver", "pancreas")),
        frozenset(("liver", "stomach")),
        frozenset(("liver", "intestines")),
        frozenset(("liver", "bone_marrow")),
        frozenset(("pancreas", "stomach")),
        frozenset(("pancreas", "intestines")),
        frozenset(("bone_marrow", "spleen")),
    }
    adjacency = torch.eye(len(names), dtype=torch.float32)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if frozenset((groups[left], groups[right])) in allowed:
                adjacency[left, right] = 1.0
                adjacency[right, left] = 1.0
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


class LinearConservationProjector(nn.Module):
    """Project states onto exact linear conservation equalities.

    For constraints ``A x = total``, this is the minimum-L2 correction.  The
    matrix is fixed and auditable; no learned parameter can weaken it.
    """

    def __init__(self, constraint_matrix: torch.Tensor) -> None:
        super().__init__()
        matrix = torch.as_tensor(constraint_matrix, dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("constraint_matrix must have shape [constraints, state]")
        gram = matrix @ matrix.transpose(0, 1)
        if int(torch.linalg.matrix_rank(gram)) != gram.shape[0]:
            raise ValueError("conservation constraints must be linearly independent")
        self.register_buffer("matrix", matrix, persistent=True)
        self.register_buffer("gram_inverse", torch.linalg.inv(gram), persistent=True)

    def forward(self, state: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.matrix.shape[-1]:
            raise ValueError("state width does not match conservation matrix")
        if totals.shape != state.shape[:-1] + (self.matrix.shape[0],):
            raise ValueError("totals must match state batch and constraint count")
        residual = state @ self.matrix.transpose(0, 1) - totals
        correction = residual @ self.gram_inverse @ self.matrix
        return state - correction


class PhysiologicalReadoutProjector(nn.Module):
    """Make derived laboratory readouts exactly consistent when available."""

    def __init__(
        self,
        variables: Sequence[str],
        medians: Sequence[float],
        scales: Sequence[float],
    ) -> None:
        super().__init__()
        self.variables = tuple(str(value) for value in variables)
        if len(self.variables) != len(medians) or len(medians) != len(scales):
            raise ValueError("variables, medians, and scales must have equal length")
        self.index = {name: position for position, name in enumerate(self.variables)}
        self.register_buffer("medians", torch.as_tensor(medians, dtype=torch.float32))
        self.register_buffer("scales", torch.as_tensor(scales, dtype=torch.float32))

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        if normalized.shape[-1] != len(self.variables):
            raise ValueError("prediction width does not match variables")
        physical = normalized * self.scales + self.medians
        projected = physical.clone()
        if all(name in self.index for name in ("anion_gap", "sodium", "chloride", "bicarbonate")):
            projected[..., self.index["anion_gap"]] = (
                physical[..., self.index["sodium"]]
                - physical[..., self.index["chloride"]]
                - physical[..., self.index["bicarbonate"]]
            )
        if all(name in self.index for name in ("serum_osmolality", "sodium", "glucose", "bun")):
            projected[..., self.index["serum_osmolality"]] = (
                2.0 * physical[..., self.index["sodium"]]
                + physical[..., self.index["glucose"]] / 18.0
                + physical[..., self.index["bun"]] / 2.8
            )
        return (projected - self.medians) / self.scales


class PersistentPhysiologyStateLayer(nn.Module):
    """Recurrent whole-body state with sparse organ-to-organ communication."""

    def __init__(
        self,
        *,
        body_dim: int,
        organ_dim: int,
        organ_count: int,
        adjacency: torch.Tensor,
        max_step_hours: float = 1.0,
        max_state_delta: float = 0.20,
        body_feedback: bool = True,
    ) -> None:
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.shape != (organ_count, organ_count):
            raise ValueError("adjacency must match organ_count")
        if max_step_hours <= 0.0 or max_state_delta <= 0.0:
            raise ValueError("step and state delta limits must be positive")
        self.body_dim = int(body_dim)
        self.organ_dim = int(organ_dim)
        self.organ_count = int(organ_count)
        self.max_step_hours = float(max_step_hours)
        self.max_state_delta = float(max_state_delta)
        self.body_feedback = bool(body_feedback)
        self.register_buffer("adjacency", adjacency, persistent=True)
        self.organ_message = nn.Sequential(
            nn.Linear(organ_dim, organ_dim),
            nn.LayerNorm(organ_dim),
            nn.SiLU(),
        )
        self.organ_dynamics = nn.Sequential(
            nn.Linear(3 * organ_dim + body_dim + 4, 2 * organ_dim),
            nn.LayerNorm(2 * organ_dim),
            nn.SiLU(),
            nn.Linear(2 * organ_dim, organ_dim),
            nn.Tanh(),
        )
        self.body_dynamics = nn.Sequential(
            nn.Linear(body_dim + organ_dim + 4, 2 * body_dim),
            nn.LayerNorm(2 * body_dim),
            nn.SiLU(),
            nn.Linear(2 * body_dim, body_dim),
            nn.Tanh(),
        )
        self.body_update_gate = nn.Sequential(
            nn.Linear(2 * body_dim + 1, body_dim), nn.Sigmoid()
        )
        self.organ_update_gate = nn.Sequential(
            nn.Linear(2 * organ_dim + 1, organ_dim), nn.Sigmoid()
        )
        self.uncertainty_growth = nn.Parameter(torch.full((organ_count,), -3.0))

    @staticmethod
    def _time_features(delta_hours: torch.Tensor) -> torch.Tensor:
        delta = delta_hours.clamp(min=0.0, max=168.0)
        log_delta = torch.log1p(delta)
        return torch.stack(
            (
                log_delta / torch.log1p(delta.new_tensor(48.0)),
                delta / 6.0,
                torch.sin(log_delta),
                torch.cos(log_delta),
            ),
            dim=-1,
        )

    def initialize(
        self,
        body: torch.Tensor,
        organs: torch.Tensor,
        organ_presence: torch.Tensor,
    ) -> PersistentPhysiologyState:
        if organs.shape[1:] != (self.organ_count, self.organ_dim):
            raise ValueError("organ state shape does not match layer")
        if body.shape[-1] != self.body_dim:
            raise ValueError("body state width does not match layer")
        presence = organ_presence.to(dtype=organs.dtype)
        uncertainty = 1.0 - presence
        return PersistentPhysiologyState(
            body=body,
            organs=organs * presence.unsqueeze(-1),
            uncertainty=uncertainty,
            elapsed_hours=body.new_zeros(body.shape[0]),
        )

    def update(
        self,
        state: PersistentPhysiologyState,
        observed_body: torch.Tensor,
        observed_organs: torch.Tensor,
        organ_presence: torch.Tensor,
        delta_hours: torch.Tensor,
    ) -> PersistentPhysiologyState:
        """Correct an already predicted state; delta_hours conditions the gate only."""
        if observed_body.shape != state.body.shape or observed_organs.shape != state.organs.shape:
            raise ValueError("observation shapes must match persistent state")
        if organ_presence.shape != state.uncertainty.shape or delta_hours.shape != state.elapsed_hours.shape:
            raise ValueError("presence and elapsed time must match persistent state")
        if not torch.isfinite(organ_presence).all() or ((organ_presence < 0) | (organ_presence > 1)).any():
            raise ValueError("organ presence must be finite in [0,1]")
        if not torch.isfinite(delta_hours).all() or (delta_hours < 0).any():
            raise ValueError("observation elapsed time must be finite and nonnegative")
        presence = organ_presence.to(dtype=state.organs.dtype)
        organ_observed = presence > 0
        body_observed = organ_observed.any(dim=1, keepdim=True)
        if not torch.isfinite(observed_organs[organ_observed]).all() or not torch.isfinite(observed_body[body_observed[:, 0]]).all():
            raise ValueError("present observations must be finite")
        # Mask before projection: multiplying a missing NaN by zero is not safe.
        observed_body = torch.where(body_observed, observed_body, state.body)
        observed_organs = torch.where(organ_observed.unsqueeze(-1), observed_organs, state.organs)
        body_gate = self.body_update_gate(
            torch.cat((state.body, observed_body, delta_hours[:, None]), dim=-1)
        )
        organ_delta = delta_hours[:, None, None].expand(-1, self.organ_count, 1)
        organ_gate = self.organ_update_gate(
            torch.cat((state.organs, observed_organs, organ_delta), dim=-1)
        ) * presence.unsqueeze(-1)
        body = torch.where(body_observed,
            state.body + body_gate * (observed_body - state.body), state.body)
        organs = state.organs + organ_gate * (observed_organs - state.organs)
        uncertainty = torch.where(
            presence > 0.0,
            state.uncertainty * (1.0 - organ_gate.mean(dim=-1)),
            state.uncertainty,
        )
        return PersistentPhysiologyState(
            body=body,
            organs=organs,
            uncertainty=uncertainty.clamp(0.0, 10.0),
            elapsed_hours=state.elapsed_hours,
        )

    def advance(self, state, delta_hours):
        """Integrate a nonnegative interval exactly once using bounded steps."""
        if delta_hours.shape != state.elapsed_hours.shape:
            raise ValueError("elapsed interval must match patients")
        if not torch.isfinite(delta_hours).all() or (delta_hours < 0).any():
            raise ValueError("elapsed interval must be finite and nonnegative")
        remaining = delta_hours
        count = int(torch.ceil(remaining.max() / self.max_step_hours).item())
        for _ in range(count):
            step = torch.minimum(remaining, torch.full_like(remaining, self.max_step_hours))
            state = self.transition(state, step)
            remaining = (remaining-step).clamp_min(0)
        return state

    def replay_observations(self, initial, observed_body, observed_organs,
                            organ_presence, event_hours):
        """Replay grouped available-time observations on one persistent state.

        event_hours are relative to initialization, not specimen collection.
        The caller must supply deduplicated simultaneous groups and no future
        events. No padding events: replay each patient's actual stream.
        """
        batch, steps = event_hours.shape
        if batch != initial.body.shape[0] or steps == 0:
            raise ValueError("nonempty aligned patient event streams required")
        if observed_body.shape != (batch, steps, self.body_dim):
            raise ValueError("body observations must match event stream")
        if observed_organs.shape != (batch, steps, self.organ_count, self.organ_dim):
            raise ValueError("organ observations must match event stream")
        if organ_presence.shape != (batch, steps, self.organ_count):
            raise ValueError("presence must match event stream")
        if not torch.isfinite(event_hours).all() or (event_hours[:, 0] < initial.elapsed_hours).any():
            raise ValueError("events must be finite and not precede state")
        if (event_hours[:, 1:] <= event_hours[:, :-1]).any():
            raise ValueError("simultaneous events must be grouped; times must strictly increase")
        state = initial
        for index in range(steps):
            delta = event_hours[:, index]-state.elapsed_hours
            state = self.advance(state, delta)
            state = self.update(state, observed_body[:, index], observed_organs[:, index],
                organ_presence[:, index], delta)
        return state

    def transition(
        self,
        state: PersistentPhysiologyState,
        delta_hours: torch.Tensor,
    ) -> PersistentPhysiologyState:
        features = self._time_features(delta_hours)
        messages = torch.einsum(
            "ij,bjd->bid", self.adjacency, self.organ_message(state.organs)
        )
        # A graph-isolation experiment must close the pooled-body bypass too.
        body_input_state = state.body if self.body_feedback else torch.zeros_like(state.body)
        body_expanded = body_input_state[:, None, :].expand(-1, self.organ_count, -1)
        time_expanded = features[:, None, :].expand(-1, self.organ_count, -1)
        organ_input = torch.cat(
            (state.organs, messages, state.organs - messages, body_expanded, time_expanded),
            dim=-1,
        )
        step_scale = (delta_hours / 6.0).clamp(min=0.0, max=1.0)
        organ_delta = self.max_state_delta * step_scale[:, None, None] * self.organ_dynamics(organ_input)
        organs = state.organs + organ_delta
        pooled = organs.mean(dim=1)
        body_input = torch.cat((state.body, pooled, features), dim=-1)
        body_delta = self.max_state_delta * step_scale[:, None] * self.body_dynamics(body_input)
        body = state.body + body_delta
        uncertainty = state.uncertainty + (
            torch.nn.functional.softplus(self.uncertainty_growth)[None, :]
            * delta_hours[:, None]
            / 24.0
        )
        return PersistentPhysiologyState(
            body=body,
            organs=organs,
            uncertainty=uncertainty.clamp(0.0, 10.0),
            elapsed_hours=state.elapsed_hours + delta_hours,
        )

    def rollout(
        self,
        initial: PersistentPhysiologyState,
        horizons: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if horizons.ndim == 1:
            horizons = horizons.unsqueeze(0).expand(initial.body.shape[0], -1)
        if horizons.ndim != 2 or horizons.shape[0] != initial.body.shape[0]:
            raise ValueError("horizons must have shape [batch, steps]")
        if bool((horizons[:, 1:] < horizons[:, :-1]).any()):
            raise ValueError("horizons must be non-decreasing")
        state = initial
        body_path = []
        organ_path = []
        uncertainty_path = []
        previous = horizons.new_zeros(horizons.shape[0])
        for position in range(horizons.shape[1]):
            remaining = (horizons[:, position] - previous).clamp_min(0.0)
            max_steps = int(
                torch.ceil(remaining.max() / self.max_step_hours).item()
            )
            for _ in range(max_steps):
                step = torch.minimum(
                    remaining,
                    torch.full_like(remaining, self.max_step_hours),
                )
                if not bool((step > 0.0).any()):
                    break
                state = self.transition(state, step)
                remaining = (remaining - step).clamp_min(0.0)
            body_path.append(state.body)
            organ_path.append(state.organs)
            uncertainty_path.append(state.uncertainty)
            previous = horizons[:, position]
        return {
            "body": torch.stack(body_path, dim=1),
            "organs": torch.stack(organ_path, dim=1),
            "uncertainty": torch.stack(uncertainty_path, dim=1),
        }


__all__ = [
    "LinearConservationProjector",
    "PersistentPhysiologyState",
    "PersistentPhysiologyStateLayer",
    "PhysiologicalReadoutProjector",
    "body_system_adjacency",
]
