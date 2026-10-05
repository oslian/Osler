"""Convolutional multi-scale JEPA for raw ICU waveforms."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class WaveformScaleEncoder(nn.Module):
    def __init__(self, channels: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels * 2, 24, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv1d(24, hidden, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded_mask = mask.unsqueeze(-1).expand_as(values)
        encoded = self.network(torch.cat((values * expanded_mask, expanded_mask), dim=1))
        return encoded.squeeze(-1)


class MultiScaleWaveformEncoder(nn.Module):
    def __init__(self, scale_count: int = 3, hidden: int = 32, latent: int = 64) -> None:
        super().__init__()
        self.scale_count = int(scale_count)
        self.branches = nn.ModuleList(
            WaveformScaleEncoder(hidden=hidden) for _ in range(self.scale_count)
        )
        self.projection = nn.Sequential(
            nn.Linear(self.scale_count * hidden, latent),
            nn.LayerNorm(latent),
            nn.GELU(),
            nn.Linear(latent, latent),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = [
            branch(values[:, index], mask[:, index])
            for index, branch in enumerate(self.branches)
        ]
        return self.projection(torch.cat(encoded, dim=-1))


class RawWaveformEventJEPA(nn.Module):
    """Predict the next-minute waveform latent and a later clinical event."""

    def __init__(
        self,
        table_dim: int,
        *,
        scale_count: int = 3,
        latent: int = 64,
        target_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.scale_count = int(scale_count)
        self.target_momentum = float(target_momentum)
        self.context_encoder = MultiScaleWaveformEncoder(scale_count, latent=latent)
        self.target_encoder = deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, latent)
        )
        self.table_encoder = nn.Sequential(
            nn.Linear(table_dim, latent), nn.LayerNorm(latent), nn.GELU()
        )
        self.event_head = nn.Sequential(
            nn.Linear(latent * 2, latent),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(latent, 1),
        )

    @staticmethod
    def _future_view(
        future: torch.Tensor, future_mask: torch.Tensor, scale_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            future.unsqueeze(1).expand(-1, scale_count, -1, -1),
            future_mask.unsqueeze(1).expand(-1, scale_count, -1),
        )

    def forward(
        self,
        table: torch.Tensor,
        past: torch.Tensor,
        past_mask: torch.Tensor,
        future: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(past, past_mask)
        predicted_future = self.predictor(context)
        table_latent = self.table_encoder(table)
        output = {
            "logit": self.event_head(torch.cat((table_latent, context), dim=-1)).squeeze(-1),
            "context_latent": context,
            "predicted_future_latent": predicted_future,
        }
        if future is not None and future_mask is not None:
            target_values, target_mask = self._future_view(
                future, future_mask, self.scale_count
            )
            with torch.no_grad():
                output["target_future_latent"] = self.target_encoder(
                    target_values, target_mask
                )
        return output

    @staticmethod
    def _variance_loss(latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[0] < 2:
            return latent.new_zeros(())
        standard_deviation = torch.sqrt(latent.var(dim=0) + 1e-4)
        return F.relu(1.0 - standard_deviation).mean()

    @staticmethod
    def _covariance_loss(latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[0] < 2:
            return latent.new_zeros(())
        centered = latent - latent.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (latent.shape[0] - 1)
        off_diagonal = covariance - torch.diag(torch.diag(covariance))
        return off_diagonal.square().sum() / latent.shape[1]

    def jepa_loss(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        target = output["target_future_latent"]
        predicted = output["predicted_future_latent"]
        alignment = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()
        variance = self._variance_loss(output["context_latent"])
        covariance = self._covariance_loss(output["context_latent"])
        return alignment + 0.10 * variance + 0.01 * covariance

    def event_loss(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        jepa_weight: float = 0.10,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(output["logit"], labels) + float(
            jepa_weight
        ) * self.jepa_loss(output)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for online, target in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(self.target_momentum).add_(
                online.data, alpha=1.0 - self.target_momentum
            )


class MultiTaskResidualWaveformJEPA(RawWaveformEventJEPA):
    """Anchor event risk to table physiology and supervise future SpO2 directly."""

    def __init__(
        self,
        table_dim: int,
        *,
        scale_count: int = 3,
        auxiliary_dim: int = 4,
        latent: int = 64,
    ) -> None:
        super().__init__(table_dim, scale_count=scale_count, latent=latent)
        self.table_event_head = nn.Linear(latent, 1)
        self.waveform_residual_head = nn.Sequential(
            nn.Linear(latent, latent // 2),
            nn.GELU(),
            nn.Linear(latent // 2, 1),
        )
        nn.init.zeros_(self.waveform_residual_head[-1].weight)
        nn.init.zeros_(self.waveform_residual_head[-1].bias)
        self.physiology_head = nn.Sequential(
            nn.Linear(latent, latent // 2),
            nn.GELU(),
            nn.Linear(latent // 2, auxiliary_dim),
        )

    def forward(
        self,
        table: torch.Tensor,
        past: torch.Tensor,
        past_mask: torch.Tensor,
        future: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(past, past_mask)
        predicted_future = self.predictor(context)
        table_latent = self.table_encoder(table)
        table_logit = self.table_event_head(table_latent).squeeze(-1)
        residual_logit = self.waveform_residual_head(context).squeeze(-1)
        output = {
            "logit": table_logit + residual_logit,
            "table_logit": table_logit,
            "residual_logit": residual_logit,
            "context_latent": context,
            "predicted_future_latent": predicted_future,
            "physiology_prediction": self.physiology_head(predicted_future),
        }
        if future is not None and future_mask is not None:
            target_values, target_mask = self._future_view(
                future, future_mask, self.scale_count
            )
            with torch.no_grad():
                output["target_future_latent"] = self.target_encoder(
                    target_values, target_mask
                )
        return output

    def multitask_loss(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        physiology_target: torch.Tensor,
        *,
        jepa_weight: float = 0.10,
        auxiliary_weight: float = 0.30,
        residual_weight: float = 0.01,
    ) -> torch.Tensor:
        event = F.binary_cross_entropy_with_logits(output["logit"], labels)
        physiology = F.smooth_l1_loss(
            output["physiology_prediction"], physiology_target
        )
        residual = output["residual_logit"].square().mean()
        return (
            event
            + float(jepa_weight) * self.jepa_loss(output)
            + float(auxiliary_weight) * physiology
            + float(residual_weight) * residual
        )


class MultiHorizonWaveformJEPA(RawWaveformEventJEPA):
    """Predict ordered short-horizon event risks from one shared waveform state."""

    def __init__(
        self,
        table_dim: int,
        *,
        horizon_count: int,
        scale_count: int = 3,
        latent: int = 64,
    ) -> None:
        super().__init__(table_dim, scale_count=scale_count, latent=latent)
        self.horizon_count = int(horizon_count)
        self.event_head = nn.Sequential(
            nn.Linear(latent * 2, latent),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(latent, self.horizon_count),
        )

    def forward(
        self,
        table: torch.Tensor,
        past: torch.Tensor,
        past_mask: torch.Tensor,
        future: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(table, past, past_mask, future, future_mask)
        if output["logit"].ndim != 2:
            raise RuntimeError("multi-horizon event head must return a matrix")
        return output

    def multihorizon_loss(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        *,
        jepa_weight: float = 0.10,
    ) -> torch.Tensor:
        elementwise = F.binary_cross_entropy_with_logits(
            output["logit"], labels, reduction="none"
        )
        event = (elementwise * label_mask).sum() / label_mask.sum().clamp_min(1.0)
        return event + float(jepa_weight) * self.jepa_loss(output)


class DenseEventModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)
