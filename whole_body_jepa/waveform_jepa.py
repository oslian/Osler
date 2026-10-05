"""Small JEPA for pre-anchor waveform-to-future-event representation learning."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class WaveformEventJEPA(nn.Module):
    """Predict a future physiologic latent and an incident event probability."""

    def __init__(
        self,
        context_dim: int,
        waveform_dim: int,
        future_dim: int,
        *,
        hidden_dim: int = 64,
        latent_dim: int = 24,
        target_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.target_momentum = float(target_momentum)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.waveform_encoder = nn.Sequential(
            nn.Linear(waveform_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.future_encoder = nn.Sequential(
            nn.Linear(future_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.target_encoder = deepcopy(self.future_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.future_decoder = nn.Linear(latent_dim, future_dim)
        self.event_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        context: torch.Tensor,
        waveform: torch.Tensor,
        future: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_latent = self.context_encoder(context)
        waveform_latent = self.waveform_encoder(waveform)
        predicted_future = self.predictor(torch.cat((context_latent, waveform_latent), dim=-1))
        logit = self.event_head(torch.cat((context_latent, predicted_future), dim=-1)).squeeze(-1)
        output = {"logit": logit, "predicted_future_latent": predicted_future}
        if future is not None:
            online_future = self.future_encoder(future)
            with torch.no_grad():
                target = self.target_encoder(future)
            output["online_future_latent"] = online_future
            output["target_future_latent"] = target
            output["future_reconstruction"] = self.future_decoder(online_future)
            output["future"] = future
        return output

    def loss(
        self,
        output: dict[str, torch.Tensor],
        label: torch.Tensor,
        *,
        jepa_weight: float = 0.25,
    ) -> torch.Tensor:
        event_loss = F.binary_cross_entropy_with_logits(output["logit"], label)
        target = output.get("target_future_latent")
        if target is None:
            return event_loss
        latent_loss = 1.0 - F.cosine_similarity(
            output["predicted_future_latent"], target, dim=-1
        ).mean()
        reconstruction = F.smooth_l1_loss(output["future_reconstruction"], output["future"])
        return event_loss + float(jepa_weight) * latent_loss + 0.05 * reconstruction

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for online, target in zip(self.future_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(self.target_momentum).add_(
                online.data, alpha=1.0 - self.target_momentum
            )
