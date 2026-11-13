"""Experimental local-organ residual on the existing shared JEPA encoder.

Graph edges represent predictive information exchange, NOT material fluxes.
The body summary is retained, but cannot bypass the graph into value heads.
"""

import torch
from torch import nn

from osler_jepa.organ_schema import canonical_organ_adjacency
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


ORGANS = ("heart", "blood_vessels", "lungs", "kidneys", "pancreas", "bone_marrow", "brain")
ARMS = ("none", "sparse", "permuted", "dense")


def graph_for(arm: str) -> torch.Tensor:
    graph = torch.tensor(canonical_organ_adjacency(ORGANS), dtype=torch.float32)
    if arm == "none":
        graph.zero_()
    elif arm == "permuted":
        # Predeclared node permutation; same edge count and degree multiset,
        # not the same per-organ degree. Report this limitation explicitly.
        permutation = torch.tensor([4, 2, 5, 0, 6, 1, 3])
        graph = graph[permutation][:, permutation]
    elif arm == "dense":
        graph = torch.ones_like(graph) - torch.eye(len(ORGANS))
    elif arm != "sparse":
        raise ValueError(f"unknown graph arm: {arm}")
    graph += torch.eye(len(ORGANS))
    return graph / graph.sum(-1, keepdim=True)


def mask_to_owner(observations, current, mask, ages, presence, allowed):
    """Remove non-owner values AND their entire encoded history before JEPA."""
    obs, cur, m, age, p = (v.clone() for v in (observations, current, mask, ages, presence))
    hidden = sorted(set(range(current.shape[-1])) - set(allowed))
    n = current.shape[-1]
    obs[..., hidden] = 0
    obs[..., [n + i for i in hidden]] = 0
    obs[..., [2 * n + i for i in hidden]] = 0
    # No system-presence identity may reveal masked observations.
    obs[..., 3 * n + 1:-1] = 0
    cur[:, hidden], m[:, hidden], age[:, hidden] = 0, 0, 0
    p.zero_()
    return obs, cur, m, age, p


class OrganExchangeResidual(nn.Module):
    """All target deltas read one local-organ recurrent trajectory.

    Zero-start value and future-latent residuals augment a frozen owner JEPA.
    No raw non-owner input, global-body feature or independent horizon head
    can bypass the graph. Graph changes are audit-only, never runtime routing.
    """

    def __init__(self, owners, *, arm: str, latent: int = 64):
        super().__init__()
        self.register_buffer("owners", torch.tensor(owners, dtype=torch.long))
        self.state_layer = PersistentPhysiologyStateLayer(
            body_dim=latent, organ_dim=latent, organ_count=len(ORGANS),
            adjacency=graph_for(arm), body_feedback=False,
        )
        self.value_heads = nn.ModuleList()
        self.future_heads = nn.ModuleList()
        for _ in owners:
            for heads, width in ((self.value_heads, 1), (self.future_heads, latent)):
                head = nn.Sequential(nn.Linear(latent + 6, latent), nn.SiLU(), nn.Linear(latent, width))
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                heads.append(head)

    def forward(self, organs, organ_presence, current, current_mask, actual_hours, base, base_latent):
        if actual_hours.shape != current.shape or not torch.isfinite(actual_hours).all():
            raise ValueError("actual_hours must be finite and match target shape")
        if (actual_hours < 0).any() or (actual_hours > 48).any():
            raise ValueError("audit supports elapsed times in [0,48] hours")
        initial = self.state_layer.initialize(organs.mean(1), organs, organ_presence)
        # A fixed integration grid avoids making one target depend on another
        # target's (future) measurement time through adaptive step boundaries.
        steps = max(1, int(torch.ceil(actual_hours.max()).item()))
        grid = torch.arange(steps + 1, device=organs.device, dtype=organs.dtype)
        path = self.state_layer.rollout(initial, grid)
        batch = torch.arange(len(organs), device=organs.device)
        values, latents = [], []
        for i, owner in enumerate(self.owners):
            time = actual_hours[:, i]
            left, right = time.floor().long(), time.ceil().long()
            fraction = (time - left)[:, None]
            at_time = ((1 - fraction) * path["organs"][batch, left, owner]
                       + fraction * path["organs"][batch, right, owner])
            change = at_time - initial.organs[:, owner]
            features = torch.cat((change, self.state_layer._time_features(actual_hours[:, i]),
                                  current[:, i:i + 1], current_mask[:, i:i + 1]), dim=-1)
            values.append(base[:, i] + .25 * torch.tanh(self.value_heads[i](features)[:, 0]))
            latents.append(base_latent[:, i] + self.future_heads[i](features))
        return {"value": torch.stack(values, 1), "future_latent": torch.stack(latents, 1),
                "body": path["body"], "organs": path["organs"]}
