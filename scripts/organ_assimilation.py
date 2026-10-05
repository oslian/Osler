"""Research-only observation correction into one persistent body, without time advance."""
from dataclasses import replace

import torch
from torch import nn


CHANNELS = ('immediate', 'delayed', 'body')


class OrganAssimilation(nn.Module):
    def __init__(self, layer):
        super().__init__()
        width = layer.organ_count * layer.organ_dim
        if width < layer.body_dim:
            raise ValueError('Body control requires expansion, not truncation')
        parameter = next(layer.parameters())
        self.projection = nn.Linear(width, layer.body_dim, bias=False).to(parameter)
        nn.init.zeros_(self.projection.weight)
        self.limit = layer.max_state_delta
        indices = torch.arange(width, device=parameter.device) % layer.body_dim
        counts = torch.bincount(indices, minlength=layer.body_dim)
        expansion = parameter.new_zeros((layer.body_dim, width))
        expansion[indices, torch.arange(width, device=parameter.device)] = counts[indices].to(parameter.dtype).rsqrt()
        self.register_buffer('body_expansion', expansion)

    def correct(self, prior, updated, pending, presence, channel):
        if channel not in CHANNELS:
            raise ValueError('Unknown assimilation channel')
        innovation = (updated.organs - prior.organs).flatten(1)
        if pending.shape != innovation.shape:
            raise ValueError('Pending correction must match organ innovation')
        observed = (presence > 0).any(dim=1, keepdim=True)
        if channel == 'immediate':
            features = innovation
        elif channel == 'delayed':
            features = pending
        else:
            features = (updated.body - prior.body) @ self.body_expansion
        # A missing observation cannot flush a delayed update or write a body correction.
        features = torch.where(observed, features, torch.zeros_like(features))
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite assimilation features')
        correction = self.limit * torch.tanh(self.projection(features))
        body = torch.where(observed, updated.body + correction, updated.body)
        next_pending = torch.where(observed, innovation, pending)
        return replace(updated, body=body), next_pending


def replay_assimilated(layer, bridge, example, channel):
    if channel not in CHANNELS:
        raise ValueError('Unknown assimilation channel')
    body, organs, presence, times = [example[k] for k in ('body', 'organs', 'presence', 'times')]
    anchor = times.new_tensor(example['anchor'])
    if (not len(times) or times.shape != (len(body),)
            or len(organs) != len(times) or len(presence) != len(times)
            or not torch.isfinite(times).all() or not torch.isfinite(anchor)
            or (times > anchor).any() or (times[1:] <= times[:-1]).any()):
        raise ValueError('Invalid cached event times or alignment')
    state = layer.initialize(body.new_zeros((1, layer.body_dim)),
        body.new_zeros((1, layer.organ_count, layer.organ_dim)),
        body.new_zeros((1, layer.organ_count)))
    # Local to this patient's replay. It never survives into another patient's call.
    pending = organs.new_zeros((1, layer.organ_count * layer.organ_dim))
    relative = (times - times[0]).to(body.dtype)
    for i in range(len(times)):
        delta = relative[i:i+1] - state.elapsed_hours
        prior = layer.advance(state, delta)
        updated = layer.update(prior, body[i:i+1], organs[i:i+1], presence[i:i+1], delta)
        state, pending = bridge.correct(prior, updated, pending, presence[i:i+1], channel)
    gap = (anchor - times[-1]).to(body.dtype).reshape(1)
    # Do not flush a pending organ event at the query or invent time to expose it.
    return layer.advance(state, gap)
