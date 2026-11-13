"""Diagnostic target projection with separate event-time and correction masks.

Not a full-body filter: non-target values are deliberately absent. This tests
whether extra no-content events can advance time without inventing innovations.
"""
import torch
from osler_jepa.patient_state import NeuralBeliefFilter
from whole_body_jepa.native_projection_control import target_projection_input


class PredictOnlyControl(NeuralBeliefFilter):
    def __init__(self, input_dim, hidden, variable_count, target_index, owner_index):
        super().__init__(input_dim, hidden)
        self.variable_count = variable_count
        self.target_index = target_index
        self.owner_index = owner_index

    def forward(self, observations, observation_mask=None):
        if observations.ndim != 3 or observation_mask is None:
            raise ValueError('explicit native history and mask required')
        if observation_mask.shape != observations.shape[:2]:
            raise ValueError('unaligned step mask')
        active = observation_mask.bool() | observations[..., -2].bool()
        correction = observations[..., self.variable_count+self.target_index].bool()
        if (correction & ~active).any():
            raise ValueError('observed target must activate event')
        projected = target_projection_input(observations, self.variable_count,
            self.target_index, self.owner_index)
        time = observations[..., -1:].detach()
        belief = observations.new_zeros((len(observations), self.hidden))
        previous = time[:, 0]
        has_previous = torch.zeros((len(observations), 1), dtype=torch.bool, device=time.device)
        for step in range(observations.shape[1]):
            event = active[:, step:step+1]
            delta = torch.where(has_previous, (time[:, step]-previous).abs().clamp(0, 1),
                torch.zeros_like(previous))
            predicted = belief * torch.exp(-torch.nn.functional.softplus(self.decay_rate)*delta)
            observation = self.observation_projection(projected[:, step])
            combined = torch.cat([predicted, observation], dim=-1)
            innovation = torch.tanh(self.innovation_projection(combined))
            gate = torch.sigmoid(self.update_gate(combined))
            corrected = predicted + gate*(innovation-predicted)
            next_state = torch.where(correction[:, step:step+1], corrected, predicted)
            belief = torch.where(event, next_state, belief)
            previous = torch.where(event, time[:, step], previous)
            has_previous |= event
        uncertainty = torch.nn.functional.softplus(self.uncertainty_head(belief))+1e-4
        return belief, uncertainty
