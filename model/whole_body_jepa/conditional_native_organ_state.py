"""One replayed anatomical state with owner-local future-change readouts."""
import torch
from torch import nn
from osler_jepa.organ_schema import CANONICAL_ORGAN_VARIABLES
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class ConditionalNativeOrganStateResidual(nn.Module):
    def __init__(self, jepa, variables, targets):
        super().__init__()
        self.jepa = jepa
        self.variables = tuple(variables)
        self.targets = tuple(targets)
        self.organ_names = tuple(CANONICAL_ORGAN_VARIABLES)
        owners = {v:i for i,o in enumerate(self.organ_names) for v in CANONICAL_ORGAN_VARIABLES[o]}
        if not targets or any(t not in owners or t not in variables for t in targets):
            raise ValueError('targets require explicit existing anatomical owner')
        self.register_buffer('owners', torch.tensor([owners[t] for t in targets]))
        self.layer = PersistentPhysiologyStateLayer(body_dim=jepa.latent, organ_dim=jepa.token_dim,
            organ_count=len(self.organ_names), adjacency=torch.eye(len(self.organ_names)), body_feedback=False)
        self.value_heads = nn.ModuleList()
        self.latent_heads = nn.ModuleList()
        for _ in targets:
            for heads, width in ((self.value_heads, 1), (self.latent_heads, jepa.latent)):
                head = nn.Sequential(nn.Linear(2*jepa.token_dim+6, jepa.hidden), nn.SiLU(),
                    nn.Linear(jepa.hidden, width))
                nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
                heads.append(head)

    def forward(self, tokens, event_hours, anchor_hour, current, current_mask,
                actual_hours, base_value, base_latent):
        """One unpadded patient stream; no labels or future events are accepted."""
        count = len(self.targets)
        if tokens.ndim != 2 or event_hours.shape != (len(tokens),) or len(tokens) == 0:
            raise ValueError('one nonempty grouped event stream required')
        if current.shape != (1,count) or current_mask.shape != current.shape or actual_hours.shape != current.shape:
            raise ValueError('target inputs must be [1,targets]')
        if base_value.shape != current.shape or base_latent.shape != (1,count,self.jepa.latent):
            raise ValueError('baseline predictions must match target contract')
        if not torch.isfinite(event_hours).all() or not torch.isfinite(torch.as_tensor(anchor_hour)) or (event_hours > anchor_hour).any():
            raise ValueError('only finite pre-anchor available times allowed')
        if not torch.isfinite(actual_hours).all() or (actual_hours < 0).any() or (actual_hours > 48).any():
            raise ValueError('forecast supports finite [0,48]h queries')
        body, organs, presence = encode_native_organ_observations(self.jepa, self.variables, tokens)
        initial = self.layer.initialize(tokens.new_zeros((1,self.jepa.latent)),
            tokens.new_zeros((1,len(self.organ_names),self.jepa.token_dim)),
            tokens.new_zeros((1,len(self.organ_names))))
        relative = (event_hours-event_hours[0]).to(tokens.dtype)
        state = self.layer.replay_observations(initial, body[None], organs[None], presence[None], relative[None])
        gap = (torch.as_tensor(anchor_hour, device=tokens.device, dtype=event_hours.dtype)-event_hours[-1]).to(tokens.dtype).reshape(1)
        state = self.layer.advance(state, gap)
        grid = torch.arange(int(torch.ceil(actual_hours.max()).item())+1, device=tokens.device, dtype=tokens.dtype)
        path = self.layer.rollout(state, grid)
        values, latents = [], []
        for index, owner in enumerate(self.owners):
            t = actual_hours[0,index]; left, right = t.floor().long(), t.ceil().long()
            future = (1-(t-left))*path['organs'][:,left,owner]+(t-left)*path['organs'][:,right,owner]
            change = future-state.organs[:,owner]
            features = torch.cat((state.organs[:,owner], change, self.layer._time_features(actual_hours[:,index]),
                current[:,index:index+1], current_mask[:,index:index+1]), -1)
            values.append(base_value[:,index]+self.value_heads[index](features)[:,0])
            latents.append(base_latent[:,index]+self.latent_heads[index](features))
        return dict(value=torch.stack(values,1), future_latent=torch.stack(latents,1), state=state)

