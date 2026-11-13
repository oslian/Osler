"""Masked supervision of the same persistent body, independent of forecasting."""
import torch
from torch import nn
from whole_body_jepa.nowcast import related_siblings


def mask_native_target(tokens, event_hours, variables, target):
    n=len(variables)
    if target not in variables or tokens.ndim!=2 or event_hours.shape!=(len(tokens),):
        raise ValueError('aligned native stream and known target required')
    hidden=[i for i,v in enumerate(variables) if v in ({target}|related_siblings(target))]
    clean=tokens.clone()
    clean[:,hidden]=0
    clean[:,[n+i for i in hidden]]=0
    clean[:,[2*n+i for i in hidden]]=7
    # Discard target-only events, not just their numbers. Presence is rebuilt
    # by encode_native_organ_observations from these remaining masks.
    keep=clean[:,n:2*n].bool().any(-1)
    if not keep.any():raise ValueError('no independent observed history remains')
    return clean[keep],event_hours[keep]


class NativeBodyReconstruction(nn.Module):
    def __init__(self, trajectory):
        super().__init__()
        self.trajectory=trajectory
        self.readout=nn.Linear(trajectory.jepa.latent,len(trajectory.variables))

    def forward(self,tokens,event_hours,anchor_hour,target):
        model=self.trajectory
        # Validate the original stream before masking can discard invalid events.
        if tokens.ndim!=2 or event_hours.shape!=(len(tokens),) or not len(tokens):
            raise ValueError('one nonempty aligned event stream required')
        anchor=torch.as_tensor(anchor_hour,device=event_hours.device,dtype=event_hours.dtype)
        if anchor.ndim!=0 or not torch.isfinite(anchor):
            raise ValueError('finite scalar anchor required')
        if (not torch.isfinite(event_hours).all() or (event_hours>anchor).any()
                or (event_hours[1:]<=event_hours[:-1]).any()):
            raise ValueError('finite strictly ordered pre-anchor times required')
        clean,times=mask_native_target(tokens,event_hours,model.variables,target)
        shape=(1,len(model.targets));zero=tokens.new_zeros(shape)
        out=model(clean,times,anchor_hour,zero,zero,zero,zero,
            tokens.new_zeros((*shape,model.jepa.latent)))
        return self.readout(out['state'].body)[:,model.variables.index(target)]
