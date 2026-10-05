"""Separate event channels in the existing neural history observation input."""
import numpy as np
import torch
from torch import nn


def prepare_events(values, fit_rows):
    values=np.asarray(values,dtype=np.float64)
    observed=np.isfinite(values)
    center=np.zeros(values.shape[1]);scale=np.ones(values.shape[1])
    for j in range(values.shape[1]):
        sample=values[fit_rows,j];sample=sample[np.isfinite(sample)]
        if len(sample):
            center[j]=np.median(sample);scale[j]=max(float(sample.std()),1.)
    numbers=np.where(observed,(values-center)/scale,0.)
    return np.column_stack([numbers,observed]).astype(np.float32),center,scale


def append_events(arrays,events):
    if len(events)!=len(arrays['input']) or not np.isfinite(events).all():
        raise ValueError('finite aligned event history required')
    result=dict(arrays)
    # The filter reads its clock from the last input coordinate.
    result['input']=np.column_stack([arrays['input'][:,:-1],events,arrays['input'][:,-1:]]).astype(np.float32)
    return result


def expand_history_projection(model,width):
    old=model.belief_filter.observation_projection[0]
    new=nn.Linear(old.in_features+width,old.out_features,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_();new.weight[:,:old.in_features-1].copy_(old.weight[:,:-1])
        new.weight[:,-1:].copy_(old.weight[:,-1:]);new.bias.copy_(old.bias)
    model.belief_filter.observation_projection[0]=new
    model.input_dim=old.in_features+width
