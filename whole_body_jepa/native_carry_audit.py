"""Direct mixing-path retention, explicitly excluding recurrent innovation."""
import torch


def post_target_direct_carry(gates, active, target_observed):
    if gates.ndim != 3 or active.shape != gates.shape[:2] or target_observed.shape != active.shape:
        raise ValueError('aligned batch/step gates and masks required')
    if not torch.isfinite(gates).all() or ((gates < 0) | (gates > 1)).any():
        raise ValueError('gates must be probabilities')
    if (target_observed & ~active).any():
        raise ValueError('target observation must activate update')
    steps=torch.arange(gates.shape[1],device=gates.device)[None,:]
    last=torch.where(target_observed,steps,-1).max(dim=1).values
    included=active & (steps>last[:,None]) & (last[:,None]>=0)
    factor=torch.where(included[...,None],1-gates,torch.ones_like(gates)).prod(dim=1)
    return factor, included.sum(dim=1), last>=0
