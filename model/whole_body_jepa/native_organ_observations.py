"""Local anatomical observation proposals from the same JEPA token weights."""
import torch
from osler_jepa.organ_schema import CANONICAL_ORGAN_VARIABLES, assert_disjoint_contracts


def encode_native_organ_observations(model, variables, tokens):
    """Encode event groups, not patient posterior states or physiological truth.

    Organ proposals pool pre-mixer tokens to prevent non-owner observations
    entering a local correction. Body proposals use the existing JEPA context
    encoder on each event separately; persistent memory is maintained by the
    caller's single state layer. Treatment fields require a separate contract
    and must be zero for this measurement-only bridge.
    """
    n = len(variables)
    if n != model.n_state or tokens.ndim != 2 or tokens.shape[-1] != model.input_dim:
        raise ValueError('native token schema must match JEPA')
    assert_disjoint_contracts(CANONICAL_ORGAN_VARIABLES)
    organs = tuple(CANONICAL_ORGAN_VARIABLES)
    mapped = {v for group in CANONICAL_ORGAN_VARIABLES.values() for v in group}
    if set(variables)-mapped:
        raise ValueError('explicit anatomical mapping required for every variable')
    if (tokens[:, -7:-1] != 0).any():
        raise ValueError('measurement-only bridge requires zero treatment fields')
    mask = tokens[:, n:2*n]
    if not torch.isfinite(mask).all() or ((mask != 0) & (mask != 1)).any():
        raise ValueError('binary observation masks required')
    observed = mask.bool()
    current, ages = tokens[:, :n], tokens[:, 2*n:3*n]
    if not torch.isfinite(current[observed]).all() or not torch.isfinite(ages[observed]).all():
        raise ValueError('present measurement and age must be finite')
    if (ages[observed] < 0).any():
        raise ValueError('measurement cannot precede its availability age')
    current = torch.where(observed, current, torch.zeros_like(current))
    ages = torch.where(observed, ages, torch.full_like(ages, 7))
    clean = tokens.clone(); clean[:, :n] = current; clean[:, 2*n:3*n] = ages
    presence = (mask[:, None, :]*model.variable_module_membership[None]).sum(-1).gt(0).to(mask.dtype)
    clean[:, 3*n+1:3*n+1+model.module_count] = presence
    features = model.token_projection(torch.stack((current, mask, ages), dim=-1))
    features = features+model.variable_embedding[None]+model.module_token_embedding(model.variable_module_ids)[None]
    membership = mask.new_tensor([[v in CANONICAL_ORGAN_VARIABLES[o] for v in variables] for o in organs])
    weights = membership[None]*mask[:, None, :]
    counts = weights.sum(-1, keepdim=True)
    proposals = torch.einsum('bnd,bon->bod', features, weights)/counts.clamp_min(1)
    organ_presence = counts[..., 0].gt(0).to(mask.dtype)
    body = model.encode_context_components(clean[:, None, :], current, mask, ages, presence)[0]
    body = torch.where(organ_presence.any(-1, keepdim=True), body, torch.zeros_like(body))
    return body, proposals, organ_presence
