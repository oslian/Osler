"""Candidate-only separation of observation content from update scheduling."""
import torch


def target_projection_input(value, variable_count, target_index, owner_index):
    """Keep target and clocks, not other support; do not alter active-step masks.

    Applied only at observation_projection, after the recurrent filter has read
    the original event activation and elapsed time. Missing ages retain the
    native seven-day sentinel. The six appended treatment fields are removed.
    """
    n = variable_count
    modules = value.shape[-1] - 3 * n - 8
    if not 0 <= target_index < n or not 0 <= owner_index < modules:
        raise ValueError('invalid native projection schema')
    result = torch.zeros_like(value)
    result[..., 2*n:3*n] = 7
    for start in (0, n, 2*n):
        result[..., start+target_index] = value[..., start+target_index]
    result[..., 3*n] = value[..., 3*n]
    result[..., 3*n+1+owner_index] = value[..., n+target_index]
    result[..., -1] = value[..., -1]
    return result


def install_projection_control(model, target_index):
    """Return a removable hook; checkpoint consumers must install it explicitly."""
    n = len(model.variable_module_ids)
    owner = int(model.variable_module_ids[target_index])
    return model.belief_filter.observation_projection.register_forward_pre_hook(
        lambda module, args: (target_projection_input(args[0], n, target_index, owner),)
    )
