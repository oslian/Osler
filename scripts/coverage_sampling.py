"""Weighted ordering without repeats until each target pool has been visited."""
import numpy as np


def coverage_indices(weights, rng, draws, pending):
    weights = np.asarray(weights, dtype=float)
    pending = np.asarray(pending, dtype=np.int64)
    if (weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all()
            or (weights <= 0).any() or not isinstance(draws, int) or draws < 1):
        raise ValueError('Invalid sampling request')
    if (pending.ndim != 1 or len(np.unique(pending)) != len(pending)
            or (pending < 0).any() or (pending >= len(weights)).any()):
        raise ValueError('Invalid remaining cycle')
    blocks = []
    remaining = draws
    while remaining:
        if not len(pending):
            pending = rng.choice(len(weights), size=len(weights), replace=False,
                                 p=weights / weights.sum())
        take = min(remaining, len(pending))
        blocks.append(pending[:take])
        pending = pending[take:]
        remaining -= take
    return np.concatenate(blocks), pending.copy()
