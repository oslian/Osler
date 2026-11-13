"""Training-matched numeric ablation preserving native observation support."""
from copy import copy


def without_other_numbers(storage, target_index):
    """Keep target values, timing and masks; zero other normalized numbers.

    Expects transformed native storage with six event fields before the clock.
    This retains measurement-process information, not a pure cadence placebo.
    """
    n = storage.query['current'].shape[1]
    if not 0 <= target_index < n:
        raise ValueError('target index outside state')
    result = copy(storage)
    result.input = storage.input.copy()
    result.query = {k:v.copy() for k,v in storage.query.items()}
    other = [i for i in range(n) if i != target_index]
    result.input[:, other] = 0
    result.input[:, -7:-4] = 0
    result.query['current'][:, other] = 0
    result.query['input'][:, other] = 0
    return result
