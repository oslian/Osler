"""Apply the established native training transforms to compact query storage."""
import numpy as np

from whole_body_jepa.availability_inputs import apply_numeric_rescale, restrict_input_support
from whole_body_jepa.history_event_input import prepare_events, append_events


def transformed_storage(storage, centers, scales, allowed, module_ids, fit_queries, *, own):
    """Return transformed storage without modifying its raw arrays or histories."""
    from copy import copy

    n = len(allowed)
    module_count = storage.query['presence'].shape[1]
    x = storage.input
    events_view = dict(input=x, current=x[:, :n], mask=x[:, n:2*n],
        ages=x[:, 2*n:3*n], presence=x[:, 3*n+1:3*n+1+module_count])
    fit_rows = np.unique(np.concatenate([np.array([0]),
        *[storage.sequences[i] for i in fit_queries]]))
    event_features, event_center, event_scale = prepare_events(storage.event_values, fit_rows)
    if own:
        event_features[:] = 0
    events_view = append_events(restrict_input_support(
        apply_numeric_rescale(events_view, centers, scales), allowed, module_ids), event_features)
    query = restrict_input_support(apply_numeric_rescale(storage.query, centers, scales), allowed, module_ids)
    result = copy(storage)
    result.input = events_view['input']
    result.query = query
    return result, event_center, event_scale
