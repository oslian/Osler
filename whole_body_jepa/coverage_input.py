"""Public canonical-event input bridge for the frozen coverage candidate."""

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.contracts import VARIABLE_UNITS
from whole_body_jepa.history_event_input import append_events
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.native_event_tokens import encode_group
from whole_body_jepa.native_history_schedule import OBSERVATION_KINDS, conflict_checked_history_schedule


REQUIRED_COLUMNS = ('stay_id', 'source_kind', 'source_id', 'event_time', 'measured_at',
                    'variable', 'value', 'unit', 'quality_valid')
ORIGIN = pd.Timestamp('2100-01-01')


def read_events(path):
    events = pd.read_csv(path, dtype={'stay_id': str, 'source_id': str})
    if 'quality_valid' in events:
        quality = events.quality_valid.astype(str).str.lower().map({'true': True, 'false': False})
        if quality.isna().any():
            raise ValueError('quality_valid must be true or false')
        events['quality_valid'] = quality.astype(bool)
    return events


def build_input(events, checkpoint, stay_id, anchor, target):
    """No target truth is required, fabricated, or fed to the model."""
    if checkpoint['format'] != 'masked-current-reconstruction-coverage-v1':
        raise ValueError('Wrong checkpoint format')
    variables = checkpoint['variables']
    if target not in checkpoint['targets']:
        raise ValueError('Target is not trained in this reconstruction checkpoint')
    if not set(REQUIRED_COLUMNS) <= set(events):
        raise ValueError('Required CSV columns: ' + ','.join(REQUIRED_COLUMNS))
    anchor = pd.Timestamp(anchor)
    if pd.isna(anchor) or anchor.tzinfo is not None:
        raise ValueError('A finite timezone-naive anchor in the source time basis is required')
    selected = events.loc[events.stay_id.astype(str).eq(str(stay_id)), list(REQUIRED_COLUMNS)].copy()
    if selected.empty:
        raise ValueError('Selected stay has no events')
    for column in ('event_time', 'measured_at'):
        selected[column] = pd.to_datetime(selected[column], errors='raise')
        if selected[column].isna().any() or selected[column].dt.tz is not None:
            raise ValueError('Finite timezone-naive event and measurement times required')
    selected = selected.loc[selected.event_time <= anchor].copy()
    if selected.empty:
        raise ValueError('No events known at anchor')
    if selected[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError('Missing event fields; omit unmeasured values, never replace them with zero')
    if not selected.source_kind.isin(OBSERVATION_KINDS).all():
        raise ValueError('This checkpoint accepts measurement events only, not treatments')
    if not selected.variable.isin(variables).all():
        raise ValueError('Unknown variable; use the checkpoint schema command')
    if not pd.api.types.is_bool_dtype(selected.quality_valid):
        raise ValueError('quality_valid must contain booleans')
    selected['value'] = pd.to_numeric(selected.value, errors='raise')
    if not np.isfinite(selected.value).all():
        raise ValueError('Observed values must be finite; omit missing measurements')
    if not selected.unit.eq(selected.variable.map(VARIABLE_UNITS)).all():
        raise ValueError('Unit mismatch; convert and qualify source units before this bridge')
    queries = pd.DataFrame([dict(stay_id=selected.stay_id.iloc[0], t=anchor)])
    canonical, schedules = conflict_checked_history_schedule(selected, queries, lookback_hours=48.)
    tokens, times = [], []
    n = len(variables)
    module_ids = checkpoint['schema']['state_dict']['variable_module_ids'].cpu().numpy()
    module_count = len(checkpoint['schema']['modules'])
    for indices in schedules[0]:
        group = canonical.iloc[indices]
        token, _ = encode_group(group, variables, np.zeros(n), np.ones(n), module_ids, module_count, 1.)
        hour = (pd.Timestamp(group.event_time.iloc[0]) - ORIGIN).total_seconds() / 3600
        token[3*n], token[-1] = hour/24., hour/168.
        tokens.append(token)
        times.append(hour)
    if not tokens:
        raise ValueError('No events in the 48-hour lookback')
    matrix = torch.from_numpy(append_events({'input': np.asarray(tokens, np.float32)},
                                            np.zeros((len(tokens), 6), np.float32))['input'])
    hours = torch.tensor(times, dtype=torch.float64)
    matrix, hours = mask_native_target(matrix, hours, variables, target)
    scaler = checkpoint['scaler']
    location = torch.as_tensor(scaler['location'], dtype=matrix.dtype)
    scale = torch.as_tensor(scaler['scale'], dtype=matrix.dtype)
    counts = torch.as_tensor(scaler['counts'])
    if (location.shape != (n,) or scale.shape != (n,) or counts.shape != (n,)
            or not torch.isfinite(location).all() or not torch.isfinite(scale).all()
            or not (scale > 0).all() or counts[variables.index(target)] <= 0):
        raise ValueError('Invalid checkpoint normalization')
    mask = matrix[:, n:2*n].bool()
    if (mask & (counts <= 0)).any():
        raise ValueError('Observed variable has no fit normalization support')
    matrix[:, :n] = torch.where(mask, (matrix[:, :n] - location) / scale, 0.)
    if not torch.isfinite(matrix).all():
        raise ValueError('Nonfinite encoded input')
    return dict(tokens=matrix, event_hours=hours,
                anchor_hour=float((anchor-ORIGIN).total_seconds()/3600), target=target)
