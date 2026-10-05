"""Auxiliary measured future labels at unchanged availability-safe queries."""
import numpy as np
import pandas as pd

from whole_body_jepa.availability_inputs import snapshots
from whole_body_jepa.raw_lab_replay import canonical_events


def future_pairs(queries, events, variables, *, horizon=6., tolerance=1., max_age=6.):
    if not (np.isfinite([horizon, tolerance, max_age]).all()
            and horizon > tolerance >= 0 and max_age >= 0):
        raise ValueError('invalid time policy')
    if len(set(variables)) != len(variables):
        raise ValueError('duplicate variable')
    if queries[['stay_id', 't']].isna().any().any():
        raise ValueError('query identity/time is missing')
    if queries.duplicated(['stay_id', 't']).any():
        raise ValueError('duplicate query')
    queries = queries.reset_index(drop=True)
    events = canonical_events(events)
    shape = (len(queries), len(variables))
    current = np.full(shape, np.nan)
    age = current.copy()
    future = current.copy()
    delta = current.copy()
    id_width=max(1,max((len(str(v)) for v in events.source_id),default=1))
    future_source_id=np.full(shape,'',dtype=f'U{id_width}')
    groups = dict(tuple(events.groupby('stay_id', sort=False)))
    hour_ns = pd.Timedelta(hours=1).value
    for stay, q in queries.groupby('stay_id', sort=False):
        ev = groups.get(stay, events.iloc[:0])
        order = q.sort_values('t')
        times = pd.DatetimeIndex(order.t).as_unit('ns')
        values, ages = snapshots(ev, times, variables, max_age)
        rows = order.index.to_numpy()
        current[rows], age[rows] = values, ages
        # Final versions may supply truth, but snapshots use only known versions.
        final = ev.drop_duplicates('source_id', keep='last')
        by_variable = dict(tuple(final.groupby('variable', sort=False)))
        for column, variable in enumerate(variables):
            f = by_variable.get(variable)
            if f is None:
                continue
            # Distinct sources can disagree at one timestamp. ID order is not
            # evidence for choosing a physiological truth.
            f = f.copy()
            f['_conflicting'] = f.groupby('measured_at')['value'].transform('nunique') > 1
            if 'quality_valid' in f:
                quality = f.quality_valid.fillna(False).astype(bool)
                f['_conflicting'] |= ~quality.groupby(f.measured_at).transform('all')
            f = f.sort_values(['measured_at', 'source_id']).drop_duplicates('measured_at')
            measured = pd.DatetimeIndex(f.measured_at).as_unit('ns').asi8
            desired = times.asi8 + round(horizon * hour_ns)
            right = np.searchsorted(measured, desired).clip(0, len(f)-1)
            left = (np.searchsorted(measured, desired)-1).clip(0, len(f)-1)
            # Equal distance uses the earlier draw, matching glucose rebuild.
            selected = np.where(abs(measured[left]-desired) <= abs(measured[right]-desired), left, right)
            actual = (measured[selected]-times.asi8)/hour_ns
            valid = ((abs(actual-horizon) <= tolerance) & np.isfinite(values[:, column])
                     & ~f['_conflicting'].to_numpy(bool)[selected])
            future[rows[valid], column] = f.value.to_numpy(float)[selected[valid]]
            delta[rows[valid], column] = actual[valid]
            future_source_id[rows[valid],column]=f.source_id.astype(str).to_numpy()[selected[valid]]
    return dict(current=current, measurement_age=age, future=future,
                actual_delta_t=delta, paired=np.isfinite(future) & np.isfinite(current),
                future_source_id=future_source_id)
