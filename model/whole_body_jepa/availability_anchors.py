"""Rebuild glucose queries and labels on information-availability clocks."""
import numpy as np
import pandas as pd

from whole_body_jepa.raw_lab_replay import canonical_events


def rebuild(events):
    """One query per available timestamp; no outcome-dependent query selection.

    Latest measured (not latest arriving) known result anchors persistence.
    Future truth is the nearest measurement 5-7h AFTER the new query time.
    Final recorded revisions define retrospective truth, never current inputs.
    """
    records = []
    for stay, ev in canonical_events(events).groupby('stay_id', sort=False):
        final = ev.drop_duplicates('source_id', keep='last')
        for t in ev.available_at.drop_duplicates():
            known = ev.loc[ev.available_at <= t].drop_duplicates('source_id', keep='last')
            known = known.sort_values(['measured_at', 'available_at', 'source_id'])
            current = known.iloc[-1]
            age = (t-current.measured_at).total_seconds()/3600
            row = dict(stay_id=stay, t=t, current=float(current.value),
                       measurement_age=age, current_event_time=current.measured_at,
                       current_available_at=current.available_at,
                       source_id=current.source_id, truth=np.nan,
                       future_event_time=pd.NaT, future_available_at=pd.NaT,
                       actual_delta_t=np.nan, fresh=age <= 6)
            # Fixed, past-only summary. Missing history is explicit, never fabricated.
            for hours in (6, 24):
                history = known.loc[known.measured_at >= t-pd.Timedelta(hours=hours)]
                values = history.value.to_numpy(float)
                distinct = history.drop_duplicates('measured_at', keep='last')
                row[f'mean_{hours}'] = float(values.mean()) if len(values) else row['current']
                row[f'std_{hours}'] = float(values.std()) if len(values) else 0.
                row[f'history_valid_{hours}'] = float(len(distinct) >= 2)
                row[f'slope_{hours}'] = 0.
                if len(distinct) >= 2:
                    dt = (distinct.measured_at-distinct.measured_at.iloc[0]).dt.total_seconds().to_numpy()/3600
                    row[f'slope_{hours}'] = float(np.dot(dt-dt.mean(), distinct.value-distinct.value.mean()) / np.dot(dt-dt.mean(), dt-dt.mean()))
            delta = (final.measured_at-t).dt.total_seconds()/3600
            future = final.loc[delta.between(5, 7)].copy()
            if len(future):
                future['distance'] = (delta.loc[future.index]-6).abs()
                label = future.sort_values(['distance', 'measured_at', 'source_id']).iloc[0]
                row.update(truth=float(label.value), future_event_time=label.measured_at,
                           future_available_at=label.available_at,
                           actual_delta_t=(label.measured_at-t).total_seconds()/3600)
            records.append(row)
    return pd.DataFrame(records)
