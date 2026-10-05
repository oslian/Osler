"""As-of event summaries that retain excursions between fixed history lags."""
import numpy as np
import pandas as pd

from whole_body_jepa.raw_lab_replay import canonical_events

WINDOWS=(6.,24.)
NUMERIC_FEATURES=6
FEATURES=('mean','std','min','max','last_delta','slope','log_count','span_fraction','slope_valid')


def event_windows(queries, events, variables, medians, scales):
    """Use latest known event versions only; no future revisions or labels."""
    if queries[['stay_id','t']].isna().any().any():raise ValueError('missing query identity/time')
    if len(set(variables))!=len(variables):raise ValueError('duplicate variable')
    if not np.isfinite(scales).all() or np.any(np.asarray(scales)<=0):raise ValueError('invalid scale')
    queries=queries.reset_index(drop=True)
    out=np.zeros((len(queries),len(variables),len(WINDOWS),len(FEATURES)),dtype=np.float64)
    groups=dict(tuple(canonical_events(events).groupby('stay_id',sort=False)))
    positions={v:i for i,v in enumerate(variables)}
    for stay,q in queries.groupby('stay_id',sort=False):
        raw=groups.get(stay)
        if raw is None:continue
        for row,t in zip(q.index,q.t):
            known=raw.loc[raw.available_at<=t].drop_duplicates('source_id',keep='last')
            known=known.loc[known.measured_at>=t-pd.Timedelta(hours=max(WINDOWS))]
            for var,ev in known.groupby('variable',sort=False):
                if var not in positions:continue
                col=positions[var]
                # A same-time disagreement is not an observed excursion.
                consistent=ev.groupby('measured_at')['value'].transform('nunique').eq(1)
                if 'quality_valid' in ev:
                    quality=ev.quality_valid.fillna(False).astype(bool)
                    consistent &= quality.groupby(ev.measured_at).transform('all')
                ev=ev.loc[consistent]
                if ev.empty:continue
                ev=ev.sort_values(['measured_at','available_at','source_id'])
                age=(t-ev.measured_at).dt.total_seconds().to_numpy()/3600
                z=(ev.value.to_numpy(float)-medians[col])/scales[col]
                for w,hours in enumerate(WINDOWS):
                    valid=(age>=0)&(age<=hours);v=z[valid];a=age[valid]
                    if not len(v):continue
                    span=float(a.max()-a.min());can_slope=span>1e-9
                    times=-a;centered=times-times.mean()
                    slope=float(np.dot(centered,v-v.mean())/np.dot(centered,centered)) if can_slope else 0.
                    previous=np.flatnonzero(a>a[-1])
                    change=float(v[-1]-v[previous[-1]]) if len(previous) else 0.
                    out[row,col,w]=[v.mean(),v.std(),v.min(),v.max(),change,slope,
                                     np.log1p(len(v)),span/hours,float(can_slope)]
    return out


def select_event_features(features, allowed, col, mode):
    if mode not in ('own','full','support'):raise ValueError('unknown feature mode')
    x=features.copy();x[:,~np.asarray(allowed,bool)]=0
    if mode=='own':x=x[:,col:col+1]
    if mode=='support':
        own=x[:,col,:,:NUMERIC_FEATURES].copy()
        x[:,:,:,:NUMERIC_FEATURES]=0;x[:,col,:,:NUMERIC_FEATURES]=own
    return x.reshape(len(x),-1)
