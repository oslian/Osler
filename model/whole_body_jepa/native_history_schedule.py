"""Past-only native event groups for a bounded patient history encoder."""
import numpy as np
import pandas as pd

OBSERVATION_KINDS = ('lab_available', 'bedside_available')


def native_history_schedule(events,queries,lookback_hours=48.):
    """Return canonical events and per-query simultaneous-event index groups.

    event_time is a caller-declared timestamp, not inferred availability.
    Source kind namespaces IDs. Query times never create observation updates.
    """
    if not np.isfinite(lookback_hours) or lookback_hours<=0:
        raise ValueError('positive finite lookback required')
    required=['stay_id','source_kind','source_id','event_time']
    e=events.copy();e['event_time']=pd.to_datetime(e.event_time,errors='raise')
    q=queries.copy();q['t']=pd.to_datetime(q.t,errors='raise')
    if e[required].isna().any().any() or q[['stay_id','t']].isna().any().any():
        raise ValueError('event/query identities and timestamps required')
    for _,part in e.groupby(required,dropna=False,sort=False):
        if len(part.drop_duplicates())!=1:
            raise ValueError('conflicting event version')
    e=e.drop_duplicates().sort_values(required,kind='stable').reset_index(drop=True)
    by_stay={stay:part.sort_values('event_time',kind='stable') for stay,part in e.groupby('stay_id',sort=False)}
    schedules=[];window=pd.Timedelta(hours=lookback_hours)
    for query in q.itertuples():
        part=by_stay.get(query.stay_id,e.iloc[:0])
        part=part.loc[(part.event_time>query.t-window)&(part.event_time<=query.t)]
        schedules.append(tuple(group.index.to_numpy(dtype=np.int64) for _,group in part.groupby('event_time',sort=True)))
    return e,schedules


def conflict_checked_history_schedule(events, queries, lookback_hours=48.):
    """Rebuild each query history from source versions known at that query.

    This is query-time replay, not an incremental update of a cached state.
    Earlier query outputs remain untouched when later conflicting evidence arrives.
    Non-measurement events retain the original contract.
    """
    e, _ = native_history_schedule(events, queries, lookback_hours)
    labs = e.source_kind.isin(OBSERVATION_KINDS)
    required = ['variable', 'measured_at', 'value']
    if any(name not in e for name in required):
        raise ValueError('lab measurement identity and value required')
    e['measured_at'] = pd.to_datetime(e.measured_at, errors='raise')
    if e.loc[labs, required].isna().any().any() or not np.isfinite(e.loc[labs, 'value']).all():
        raise ValueError('finite lab values and measurement timestamps required')
    if (e.loc[labs, 'measured_at'] > e.loc[labs, 'event_time']).any():
        raise ValueError('measurement cannot occur after availability')
    identity = e.loc[labs].groupby(['stay_id', 'source_kind', 'source_id'])
    if (identity.variable.nunique() > 1).any() or (identity.measured_at.nunique() > 1).any():
        raise ValueError('lab source identity changes across revisions')
    by_stay = {s: g.sort_values('event_time', kind='stable') for s, g in e.groupby('stay_id')}
    schedules = []
    window = pd.Timedelta(hours=lookback_hours)
    for query in queries.itertuples():
        time = pd.Timestamp(query.t)
        known = by_stay.get(query.stay_id, e.iloc[:0])
        known = known.loc[known.event_time <= time]
        other = known.loc[~known.source_kind.isin(OBSERVATION_KINDS)]
        current = known.loc[known.source_kind.isin(OBSERVATION_KINDS)].drop_duplicates(
            ['source_kind', 'source_id'], keep='last')
        consistent = current.groupby(['variable', 'measured_at']).value.transform('nunique').eq(1)
        # Resolve conflicts before windowing: an older known source can invalidate
        # a later report even when the older arrival is outside the replay window.
        accepted = pd.concat([other, current.loc[consistent]])
        accepted = accepted.loc[accepted.event_time > time - window]
        schedules.append(tuple(g.index.to_numpy(dtype=np.int64)
                               for _, g in accepted.groupby('event_time', sort=True)))
    return e, schedules
