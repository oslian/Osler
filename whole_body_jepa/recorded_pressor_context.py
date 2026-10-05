"""Retrospective recorded-rate context; never asserts bedside availability."""
import numpy as np
import pandas as pd

DRUGS = ('norepinephrine', 'epinephrine', 'phenylephrine', 'dopamine', 'vasopressin')
DIMENSIONS = ('mass_mg_per_hour', 'mass_mcg_per_kg_min', 'drug_units_per_hour')


def recorded_snapshots(anchors, events, lookback_minutes=360):
    """Do not sum drugs/dimensions, carry invalid entries forward, or infer delivery.

    Anchors have stay and offset; events have stay, id, drug, dimension, offset,
    rate. Invalid rate/unit rows MUST remain in events to block stale values.
    The caller must not substitute event time for independently known availability.
    """
    if lookback_minutes <= 0:
        raise ValueError('positive lookback required')
    required = ['stay', 'id', 'drug', 'dimension', 'offset', 'rate']
    if events[required[:3] + ['offset']].isna().any().any():
        raise ValueError('identity and recorded time required')
    if not np.isfinite(events.offset.to_numpy(float)).all():
        raise ValueError('finite recorded time required')
    if not np.isfinite(anchors.offset.to_numpy(float)).all():
        raise ValueError('finite anchor time required')
    # Re-documentation of an identical source row is not a new administration.
    events = events[required].drop_duplicates()
    if events.duplicated(['stay', 'id']).any():
        raise ValueError('conflicting source identity')
    output = np.full((len(anchors), len(DRUGS), len(DIMENSIONS)), np.nan)
    age = np.full((len(anchors), len(DRUGS)), np.nan)
    status = np.full((len(anchors), len(DRUGS)), 'missing', dtype='<U24')
    groups = {(s, d): g.sort_values('offset') for (s, d), g in events.groupby(['stay', 'drug'])}
    for i, anchor in enumerate(anchors.itertuples(index=False)):
        for j, drug in enumerate(DRUGS):
            group = groups.get((anchor.stay, drug))
            if group is None: continue
            known = group[(group.offset <= anchor.offset) &
                          (group.offset >= anchor.offset - lookback_minutes)]
            if known.empty: continue
            last_time = known.offset.max()
            last = known[known.offset.eq(last_time)]
            age[i, j] = (anchor.offset - last_time) / 60
            rates = pd.to_numeric(last.rate, errors='coerce').to_numpy(float)
            if (not np.isfinite(rates).all() or (rates < 0).any()
                    or not last.dimension.isin(DIMENSIONS).all()):
                status[i, j] = 'invalid_latest'
                continue
            if last.dimension.nunique() != 1 or last.rate.nunique() != 1:
                status[i, j] = 'conflicting_latest'
                continue
            k = DIMENSIONS.index(last.dimension.iloc[0])
            output[i, j, k] = rates[0]
            status[i, j] = 'recorded_only'
    return dict(rate=output, known=np.isfinite(output), age_hours=age, status=status,
                online_eligible=False, independent_availability_verified=False)
