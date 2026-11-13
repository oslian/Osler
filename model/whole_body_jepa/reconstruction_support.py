"""Mask-support audit using the same source-version replay as reconstruction."""
import numpy as np
import pandas as pd

from whole_body_jepa.native_history_schedule import OBSERVATION_KINDS, conflict_checked_history_schedule
from whole_body_jepa.nowcast import related_siblings


def reconstruction_support(events, queries, variables, max_age_hours=1., lookback_hours=48.):
    if not np.isfinite(max_age_hours) or max_age_hours < 0:
        raise ValueError('finite nonnegative measurement age required')
    if not set(queries.variable) <= set(variables):
        raise ValueError('Unknown requested target')
    measured = events.loc[events.source_kind.isin(OBSERVATION_KINDS)]
    canonical, schedules = conflict_checked_history_schedule(measured, queries, lookback_hours)
    output = []
    for query, groups in zip(queries.itertuples(index=False), schedules):
        status = 'no_fresh_anchor_label'; observed_variables = event_count = 0
        if groups:
            history = canonical.iloc[np.concatenate(groups)].copy()
            history = history.loc[history.variable.isin(variables)]
            keys = ['event_time','variable']
            latest = history.groupby(keys).measured_at.transform('max')
            current = history.loc[history.measured_at.eq(latest)].copy()
            current['valid'] = np.isfinite(current.value)
            if 'quality_valid' in current:
                current['valid'] &= current.quality_valid.fillna(False).astype(bool)
            valid = current.groupby(keys).valid.transform('all')
            valid &= current.groupby(keys).value.transform('nunique').eq(1)
            age = (current.event_time-current.measured_at).dt.total_seconds()/3600
            observed = current.loc[valid & age.between(0,max_age_hours)]
            label = observed.variable.eq(query.variable) & observed.event_time.eq(query.t)
            if label.any():
                independent = observed.loc[~observed.variable.isin({query.variable}|related_siblings(query.variable))]
                observed_variables = int(independent.variable.nunique())
                event_count = int(independent.event_time.nunique())
                status = 'accepted' if event_count else 'no_independent_history'
        output.append(dict(status=status, independent_variables=observed_variables,
                           independent_event_groups=event_count))
    return pd.DataFrame(output, columns=['status','independent_variables','independent_event_groups'])
