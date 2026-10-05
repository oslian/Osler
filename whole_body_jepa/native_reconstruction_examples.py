"""Current reconstruction examples rebuilt at each query's knowledge time."""
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.native_history_schedule import OBSERVATION_KINDS, conflict_checked_history_schedule
from whole_body_jepa.native_event_tokens import encode_group
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.history_event_input import append_events


def reconstruction_examples(events, queries, variables, medians, scales,
                            module_ids, module_count, max_age_hours=1., lookback_hours=48.):
    """Yield fresh targets arriving at each anchor, with fully masked histories.

    Units/availability must already be qualified by the source extractor.
    Inputs contain measurement events only, not administration intent or rates.
    """
    if not np.isfinite(max_age_hours) or max_age_hours<0:
        raise ValueError('finite nonnegative measurement age required')
    measured=events.loc[events.source_kind.isin(OBSERVATION_KINDS)]
    canonical,schedules=conflict_checked_history_schedule(measured,queries,lookback_hours)
    n=len(variables);encoded={}
    for position,groups in enumerate(schedules):
        if not groups:continue
        anchor=pd.Timestamp(queries.iloc[position]['t'])
        tokens=[];hours=[]
        for group in groups:
            key=tuple(group.tolist())
            if key not in encoded:
                part=canonical.iloc[group]
                token,_=encode_group(part,variables,medians,scales,module_ids,module_count,max_age_hours)
                hour=(pd.Timestamp(part.event_time.iloc[0])-pd.Timestamp('2100-01-01')).total_seconds()/3600
                token[3*n]=hour/24;token[-1]=hour/168
                encoded[key]=(token,hour)
            token,hour=encoded[key]
            tokens.append(token);hours.append(hour)
        # A stale carried value is never manufactured into an anchor label.
        if pd.Timestamp(canonical.iloc[groups[-1]].event_time.iloc[0])!=anchor:continue
        matrix=torch.from_numpy(append_events({'input':np.asarray(tokens,np.float32)},
            np.zeros((len(tokens),6),np.float32))['input'])
        times=torch.tensor(hours,dtype=torch.float64)
        for col in np.flatnonzero(np.asarray(tokens[-1])[n:2*n]>0):
            try:
                clean,clean_times=mask_native_target(matrix,times,variables,variables[col])
            except ValueError as exc:
                if str(exc)=='no independent observed history remains':continue
                raise
            yield dict(query_position=position,target_index=int(col),
                truth_normalized=float(tokens[-1][col]),tokens=clean,event_hours=clean_times,
                anchor_hour=float(hours[-1]))
