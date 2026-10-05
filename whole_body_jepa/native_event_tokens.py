"""New-event observations, never a repeated carry-forward snapshot."""
import numpy as np
import pandas as pd

from whole_body_jepa.native_history_schedule import OBSERVATION_KINDS


def encode_group(group,variables,medians,scales,module_ids,module_count,max_age_hours):
    if group.event_time.nunique()!=1 or group.stay_id.nunique()!=1:
        raise ValueError('one simultaneous single-stay group required')
    n=len(variables);positions={v:i for i,v in enumerate(variables)}
    current=np.zeros(n,np.float32);mask=np.zeros(n,np.float32);ages=np.full(n,7.,np.float32)
    time=pd.Timestamp(group.event_time.iloc[0])
    labs=group.loc[group.source_kind.isin(OBSERVATION_KINDS)]
    for variable,part in labs.groupby('variable',sort=False):
        if variable not in positions:continue
        part=part.loc[part.measured_at.notna()]
        if part.empty:continue
        latest=part.loc[part.measured_at.eq(part.measured_at.max())]
        if 'quality_valid' in latest and not latest.quality_valid.fillna(False).all():continue
        values=latest.value.to_numpy(float)
        age=(time-pd.Timestamp(latest.measured_at.iloc[0])).total_seconds()/3600
        if not np.isfinite(values).all() or len(np.unique(values))!=1 or not 0<=age<=max_age_hours:continue
        i=positions[variable];current[i]=(values[0]-medians[i])/scales[i];mask[i]=1.;ages[i]=age/24.
    presence=np.array([mask[np.asarray(module_ids)==i].any() for i in range(module_count)],np.float32)
    hours=(time-pd.Timestamp('2100-01-01')).total_seconds()/3600
    # Keep the old clock convention for the matched diagnostic.
    clock=np.clip(hours,0,168)
    token=np.concatenate([current,mask,ages,[clock/24.],presence,[clock/168.]]).astype(np.float32)
    rates=group.loc[group.source_kind.eq('insulin_record')]
    event=np.array([np.log1p(len(rates)),np.nan,np.nan],np.float64)
    if len(rates):
        event[1]=0.;values=rates.value.to_numpy(float)
        if np.isfinite(values).all() and len(np.unique(values))==1:event[2]=values[0]
    return token,event
