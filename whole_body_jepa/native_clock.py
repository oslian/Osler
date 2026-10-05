"""Preserve elapsed-time coordinates without changing observed physiology."""
import numpy as np


def restore_clock(tokens,token_ids,event_hours,n_state):
    x=np.asarray(tokens);ids=np.asarray(token_ids);hours=np.asarray(event_hours,float)
    if ids.ndim!=1 or hours.shape!=ids.shape or not np.isfinite(hours).all():
        raise ValueError('finite aligned event times required')
    if (ids<0).any() or (ids>=len(x)).any():
        raise ValueError('invalid elapsed time or token index')
    order=np.argsort(ids,kind='stable');i=ids[order];h=hours[order]
    if len(i)>1 and ((np.diff(i)==0)&(np.diff(h)!=0)).any():
        raise ValueError('one token cannot represent multiple event times')
    if not np.array_equal(np.unique(ids),np.arange(len(x))):
        raise ValueError('every token needs an event time')
    result=x.copy();result[ids,3*n_state]=hours/24.;result[ids,-1]=hours/168.
    return result
